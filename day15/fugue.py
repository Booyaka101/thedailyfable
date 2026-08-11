"""fugue.py — fugue machinery on top of the counterpoint engine.

  - tonal_answer: subject -> answer with the classical 1/5-degree head
    adjustment, then real transposition.
  - diatonic_transpose / diatonic_invert: degree-space operations for
    sequences, inversions and major/minor-mode subject statements.
  - find_strettos: brute-force search over (time offset x transposition)
    for where the subject works against itself. The machine's discovery,
    not mine.
  - invertible: does a countersubject survive being flipped to the other
    side of the subject (invertible counterpoint at the octave)?
"""

from __future__ import annotations
from cp import (Note, Key, classify, score_events, transpose, shift,
                BAR, ic, CONSONANT, PERFECT)


# ------------------------------------------------------ degree-space ops

def scale_list(key):
    if key.mode == "major":
        degs = [0, 2, 4, 5, 7, 9, 11]
    else:
        degs = [0, 2, 3, 5, 7, 8, 10]   # natural minor; ficta handled by caller
    return [(key.tonic + d) % 12 for d in degs]


def pitch_to_degree(p, key):
    """(degree_index 0-6, octave, chromatic_offset). Chromatic notes map to
    the nearest scale pc below with offset +1 (covers raised 6/7 in minor)."""
    pcs = scale_list(key)
    pc = p % 12
    if pc in pcs:
        di = pcs.index(pc)
        off = 0
    else:
        # raised version of the scale step below
        pc0 = (pc - 1) % 12
        if pc0 in pcs:
            di = pcs.index(pc0)
            off = 1
        else:
            pc0 = (pc + 1) % 12
            di = pcs.index(pc0) if pc0 in pcs else 0
            off = -1
    # octave bookkeeping relative to tonic
    base = p - ((pc - key.tonic) % 12) if off == 0 else p - off - (((pc - off) - key.tonic) % 12)
    octv = base // 12
    return di, octv, off


def degree_to_pitch(di, octv, off, key):
    pcs = scale_list(key)
    step = (pcs[di % 7] - key.tonic) % 12
    return octv * 12 + key.tonic + step + off + 12 * (di // 7)


def diatonic_transpose(notes, key, steps, target_key=None):
    """Transpose by scale steps within key (or re-render into target_key
    at the same degrees when target_key given)."""
    tk = target_key or key
    out = []
    for n in notes:
        di, octv, off = pitch_to_degree(n.pitch, key)
        di2 = di + steps
        octv2 = octv + di2 // 7
        di2 %= 7
        out.append(Note(n.on, n.dur, degree_to_pitch(di2, octv2, off, tk)))
    return out


def diatonic_invert(notes, key, axis_pitch):
    """Melodic inversion in degree space around axis_pitch (a scale tone)."""
    ax_di, ax_oct, _ = pitch_to_degree(axis_pitch, key)
    ax = ax_oct * 7 + ax_di
    out = []
    for n in notes:
        di, octv, off = pitch_to_degree(n.pitch, key)
        lin = octv * 7 + di
        lin2 = 2 * ax - lin
        out.append(Note(n.on, n.dur, degree_to_pitch(lin2 % 7, lin2 // 7, 0, key)))
    return out


def rerender(notes, src_key, dst_key, dp_octave=0):
    """Same degrees, different key (subject in F major, G minor...)."""
    out = []
    for n in notes:
        di, octv, off = pitch_to_degree(n.pitch, src_key)
        # a raised-7 in minor becomes the diatonic 7 in major
        if dst_key.mode == "major" and off == 1 and di == 6:
            off = 0
        p = degree_to_pitch(di, octv, off, dst_key)
        out.append(Note(n.on, n.dur, p + 12 * dp_octave))
    return out


# ---------------------------------------------------------- tonal answer

def tonal_answer(subject, key, head_notes=2):
    """Classical tonal answer: within the head, degree 1 -> 5 (+7) and
    degree 5 -> 1 (+5); after the head, real transposition (+7)."""
    out = []
    for i, n in enumerate(subject):
        deg = (n.pitch - key.tonic) % 12
        if i < head_notes and deg == 7:      # 5 -> 1
            dp = 5
        elif i < head_notes and deg == 0:    # 1 -> 5
            dp = 7
        else:
            dp = 7
        out.append(Note(n.on, n.dur, n.pitch + dp))
    return out


# ------------------------------------------------------ stretto discovery

def try_pair(a, b):
    """Hard-rule fitness of two voices against each other (order-free)."""
    def mean_pitch(v):
        return sum(n.pitch * n.dur for n in v) / max(1, sum(n.dur for n in v))
    hiv, lov = (a, b) if mean_pitch(a) >= mean_pitch(b) else (b, a)
    ev, pr = classify([hiv, lov])
    viol = sum(1 for e in ev if e.kind == "VIOL")
    hard = sum(1 for p in pr if p.hard)
    s = score_events(ev, pr)
    return viol, hard, s


def find_strettos(subject, key, offsets_units, transpositions, dur_scale=1):
    """All (dt, dp) where a second subject entry can overlap the first.

    offsets_units: iterable of time offsets (16th units) for the second entry.
    transpositions: iterable of semitone transpositions to test.
    dur_scale: 2 for testing against the augmented subject.
    Returns rows sorted best-first:
      (viol, hard, score, dt, dp, overlap_units)
    """
    rows = []
    subj_len = max(n.off for n in subject)
    lead = subject
    if dur_scale != 1:
        lead = [Note(n.on * dur_scale, n.dur * dur_scale, n.pitch) for n in subject]
    for dt in offsets_units:
        for dp in transpositions:
            follower = [Note(n.on + dt, n.dur, n.pitch + dp) for n in subject]
            overlap = max(0, max(n.off for n in lead) - dt)
            if overlap <= 0:
                continue
            # keep only diatonic followers (subject tones must stay in key,
            # allowing the raised 6/7)
            base, var = key.scale_pcs()
            ok = all((n.pitch % 12) in (base | var) for n in follower)
            if not ok:
                continue
            viol, hard, s = try_pair(lead, follower)
            rows.append((viol, hard, round(s, 1), dt, dp, overlap))
    rows.sort(key=lambda r: (r[0], r[1], -r[5], -r[2]))
    return rows


# --------------------------------------------------------- invertibility

def invertible(cs, subject, shift_octaves=1):
    """Check countersubject at the octave: cs above subject AND cs below
    subject (cs shifted down by octaves). Returns (ok, report_up, report_down)."""
    up = try_pair(cs, subject)
    cs_low = transpose(cs, -12 * shift_octaves)
    down = try_pair(subject, cs_low)
    ok = up[0] == 0 and up[1] == 0 and down[0] == 0 and down[1] == 0
    return ok, up, down
