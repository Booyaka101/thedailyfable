"""cp.py — a counterpoint engine.

Time is integer 16th-note units (bar of 4/4 = 16 units, quarter = 4).
A Note is (on, dur, pitch) with MIDI pitch; a Voice is a sorted list of
non-overlapping Notes (gaps are rests).

The engine has three parts:
  1. classify(voices)  — walk the score and classify every vertical
     interval at every onset: consonance, or a *named* legal dissonance
     (passing, neighbor, suspension, ...), or a violation. Also melodic
     and motion (parallel/hidden) violations. This is both the fitness
     function and, at the end, the piece's proof of correctness.
  2. score(voices)     — weighted sum over the classification.
  3. compose(...)      — beam search: write one voice against fixed
     voices under the rules, with style priors (complementary rhythm,
     stepwise preference, suspensions rewarded).

Written for day 15 of the daily practice. The composer using this file
cannot hear; these rules are its ears.
"""

from __future__ import annotations
import bisect
from dataclasses import dataclass, field
from typing import Optional

BAR = 16  # 4/4 in 16th units
BEAT = 4

# ------------------------------------------------------------------ pitches

NAMES_SHARP = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]

@dataclass(frozen=True)
class Key:
    tonic: int          # pitch class 0-11
    mode: str           # "minor" | "major"
    # LilyPond spelling for each pc in this key's neighborhood
    spell: dict = field(default_factory=dict, compare=False)

    def scale_pcs(self):
        """Diatonic pcs. Minor returns natural minor plus raised 6/7 as a
        separate 'variant' set (melodic options, rule-governed)."""
        if self.mode == "major":
            degs = [0, 2, 4, 5, 7, 9, 11]
            return {(self.tonic + d) % 12 for d in degs}, set()
        degs = [0, 2, 3, 5, 7, 8, 10]
        base = {(self.tonic + d) % 12 for d in degs}
        variants = {(self.tonic + 9) % 12, (self.tonic + 11) % 12}  # raised 6, 7
        return base, variants

    def degree(self, pc):
        """Chromatic degree of pc relative to tonic (0-11)."""
        return (pc - self.tonic) % 12


def key_of(tonic_pc, mode, spell):
    return Key(tonic_pc, mode, spell)

# spelling tables (LilyPond names) for the keys this piece visits
SPELL_D_MINOR = {0:"c",1:"cis",2:"d",3:"ees",4:"e",5:"f",6:"fis",7:"g",8:"gis",9:"a",10:"bes",11:"b"}
SPELL_A_MINOR = {0:"c",1:"cis",2:"d",3:"dis",4:"e",5:"f",6:"fis",7:"g",8:"gis",9:"a",10:"bes",11:"b"}
SPELL_F_MAJOR = {0:"c",1:"des",2:"d",3:"ees",4:"e",5:"f",6:"fis",7:"g",8:"aes",9:"a",10:"bes",11:"b"}
SPELL_G_MINOR = {0:"c",1:"cis",2:"d",3:"ees",4:"e",5:"f",6:"fis",7:"g",8:"aes",9:"a",10:"bes",11:"b"}
SPELL_C_MAJOR = {0:"c",1:"cis",2:"d",3:"ees",4:"e",5:"f",6:"fis",7:"g",8:"gis",9:"a",10:"bes",11:"b"}

D_MINOR = key_of(2, "minor", SPELL_D_MINOR)
A_MINOR = key_of(9, "minor", SPELL_A_MINOR)
F_MAJOR = key_of(5, "major", SPELL_F_MAJOR)
G_MINOR = key_of(7, "minor", SPELL_G_MINOR)
C_MAJOR = key_of(0, "major", SPELL_C_MAJOR)

# ------------------------------------------------------------------- notes

@dataclass(frozen=True)
class Note:
    on: int
    dur: int
    pitch: int
    def __post_init__(self):
        assert self.dur > 0

    @property
    def off(self):
        return self.on + self.dur


def sounding(voice, t):
    """Note sounding at time t in voice, or None. voice sorted by onset."""
    # binary search: last note with on <= t
    lo, hi = 0, len(voice)
    while lo < hi:
        mid = (lo + hi) // 2
        if voice[mid].on <= t:
            lo = mid + 1
        else:
            hi = mid
    i = lo - 1
    if i >= 0 and voice[i].on <= t < voice[i].off:
        return voice[i]
    return None


def note_index_at(voice, t):
    lo, hi = 0, len(voice)
    while lo < hi:
        mid = (lo + hi) // 2
        if voice[mid].on <= t:
            lo = mid + 1
        else:
            hi = mid
    i = lo - 1
    if i >= 0 and voice[i].on <= t < voice[i].off:
        return i
    return None


def strength(t):
    """Metric strength at 16th-unit t: 4 downbeat, 3 half-bar, 2 beats 2/4,
    1 8th offbeat, 0 16th offbeat."""
    p = t % BAR
    if p == 0:
        return 4
    if p == 8:
        return 3
    if p % 4 == 0:
        return 2
    if p % 2 == 0:
        return 1
    return 0

# ------------------------------------------------------ interval classification

CONSONANT = {0, 3, 4, 7, 8, 9}   # P1 m3 M3 P5 m6 M6 (mod 12)
PERFECT = {0, 7}                  # P1/P8 and P5 (mod 12)

def ic(a, b):
    return abs(a - b) % 12


@dataclass
class Event:
    """A classified vertical moment between two voices."""
    t: int
    hi_voice: int
    lo_voice: int
    hi: int
    lo: int
    interval: int          # mod-12 interval
    kind: str              # "cons" | "pass" | "nbr" | "susp" | "app" | "VIOL"
    detail: str = ""


@dataclass
class Problem:
    t: int
    kind: str      # "parallel5", "parallel8", "hidden", "cross", "melodic", ...
    voices: tuple
    detail: str
    hard: bool


def _melodic_context(voice, idx):
    """(prev_pitch, next_pitch) around voice[idx], None at edges/rests-adjacent."""
    n = voice[idx]
    prev = voice[idx - 1] if idx > 0 and voice[idx - 1].off == n.on else None
    nxt = voice[idx + 1] if idx + 1 < len(voice) and voice[idx + 1].on == n.off else None
    return (prev.pitch if prev else None), (nxt.pitch if nxt else None)


def classify(voices, bass_index=None, final_bar_start=None, pedal_spans=None):
    """Full-score classification.

    voices: list of Voices, index 0 = top.
    bass_index: which voice is the bass for P4 treatment (default: lowest
      sounding voice at each moment).
    pedal_spans: [(voice_idx, t0, t1), ...] — pedal points. Dissonance in a
      pair involving an established pedal note inside a span is the named
      classical license "pedal", not a violation.
    Returns (events, problems).
    """
    events = []
    problems = []
    nv = len(voices)
    pedal_spans = pedal_spans or []

    def is_pedal_pair(t, vi, vj):
        for pv, p0, p1 in pedal_spans:
            if pv in (vi, vj) and p0 <= t < p1:
                return True
        return False

    onsets = sorted({n.on for v in voices for n in v})

    # ---- vertical classification at every onset
    for t in onsets:
        snd = [(vi, sounding(voices[vi], t)) for vi in range(nv)]
        snd = [(vi, n) for vi, n in snd if n is not None]
        if len(snd) < 2:
            continue
        lowest_pitch = min(n.pitch for _, n in snd)
        for a in range(len(snd)):
            for b in range(a + 1, len(snd)):
                vi, ni = snd[a]
                vj, nj = snd[b]
                if ni.on != t and nj.on != t:
                    continue  # no articulation in this pair at t
                hi_v, hi_n, lo_v, lo_n = (vi, ni, vj, nj) if ni.pitch >= nj.pitch else (vj, nj, vi, ni)
                iv = ic(hi_n.pitch, lo_n.pitch)
                involves_bass = lo_n.pitch == lowest_pitch
                cons = iv in CONSONANT or (iv == 5 and not involves_bass)
                if cons:
                    events.append(Event(t, hi_v, lo_v, hi_n.pitch, lo_n.pitch, iv, "cons"))
                    continue
                # dissonance: find an explanation
                expl = _explain_dissonance(voices, t, (hi_v, hi_n), (lo_v, lo_n))
                if expl is None and is_pedal_pair(t, hi_v, lo_v):
                    expl = ("pedal", "dissonance over/under a pedal point")
                if expl is None:
                    events.append(Event(t, hi_v, lo_v, hi_n.pitch, lo_n.pitch, iv,
                                        "VIOL", "unexplained dissonance"))
                else:
                    events.append(Event(t, hi_v, lo_v, hi_n.pitch, lo_n.pitch, iv,
                                        expl[0], expl[1]))

    # ---- motion rules per pair (parallels / hidden / crossing)
    for a in range(nv):
        for b in range(a + 1, nv):
            problems.extend(_motion_problems(voices, a, b, final_bar_start))

    # ---- melodic rules per voice
    for vi, v in enumerate(voices):
        problems.extend(_melodic_problems(v, vi))

    return events, problems


def _explain_dissonance(voices, t, hi, lo):
    """Try to explain a dissonant onset at t between (voice_idx, Note) pairs.
    Returns (kind, detail) or None."""
    (hv, hn), (lv, ln) = hi, lo
    s = strength(t)

    # which of the two articulated at t?
    movers, holders = [], []
    for vi, n in (hi, lo):
        (movers if n.on == t else holders).append((vi, n))

    # --- suspension: a held note made dissonant by the other's arrival,
    #     resolving DOWN BY STEP at its next onset (re-struck resolution ok).
    for vi, n in holders:
        v = voices[vi]
        idx = v.index(n)
        nxt = v[idx + 1] if idx + 1 < len(v) else None
        if nxt is not None and nxt.on == n.off and 1 <= n.pitch - nxt.pitch <= 2:
            # preparation: n was sounding (consonant assumed checked at its onset)
            if s >= 2 and n.off - t <= BAR:
                return ("susp", f"suspension in v{vi}, resolves {n.pitch}->{nxt.pitch}")
        # bass can also suspend upward against it? keep to down-resolution.

    # --- moving-voice figures
    for vi, n in movers:
        v = voices[vi]
        idx = v.index(n)
        prev_p, next_p = _melodic_context(v, idx)
        if prev_p is None:
            continue
        step_in = 1 <= abs(n.pitch - prev_p) <= 2
        if next_p is None:
            # not yet known (during search) — treat optimistically as pending
            if step_in and s <= 2:
                return ("pend", f"pending resolution in v{vi}")
            continue
        step_out = 1 <= abs(next_p - n.pitch) <= 2
        same_dir = (n.pitch - prev_p) * (next_p - n.pitch) > 0
        if step_in and step_out and same_dir and s <= 2:
            return ("pass", f"passing tone in v{vi}")
        if step_in and next_p == prev_p and s <= 1:
            return ("nbr", f"neighbor tone in v{vi}")
        # appoggiatura: leap in, step down out, on the beat — allow sparingly
        if not step_in and step_out and n.pitch - next_p in (1, 2) and s >= 2:
            return ("app", f"appoggiatura in v{vi}")
        # anticipation: step in, re-struck same pitch next, weak position
        if step_in and next_p == n.pitch and s == 0:
            return ("ant", f"anticipation in v{vi}")
    return None


def _motion_problems(voices, a, b, final_bar_start):
    """Parallel/hidden perfects and crossing between voices a and b."""
    probs = []
    va, vb = voices[a], voices[b]
    onsets = sorted({n.on for n in va} | {n.on for n in vb})
    prev = None  # (t, pa, pb, moved_a, moved_b)
    for t in onsets:
        na, nb = sounding(va, t), sounding(vb, t)
        if na is None or nb is None:
            prev = None
            continue
        moved_a, moved_b = na.on == t, nb.on == t
        pa, pb = na.pitch, nb.pitch
        # crossing (a is the upper voice by construction: index a < b)
        if pa < pb:
            probs.append(Problem(t, "cross", (a, b), f"{pa} below {pb}", True))
        if prev is not None:
            t0, qa, qb, _, _ = prev
            iv0, iv1 = ic(qa, qb), ic(pa, pb)
            if (moved_a and pa != qa) and (moved_b and pb != qb):
                # both voices genuinely moved
                if iv1 in PERFECT and iv0 == iv1:
                    which = "parallel5" if iv1 == 7 else "parallel8"
                    # allow the very final chord approach octave? no: keep hard.
                    probs.append(Problem(t, which, (a, b), f"{qa},{qb} -> {pa},{pb}", True))
                elif iv1 in PERFECT and iv0 in PERFECT and iv0 != iv1:
                    probs.append(Problem(t, "consec_perf", (a, b),
                                         f"perfect to perfect {iv0}->{iv1}", False))
                elif iv1 in PERFECT:
                    dir_a, dir_b = pa - qa, pb - qb
                    if dir_a * dir_b > 0:  # similar motion into a perfect
                        upper_leap = abs(pa - qa) > 2 if a < b else abs(pb - qb) > 2
                        outer = (a == 0 and b == len(voices) - 1)
                        if upper_leap:
                            probs.append(Problem(t, "hidden", (a, b),
                                                 f"similar into {'5th' if iv1==7 else '8ve'}",
                                                 outer))
                # unison arrival mid-piece
                if pa == pb and (final_bar_start is None or t < final_bar_start):
                    probs.append(Problem(t, "unison", (a, b), "voices at unison", False))
        prev = (t, pa, pb, moved_a, moved_b)
    return probs


LEAP_OK = {0, 1, 2, 3, 4, 5, 7, 8, 12}  # melodic intervals in semitones

def _melodic_problems(v, vi):
    probs = []
    for i in range(1, len(v)):
        if v[i - 1].off != v[i].on:
            continue  # rest between: melodic connection broken
        d = v[i].pitch - v[i - 1].pitch
        ad = abs(d)
        if ad not in LEAP_OK:
            hard = ad in (6, 10, 11) or ad > 12  # tritone, 7ths, >8ve
            probs.append(Problem(v[i].on, "melodic", (vi,), f"leap of {d}", hard))
        # aug 2nd approximation: 3-semitone move where neither pitch class
        # pair fits a minor 3rd within a diatonic frame is caught in search
        # (candidate generation), not here.
        if i >= 2 and v[i - 2].off == v[i - 1].on:
            d0 = v[i - 1].pitch - v[i - 2].pitch
            if abs(d0) >= 5 and abs(d) >= 5 and d0 * d > 0:
                probs.append(Problem(v[i].on, "melodic", (vi,),
                                     "two large leaps same direction", True))
            if abs(d0) >= 7 and not (1 <= abs(d) <= 2 and d0 * d < 0):
                probs.append(Problem(v[i].on, "melodic", (vi,),
                                     "big leap not recovered by step", False))
    return probs

# --------------------------------------------------------------- scoring

W = {
    "VIOL": -60.0,          # unexplained dissonance
    "pend": -2.0,           # unresolved-yet dissonance (search-time optimism)
    "pass": -0.4,
    "nbr": -0.6,
    "app": -1.5,
    "ant": -1.2,
    "susp": +2.5,           # suspensions are rewarded
    "pedal": -0.2,
    "cons": 0.0,
    "parallel5": -80.0,
    "parallel8": -80.0,
    "consec_perf": -6.0,
    "hidden_hard": -30.0,
    "hidden_soft": -3.0,
    "cross": -40.0,
    "unison": -4.0,
    "melodic_hard": -40.0,
    "melodic_soft": -4.0,
}

def score_events(events, problems):
    s = 0.0
    for e in events:
        s += W.get(e.kind, 0.0)
    for p in problems:
        if p.kind == "hidden":
            s += W["hidden_hard"] if p.hard else W["hidden_soft"]
        elif p.kind == "melodic":
            s += W["melodic_hard"] if p.hard else W["melodic_soft"]
        elif p.kind in ("parallel5", "parallel8", "cross", "consec_perf", "unison"):
            s += W[p.kind]
        else:
            s += -5.0
    return s


def score(voices, **kw):
    ev, pr = classify(voices, **kw)
    return score_events(ev, pr)

# ----------------------------------------------------------- beam search

@dataclass
class StylePrior:
    """Knobs for what the free voice should want, beyond legality."""
    step_bonus: float = 0.8          # per stepwise connection
    leap_penalty: float = 0.35       # per semitone of leap beyond a 3rd
    complement_bonus: float = 1.2    # moving while other voices hold
    sustain_bonus: float = 0.0       # holding while others move
    suspension_seek: float = 2.0     # extra shaping toward suspensions
    prefer_dur: dict = None          # dur -> bonus
    tessitura_center: int = 60
    tessitura_width: int = 14        # soft range half-width in semitones
    repeat_penalty: float = 2.5      # immediate same-pitch repetition
    triad_bonus: float = 1.0         # complete triad on strong beats
    perfect_penalty: float = 0.3     # perfect-interval onsets on strong beats

    def __post_init__(self):
        if self.prefer_dur is None:
            self.prefer_dur = {2: 0.5, 4: 0.4, 8: 0.0, 3: -0.6, 1: -0.8, 6: 0.1, 12: -0.2, 16: -0.5}


def durations_at(t, region_end, policy_dur):
    """Legal durations starting at t (metric grammar) intersected with region."""
    p = t % BAR
    outs = []
    for d in (1, 2, 3, 4, 6, 8, 12, 16):
        if t + d > region_end:
            continue
        if d == 1:
            outs.append(d)                      # 16ths anywhere
        elif d == 2 and p % 2 == 0:
            outs.append(d)                      # 8ths on 8th grid
        elif d == 3 and p % 2 == 0:
            outs.append(d)                      # dotted 8th on 8th grid
        elif d == 4 and p % 2 == 0:
            outs.append(d)                      # quarters on 8th grid (syncope ok)
        elif d == 6 and p % 4 == 0:
            outs.append(d)
        elif d == 8 and p % 4 == 0:
            outs.append(d)                      # halves on beat (incl. syncopated)
        elif d == 12 and p % 4 == 0:
            outs.append(d)
        elif d == 16 and p == 0:
            outs.append(d)
    return [d for d in outs if policy_dur.get(d, -99) > -50]


def candidate_pitches(key, prev_pitch, lo, hi):
    base, variants = key.scale_pcs()
    allowed = base | variants
    outs = []
    lo_c = max(lo, (prev_pitch - 12) if prev_pitch is not None else lo)
    hi_c = min(hi, (prev_pitch + 12) if prev_pitch is not None else hi)
    for p in range(lo_c, hi_c + 1):
        if p % 12 not in allowed:
            continue
        if prev_pitch is not None:
            ad = abs(p - prev_pitch)
            if ad not in LEAP_OK and ad != 0:
                continue
            # crude aug-2 guard: 3-semitone step between b6 and #7 of the key
            if ad == 3:
                degs = {key.degree(p % 12), key.degree(prev_pitch % 12)}
                if degs == {8, 11}:
                    continue
        outs.append(p)
    return outs


@dataclass
class _State:
    notes: tuple
    t: int
    score: float


def compose(fixed_voices, voice_slot, n_voices, region, key_track, vrange,
            prior=None, start_pitch=None, end_pitch=None, end_by_step=False,
            beam_width=220, rest_head=0, seed_notes=(), rng=None,
            return_top=1):
    """Beam-search a free voice.

    fixed_voices: {index: Voice} of already-written voices.
    voice_slot: index of the voice being written (0 = top).
    region: (t0, t1) exclusive end.
    key_track: list of (start_t, Key), sorted.
    vrange: (lo, hi) MIDI.
    start_pitch: optional required first pitch.
    end_pitch: optional required final pitch (search rejects others).
    end_by_step: require final connection to be stepwise.
    rest_head: units of rest at region start before the voice enters.
    seed_notes: notes (absolute time) fixed at the start of the region.
    Returns list[Note] (the free voice inside the region).
    """
    prior = prior or StylePrior()
    t0, t1 = region
    lo, hi = vrange
    fixed_voices = {vi: sorted(v, key=lambda n: n.on)
                    for vi, v in fixed_voices.items()}
    fixed_ons = {vi: [n.on for n in v] for vi, v in fixed_voices.items()}
    key_xs = [k[0] for k in key_track]

    def key_at(t):
        i = bisect.bisect_right(key_xs, t) - 1
        return key_track[max(0, i)][1]

    def others_sounding(t):
        outs = []
        for vi, v in fixed_voices.items():
            n = sounding(v, t)
            if n is not None:
                outs.append((vi, n))
        return outs

    def nn_is_last(nn):
        return nn.off >= t1

    def others_onset_between(a, b):
        for vi, ons in fixed_ons.items():
            i = bisect.bisect_right(ons, a)
            if i < len(ons) and ons[i] < b:
                return True
        return False

    def onsets_in(vi, a, b):
        """Fixed voice vi's notes with a < on < b."""
        ons = fixed_ons[vi]
        i = bisect.bisect_right(ons, a)
        v = fixed_voices[vi]
        out = []
        while i < len(ons) and ons[i] < b:
            out.append(v[i])
            i += 1
        return out

    def vertical_delta(cand_notes, new_note):
        """Score contribution of adding new_note: check at its onset against
        others, plus re-check others' onsets during its span, plus finalize
        the previous note's pending dissonance."""
        s = 0.0
        t = new_note.on
        snd = others_sounding(t)
        my = new_note.pitch
        all_pitches = [n.pitch for _, n in snd] + [my]
        low = min(all_pitches)
        for vi, n in snd:
            iv = ic(my, n.pitch)
            lo_p = min(my, n.pitch)
            involves_bass = lo_p == low
            cons = iv in CONSONANT or (iv == 5 and not involves_bass)
            if cons:
                # color pref: penalize perfect-interval onsets (episodes crank
                # this up so sequences ride on 3rds/6ths and survive diatonic
                # transposition)
                if iv in PERFECT and strength(t) >= 2:
                    s -= prior.perfect_penalty
                continue
            # dissonant onset by me: is it plausibly passing/nbr/susp-resolution?
            prev = cand_notes[-1] if cand_notes else None
            prev_p = prev.pitch if prev is not None and prev.off == t else None
            step_in = prev_p is not None and 1 <= abs(my - prev_p) <= 2
            st = strength(t)
            if step_in and st <= 2:
                # optimistic (finalized on next note) — but a region's LAST
                # note has no next note: its dissonance would never resolve
                s += W["VIOL"] if nn_is_last(new_note) else W["pend"]
            elif st >= 2 and prev_p is not None and abs(my - prev_p) > 2:
                s += W["app"] * 2       # appoggiatura: discourage but legal-ish
            else:
                s += W["VIOL"]
        # others articulating during my span: suspension-shaped or clash?
        # (also: a fixed voice crossing my HELD note was invisible to the
        # search until day 15's first draft — check it here)
        for vi in fixed_voices:
            for n in onsets_in(vi, new_note.on, new_note.off):
                if True:
                    iv = ic(my, n.pitch)
                    lo_p = min(my, n.pitch)
                    snd2 = [p for _, m in others_sounding(n.on) for p in [m.pitch]]
                    low2 = min(snd2 + [my]) if snd2 else min(my, n.pitch)
                    cons = iv in CONSONANT or (iv == 5 and min(my, n.pitch) != low2)
                    if not cons:
                        # I become a suspension IF I resolve down by step next;
                        # can't know yet -> mild optimism on strong, hard on weak
                        if nn_is_last(new_note):
                            s += W["VIOL"]
                        elif strength(n.on) >= 2:
                            s += 0.5    # suspension potential (bonus later)
                        else:
                            s += W["VIOL"] * 0.5
                    if (voice_slot < vi and my < n.pitch) or \
                       (voice_slot > vi and my > n.pitch):
                        s += W["cross"]
        return s

    def finalize_prev(cand_notes, new_note):
        """Now that new_note fixes prev's continuation, settle prev's pending
        dissonances (passing/neighbor legality + suspension resolution)."""
        if not cand_notes:
            return 0.0
        prev = cand_notes[-1]
        if prev.off != new_note.on:
            return 0.0
        s = 0.0
        # was prev dissonant at its onset vs any fixed voice?
        for vi, v in fixed_voices.items():
            n = sounding(v, prev.on)
            if n is None:
                continue
            iv = ic(prev.pitch, n.pitch)
            snd = others_sounding(prev.on)
            all_p = [m.pitch for _, m in snd] + [prev.pitch]
            cons = iv in CONSONANT or (iv == 5 and min(prev.pitch, n.pitch) != min(all_p))
            if cons:
                continue
            pp = cand_notes[-2].pitch if len(cand_notes) >= 2 and cand_notes[-2].off == prev.on else None
            step_in = pp is not None and 1 <= abs(prev.pitch - pp) <= 2
            step_out = 1 <= abs(new_note.pitch - prev.pitch) <= 2
            same_dir = pp is not None and (prev.pitch - pp) * (new_note.pitch - prev.pitch) > 0
            if step_in and step_out and same_dir:
                s += W["pass"] - W["pend"]
            elif step_in and new_note.pitch == pp:
                s += W["nbr"] - W["pend"]
            else:
                s += W["VIOL"] - W["pend"]      # pending hope failed
        # did a fixed voice articulate under prev making it a suspension?
        for vi in fixed_voices:
            for n in onsets_in(vi, prev.on, prev.off):
                if True:
                    iv = ic(prev.pitch, n.pitch)
                    snd2 = others_sounding(n.on)
                    all_p = [m.pitch for _, m in snd2] + [prev.pitch]
                    cons = iv in CONSONANT or (iv == 5 and min(prev.pitch, n.pitch) != min(all_p))
                    if not cons:
                        if 1 <= prev.pitch - new_note.pitch <= 2 and strength(n.on) >= 2:
                            s += W["susp"] + prior.suspension_seek
                        else:
                            s += W["VIOL"] * 0.5
        return s

    def motion_delta(cand_notes, new_note):
        """Parallels/hidden/cross vs fixed voices for the connection into
        new_note. The reference event is the LAST articulation before my
        onset in either voice (draft 3 found parallels forming at the fixed
        voice's onsets under my held notes when this sampled my onset only)."""
        if not cand_notes:
            return 0.0
        prev = cand_notes[-1]
        if prev.off != new_note.on:
            return 0.0
        s = 0.0
        t_now = new_note.on
        for vi, v in fixed_voices.items():
            n1 = sounding(v, t_now)
            if n1 is None:
                continue
            # last event in (my prev onset .. now) where the fixed voice spoke
            t_ev = prev.on
            ons = fixed_ons[vi]
            j = bisect.bisect_left(ons, t_now) - 1
            if j >= 0 and ons[j] > prev.on:
                t_ev = ons[j]
            n0 = sounding(v, t_ev)
            if n0 is None:
                continue
            iv0, iv1 = ic(prev.pitch, n0.pitch), ic(new_note.pitch, n1.pitch)
            i_moved = new_note.pitch != prev.pitch
            o_moved = n1.pitch != n0.pitch
            if i_moved and o_moved and iv1 in PERFECT and iv0 == iv1:
                s += W["parallel5"] if iv1 == 7 else W["parallel8"]
            elif i_moved and o_moved and iv1 in PERFECT and iv0 in PERFECT:
                s += W["consec_perf"]
            elif iv1 in PERFECT and i_moved and o_moved:
                if (new_note.pitch - prev.pitch) * (n1.pitch - n0.pitch) > 0:
                    upper_is_me = voice_slot < vi
                    upper_leap = abs((new_note.pitch - prev.pitch) if upper_is_me
                                     else (n1.pitch - n0.pitch)) > 2
                    if upper_leap:
                        outer = {voice_slot, vi} == {0, n_voices - 1}
                        s += W["hidden_hard"] if outer else W["hidden_soft"]
            # crossing
            if (voice_slot < vi and new_note.pitch < n1.pitch) or \
               (voice_slot > vi and new_note.pitch > n1.pitch):
                s += W["cross"]
            # overlap
            if (voice_slot < vi and new_note.pitch < n0.pitch) or \
               (voice_slot > vi and new_note.pitch > n0.pitch):
                s += -6.0
            # unison
            if new_note.pitch == n1.pitch:
                s += W["unison"]
            # spacing with neighbors (soft, adjacent voice indexes only)
            if abs(vi - voice_slot) == 1 and vi != n_voices - 1 and voice_slot != n_voices - 1:
                if abs(new_note.pitch - n1.pitch) > 19:
                    s += -2.0
        return s

    def melodic_delta(cand_notes, new_note):
        s = 0.0
        if cand_notes and cand_notes[-1].off == new_note.on:
            prev = cand_notes[-1]
            d = new_note.pitch - prev.pitch
            ad = abs(d)
            if ad == 0:
                s -= prior.repeat_penalty
                if len(cand_notes) >= 2 and cand_notes[-2].pitch == prev.pitch:
                    s -= prior.repeat_penalty * 2
            elif ad <= 2:
                s += prior.step_bonus
            else:
                s -= prior.leap_penalty * max(0, ad - 4)
            if len(cand_notes) >= 2 and cand_notes[-2].off == prev.on:
                d0 = prev.pitch - cand_notes[-2].pitch
                if abs(d0) >= 5 and abs(d) >= 5 and d0 * d > 0:
                    s += W["melodic_hard"]
                elif abs(d0) >= 7:
                    s += 1.2 if (1 <= ad <= 2 and d0 * d < 0) else -3.0
            # oscillation guard: p,q,p,q reads as a stuck trill (day 15's
            # draft 5 wrote C#-C-C#-C for two whole bars over the pedal)
            if len(cand_notes) >= 3:
                if (new_note.pitch == cand_notes[-2].pitch
                        and cand_notes[-1].pitch == cand_notes[-3].pitch
                        and new_note.pitch != cand_notes[-1].pitch):
                    s -= 8.0
                    if len(cand_notes) >= 5 and cand_notes[-2].pitch == cand_notes[-4].pitch:
                        s -= 10.0
        # tessitura
        dev = abs(new_note.pitch - prior.tessitura_center)
        if dev > prior.tessitura_width:
            s -= 0.5 * (dev - prior.tessitura_width)
        return s

    def style_delta(cand_notes, new_note):
        s = prior.prefer_dur.get(new_note.dur, -1.0)
        # complementary rhythm: bonus if others hold at my onset
        if not others_onset_between(new_note.on - 1, new_note.on + 1):
            s += prior.complement_bonus * 0.5
        others_move_during = others_onset_between(new_note.on, new_note.off)
        if new_note.dur >= 8 and others_move_during:
            s += prior.sustain_bonus
        if new_note.dur <= 2 and not others_move_during:
            s += prior.complement_bonus * 0.5
        return s

    # ---- search
    init_notes = tuple(seed_notes)
    init_t = t0 + rest_head if not init_notes else init_notes[-1].off
    states = [_State(init_notes, init_t, 0.0)]
    while True:
        # all states that reached t1 are complete
        done = [st for st in states if st.t >= t1]
        live = [st for st in states if st.t < t1]
        if not live:
            break
        new_states = []
        dropped_states = []
        for st in live:
            k = key_at(st.t)
            prev_pitch = st.notes[-1].pitch if st.notes and st.notes[-1].off == st.t else \
                         (st.notes[-1].pitch if st.notes else None)
            first = len(st.notes) == len(init_notes)   # seeds don't count
            if first and start_pitch is not None:
                pitches = [start_pitch]
            else:
                pitches = candidate_pitches(k, prev_pitch, lo, hi)
            durs = durations_at(st.t, t1, prior.prefer_dur)
            if not durs:
                durs = [t1 - st.t]
            for d in durs:
                for p in pitches:
                    nn = Note(st.t, d, p)
                    if nn.off >= t1:
                        if end_pitch is not None and p != end_pitch:
                            continue
                        if end_by_step and prev_pitch is not None and not (1 <= abs(p - prev_pitch) <= 2):
                            continue
                    ds = (vertical_delta(st.notes, nn) + finalize_prev(st.notes, nn)
                          + motion_delta(st.notes, nn) + melodic_delta(st.notes, nn)
                          + style_delta(st.notes, nn))
                    cand = _State(st.notes + (nn,), nn.off, st.score + ds)
                    # a step that picks up a fresh hard violation is culled,
                    # not merely penalized — a -80 parallel still won twice
                    # in draft 2 when every rival carried accumulated softs
                    (new_states if ds > -35 else dropped_states).append(cand)
        if not new_states:
            new_states = dropped_states   # only lawbreaking moves exist here
        # prune
        new_states.sort(key=lambda s: -s.score)
        # dedupe by (t, last pitch, last dur)
        seen = set()
        pruned = []
        for s2 in new_states:
            key2 = (s2.t, s2.notes[-1].pitch if s2.notes else None,
                    s2.notes[-1].dur if s2.notes else None,
                    s2.notes[-2].pitch if len(s2.notes) >= 2 else None)
            if key2 in seen:
                continue
            seen.add(key2)
            pruned.append(s2)
            if len(pruned) >= beam_width:
                break
        states = pruned + done
        if all(s.t >= t1 for s in states):
            break

    complete = [s for s in states if s.t >= t1]
    if not complete:
        raise RuntimeError(f"compose: no complete solutions in region {region}")
    complete.sort(key=lambda s: -s.score)
    if return_top == 1:
        return list(complete[0].notes)
    return [list(s.notes) for s in complete[:return_top]]


# ------------------------------------------------------------- utilities

def transpose(notes, semitones, dt=0):
    return [Note(n.on + dt, n.dur, n.pitch + semitones) for n in notes]


def shift(notes, dt):
    return [Note(n.on + dt, n.dur, n.pitch) for n in notes]


def report(voices, names=None, **kw):
    """Human-readable rule report."""
    ev, pr = classify(voices, **kw)
    names = names or [f"v{i}" for i in range(len(voices))]
    lines = []
    counts = {}
    for e in ev:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    lines.append("vertical events: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    viols = [e for e in ev if e.kind == "VIOL"]
    for e in viols:
        lines.append(f"  VIOL bar {e.t // BAR + 1} beat {(e.t % BAR) / 4 + 1:.2g}: "
                     f"{names[e.hi_voice]}({e.hi}) vs {names[e.lo_voice]}({e.lo}) iv={e.interval} {e.detail}")
    hard = [p for p in pr if p.hard]
    soft = [p for p in pr if not p.hard]
    lines.append(f"problems: {len(hard)} hard, {len(soft)} soft")
    for p in hard:
        vs = ",".join(names[v] for v in p.voices)
        lines.append(f"  HARD bar {p.t // BAR + 1} beat {(p.t % BAR) / 4 + 1:.2g}: {p.kind} [{vs}] {p.detail}")
    for p in soft:
        vs = ",".join(names[v] for v in p.voices)
        lines.append(f"  soft bar {p.t // BAR + 1} beat {(p.t % BAR) / 4 + 1:.2g}: {p.kind} [{vs}] {p.detail}")
    return "\n".join(lines), ev, pr
