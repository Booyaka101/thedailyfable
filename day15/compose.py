"""compose.py — day 15: a three-voice fugue in D minor, composed with the
counterpoint engine.

Architecture (mine) / notes (the engine's, under law):

  bars  1-2   exposition: subject, alto alone
  bars  3-4   answer (tonal), soprano; countersubject, alto (engine,
              filtered for invertibility)
  bars  5-6   subject, bass; countersubject, soprano; alto free (engine)
  bars  7-8   episode 1: sequence on the tail-turn (alto fixed),
              bass+soprano engine — collection pivots to F major
  bars  9-10  middle entry, F major, alto; soprano CS if it survives
              re-rendering, else engine; bass engine
  bars 11-12  episode 2: lament tetrachord bass (fixed), engine suspension
              chains above — darkens to G minor
  bars 13-14  middle entry, G minor, bass; upper voices engine
  bars 15-16  episode 3: bass rises to the dominant; half cadence on A
  bars 17-18  stretto I  (engine-discovered): soprano leads, bass follows
              at 5 beats, lower double octave
  bars 19-20  stretto II (tighter): alto leads, soprano follows at 3.5
              beats above — the syncopated entry the map found
  bars 21-22  dominant pedal, upper voices in dialogue
  bars 23-26  finale: subject AUGMENTED in the bass; normal subject rides
              on top of it from bar 24 (the machine found this fits)
  bars 27-28  cadence: iv, 4-3 suspension over V, Picardy D major
"""

import json
import sys
import time
from cp import (Note, D_MINOR, A_MINOR, F_MAJOR, G_MINOR, BAR,
                compose, StylePrior, classify, score_events, report,
                transpose, shift)
from fugue import tonal_answer, rerender, try_pair, find_strettos
import emit

T0 = time.time()

def u(bar):  # bar (1-indexed) -> 16th units
    return (bar - 1) * 16

END = u(29)          # 28 bars
SOP, ALTO, BASS = 0, 1, 2
RANGES = {SOP: (60, 81), ALTO: (53, 76), BASS: (40, 64)}
KEY_TRACK = [(0, D_MINOR), (u(7), F_MAJOR), (u(11), G_MINOR), (u(15), D_MINOR)]

# ------------------------------------------------------------ the subject
SUBJECT = [
    Note(0, 4, 62), Note(4, 2, 57), Note(6, 2, 58), Note(8, 2, 62),
    Note(10, 2, 61), Note(12, 2, 62), Note(14, 2, 64), Note(16, 4, 65),
    Note(20, 2, 64), Note(22, 2, 62), Note(24, 2, 64), Note(26, 2, 61),
    Note(28, 4, 62),
]
ANSWER = tonal_answer(SUBJECT, D_MINOR, head_notes=2)
SUBJECT_F = rerender(SUBJECT, D_MINOR, F_MAJOR)
SUBJECT_G = rerender(SUBJECT, D_MINOR, G_MINOR)

sop, alto, bass = [], [], []
sections = []

def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

def add(voice_list, notes, drop_before=None):
    """Append notes (dropping seed echoes before drop_before)."""
    for n in notes:
        if drop_before is not None and n.on < drop_before:
            continue
        voice_list.append(n)

def pair_stats(a, b):
    v, h, s = try_pair(a, b)
    return f"viol={v} hard={h} score={s:.1f}"

# ===================================================== bars 1-2: subject
sections.append({"name": "exposition — subject (alto)", "bar": 1, "t": 0})
add(alto, SUBJECT)
log("bars 1-2: subject in alto")

# ============================================ bars 3-4: answer + countersubject
sections.append({"name": "answer (soprano) + countersubject (alto)", "bar": 3, "t": u(3)})
answer_sop = shift(ANSWER, u(3))
add(sop, answer_sop)

# countersubject: engine, seeded with the subject's final note for melodic
# continuity, then filtered for how it behaves on BOTH sides (bars 3-4
# orientation below the answer; bars 5-6 orientation above the subject).
cs_prior = StylePrior(
    complement_bonus=1.6, suspension_seek=3.0, step_bonus=0.9,
    tessitura_center=60, tessitura_width=10,
    prefer_dur={2: 0.6, 4: 0.4, 8: -0.1, 3: -0.4, 1: -1.5, 6: 0.1, 12: -1.0, 16: -2.0},
)
log("composing countersubject candidates...")
cands = compose(
    fixed_voices={SOP: sop}, voice_slot=ALTO, n_voices=2,
    region=(u(2) + 12, u(5)), key_track=[(0, D_MINOR), (u(3), A_MINOR)],
    vrange=(55, 72), prior=cs_prior,
    seed_notes=(Note(u(2) + 12, 4, 62),),   # subject's last note
    beam_width=320, return_top=80,
)
log(f"  {len(cands)} candidates")

best_cs = None
subj_bass_test = shift(transpose(SUBJECT, -12), u(3))  # same alignment as answer
for cand in cands:
    cs = [n for n in cand if n.on >= u(3)]
    if not cs:
        continue
    v1, h1, s1 = try_pair(cs, answer_sop)                    # as composed
    cs_over = shift(transpose([Note(n.on - u(3), n.dur, n.pitch) for n in cs], 5), u(3))
    v2, h2, s2 = try_pair(cs_over, subj_bass_test)           # bars 5-6 usage
    if v1 == 0 and h1 == 0 and v2 == 0 and h2 == 0:
        best_cs = (cand, cs, s1 + s2)
        break
if best_cs is None:
    # relax: allow soft-only problems in the flipped orientation
    scored = []
    for cand in cands:
        cs = [n for n in cand if n.on >= u(3)]
        if not cs:
            continue
        v1, h1, s1 = try_pair(cs, answer_sop)
        cs_over = shift(transpose([Note(n.on - u(3), n.dur, n.pitch) for n in cs], 5), u(3))
        v2, h2, s2 = try_pair(cs_over, subj_bass_test)
        scored.append((v1 + v2, h1 + h2, -(s1 + s2), cand, cs))
    scored.sort()
    best_cs = (scored[0][3], scored[0][4], -scored[0][2])
    log(f"  WARNING: no fully-invertible CS; best has viol+hard={scored[0][0]}+{scored[0][1]}")

cs_full, CS, _ = best_cs
add(alto, cs_full, drop_before=u(2) + 16)   # keep everything after subject end
CS_REL = [Note(n.on - u(3), n.dur, n.pitch) for n in CS]  # relative form
log(f"  CS chosen: {len(CS)} notes; vs answer: {pair_stats(CS, answer_sop)}")

# ==================================== bars 5-6: bass subject, sop CS, alto free
sections.append({"name": "subject (bass) + countersubject (soprano)", "bar": 5, "t": u(5)})
subj_bass = shift(transpose(SUBJECT, -12), u(5))
add(bass, subj_bass)
cs_sop = shift(transpose(CS_REL, 5), u(5))
add(sop, cs_sop)
log(f"bars 5-6: CS(sop) vs subject(bass): {pair_stats(cs_sop, subj_bass)}")

free_prior = StylePrior(complement_bonus=1.2, suspension_seek=2.2,
                        tessitura_center=62, tessitura_width=11)
# alto keeps below the (low-lying) soprano CS and takes a breath before the
# episode: forcing it up to G4 in draft 1 crossed it over the soprano.
alto_56 = compose(
    fixed_voices={SOP: sop, BASS: bass}, voice_slot=ALTO, n_voices=3,
    region=(u(4) + 12, u(7) - 4), key_track=KEY_TRACK, vrange=(53, 66),
    prior=free_prior, seed_notes=(alto[-1],), beam_width=260,
)
add(alto, alto_56, drop_before=alto[-1].off)
log("bars 5-6: alto free voice composed")

# =============================================== bars 7-8: episode 1 (to F)
sections.append({"name": "episode 1 — sequence on the turn", "bar": 7, "t": u(7)})
ep1_alto = [
    Note(u(7), 4, 69), Note(u(7)+4, 2, 67), Note(u(7)+6, 2, 65),
    Note(u(7)+8, 2, 67), Note(u(7)+10, 2, 64), Note(u(7)+12, 4, 65),
    Note(u(8), 4, 67), Note(u(8)+4, 2, 65), Note(u(8)+6, 2, 64),
    Note(u(8)+8, 2, 65), Note(u(8)+10, 2, 62), Note(u(8)+12, 4, 64),
]
add(alto, ep1_alto)

ep_prior = StylePrior(complement_bonus=1.4, suspension_seek=2.5,
                      perfect_penalty=1.5, tessitura_center=48,
                      tessitura_width=10,
                      prefer_dur={2: 0.6, 4: 0.3, 8: -0.2, 3: -0.5, 1: -1.2, 6: 0.0, 12: -1.2, 16: -2.5})
bass_ep1 = compose(
    fixed_voices={ALTO: alto, SOP: sop}, voice_slot=BASS, n_voices=3,
    region=(u(6) + 12, u(9)), key_track=KEY_TRACK, vrange=RANGES[BASS],
    prior=ep_prior, seed_notes=(bass[-1],), end_pitch=53, beam_width=260,
)
add(bass, bass_ep1, drop_before=bass[-1].off)
sop_ep1_prior = StylePrior(complement_bonus=1.4, suspension_seek=3.0,
                           perfect_penalty=1.5, tessitura_center=72, tessitura_width=9)
sop_ep1 = compose(
    fixed_voices={ALTO: alto, BASS: bass},
    voice_slot=SOP, n_voices=3,
    region=(u(6) + 12, u(9)), key_track=KEY_TRACK, vrange=(62, 79),
    prior=sop_ep1_prior, seed_notes=(sop[-1],), end_by_step=True, beam_width=260,
)
add(sop, sop_ep1, drop_before=sop[-1].off)
log("bars 7-8: episode 1 composed")

# ============================================ bars 9-10: middle entry, F major
sections.append({"name": "middle entry — F major (alto)", "bar": 9, "t": u(9)})
entry_f = shift(SUBJECT_F, u(9))
add(alto, entry_f)

# try the countersubject re-rendered to F in the soprano
cs_f = rerender(shift(transpose(CS_REL, 5), u(9)), D_MINOR, F_MAJOR)
vF, hF, sF = try_pair(cs_f, entry_f)
log(f"bars 9-10: CS re-rendered to F vs entry: viol={vF} hard={hF}")
if vF == 0 and hF == 0:
    add(sop, cs_f, drop_before=sop[-1].off)
    sop_had_cs_f = True
else:
    sop_910 = compose(
        fixed_voices={ALTO: alto, BASS: bass}, voice_slot=SOP, n_voices=3,
        region=(sop[-1].on, u(11)), key_track=KEY_TRACK, vrange=(64, 81),
        prior=sop_ep1_prior, seed_notes=(sop[-1],), beam_width=260,
    )
    add(sop, sop_910, drop_before=sop[-1].off)
    sop_had_cs_f = False

bass_910 = compose(
    fixed_voices={ALTO: alto, SOP: sop},
    voice_slot=BASS, n_voices=3,
    region=(bass[-1].on, u(11)), key_track=KEY_TRACK, vrange=(40, 59),
    prior=StylePrior(complement_bonus=1.0, suspension_seek=1.5,
                     tessitura_center=48, tessitura_width=10,
                     prefer_dur={2: 0.3, 4: 0.5, 8: 0.2, 3: -0.5, 1: -1.5, 6: 0.2, 12: -0.6, 16: -2.0}),
    seed_notes=(bass[-1],), end_pitch=50, beam_width=260,
)
add(bass, bass_910, drop_before=bass[-1].off)
log("bars 9-10: F-major entry set")

# ============================================ bars 11-12: episode 2 (to g)
sections.append({"name": "episode 2 — lament bass, suspension chains", "bar": 11, "t": u(11)})
ep2_bass = [Note(u(11), 8, 50), Note(u(11)+8, 8, 48),
            Note(u(12), 8, 46), Note(u(12)+8, 8, 45)]
add(bass, ep2_bass)

susp_prior = StylePrior(complement_bonus=1.0, suspension_seek=4.0,
                        sustain_bonus=1.0, tessitura_center=70, tessitura_width=9,
                        prefer_dur={2: 0.2, 4: 0.6, 8: 0.4, 3: -0.5, 1: -1.5, 6: 0.3, 12: -0.4, 16: -1.8})
sop_ep2 = compose(
    fixed_voices={BASS: bass, ALTO: alto}, voice_slot=SOP, n_voices=3,
    region=(sop[-1].on, u(13)), key_track=KEY_TRACK, vrange=(62, 79),
    prior=susp_prior, seed_notes=(sop[-1],), end_pitch=69, beam_width=260,
)
add(sop, sop_ep2, drop_before=sop[-1].off)
alto_ep2 = compose(
    fixed_voices={BASS: bass, SOP: sop},
    voice_slot=ALTO, n_voices=3,
    region=(alto[-1].on, u(13)), key_track=KEY_TRACK, vrange=RANGES[ALTO],
    prior=StylePrior(complement_bonus=1.2, suspension_seek=3.0,
                     tessitura_center=63, tessitura_width=9),
    seed_notes=(alto[-1],), end_pitch=65, end_by_step=True, beam_width=260,
)
add(alto, alto_ep2, drop_before=alto[-1].off)
log("bars 11-12: episode 2 composed")

# ============================================ bars 13-14: entry in G minor
# (draft 1 put this in the bass an octave down: the subject's low D fell
# below the instrument, and the lament's A2 sat a 7th from the entry. The
# alto takes it at pitch.)
sections.append({"name": "middle entry — G minor (alto)", "bar": 13, "t": u(13)})
entry_g = shift(SUBJECT_G, u(13))
add(alto, entry_g, drop_before=alto[-1].off)

sop_1314 = compose(
    fixed_voices={ALTO: alto, BASS: bass}, voice_slot=SOP, n_voices=3,
    region=(sop[-1].on, u(15)), key_track=KEY_TRACK, vrange=(68, 81),
    prior=StylePrior(complement_bonus=1.3, suspension_seek=2.5,
                     tessitura_center=74, tessitura_width=9),
    seed_notes=(sop[-1],), beam_width=260,
)
add(sop, sop_1314, drop_before=sop[-1].off)
bass_1314 = compose(
    fixed_voices={ALTO: alto, SOP: sop},
    voice_slot=BASS, n_voices=3,
    region=(bass[-1].on, u(15)), key_track=KEY_TRACK, vrange=(40, 58),
    prior=StylePrior(complement_bonus=1.0, suspension_seek=1.5,
                     tessitura_center=50, tessitura_width=9,
                     prefer_dur={2: 0.3, 4: 0.5, 8: 0.2, 3: -0.5, 1: -1.5, 6: 0.2, 12: -0.6, 16: -2.0}),
    seed_notes=(bass[-1],), end_pitch=52, end_by_step=True, beam_width=260,
)
add(bass, bass_1314, drop_before=bass[-1].off)
log("bars 13-14: G-minor entry set (alto)")

# ============================================ bars 15-16: episode 3 (to V)
sections.append({"name": "episode 3 — rise to the dominant", "bar": 15, "t": u(15)})
ep3_bass = [Note(u(15), 8, 53), Note(u(15)+8, 8, 55),
            Note(u(16), 8, 57), Note(u(16)+8, 8, 45)]
add(bass, ep3_bass)

sop_ep3 = compose(
    fixed_voices={BASS: bass, ALTO: alto}, voice_slot=SOP, n_voices=3,
    region=(sop[-1].on, u(17)), key_track=KEY_TRACK, vrange=(62, 81),
    prior=susp_prior, seed_notes=(sop[-1],),
    end_pitch=73, end_by_step=True, beam_width=280,
)
add(sop, sop_ep3, drop_before=sop[-1].off)
alto_ep3 = compose(
    fixed_voices={BASS: bass, SOP: sop},
    voice_slot=ALTO, n_voices=3,
    region=(alto[-1].on, u(17)), key_track=KEY_TRACK, vrange=RANGES[ALTO],
    prior=free_prior, seed_notes=(alto[-1],), end_pitch=69, beam_width=260,
)
add(alto, alto_ep3, drop_before=alto[-1].off)
log("bars 15-16: episode 3 composed (half cadence)")

# ============================================ bars 17-18+: STRETTO I
sections.append({"name": "stretto I — soprano leads, bass at 5 beats", "bar": 17, "t": u(17)})
str1_sop = shift(transpose(SUBJECT, 12), u(17))            # D5, 256-288
# bass follower states the HEAD only: its full tail collided with stretto
# II's alto entry (the dt=12 residual pair the map rates dirty). Incomplete
# stretto entries are idiomatic; the engine proved head-only is clean.
str1_bass = shift(transpose([n for n in SUBJECT if n.on < 16], -12), u(17) + 20)
add(sop, str1_sop)
add(bass, str1_bass)
log(f"stretto I pair (head-only follower): {pair_stats(str1_sop, str1_bass)}")

alto_str1 = compose(
    fixed_voices={SOP: sop, BASS: bass}, voice_slot=ALTO, n_voices=3,
    region=(alto[-1].on, u(19)), key_track=KEY_TRACK, vrange=RANGES[ALTO],
    prior=free_prior, seed_notes=(alto[-1],),
    end_pitch=64, end_by_step=True, beam_width=260,
)
add(alto, alto_str1, drop_before=alto[-1].off)

# ============================================ bars 19-20+: STRETTO II
sections.append({"name": "stretto II — alto leads, soprano at 3.5 beats", "bar": 19, "t": u(19)})
str2_alto = shift(SUBJECT, u(19))                          # D4, 288-320
str2_sop = shift(transpose(SUBJECT, 12), u(19) + 14)       # D5, 302-334
add(alto, str2_alto)
log(f"stretto II pair: {pair_stats(str2_alto, str2_sop)}")

sop_link = compose(
    fixed_voices={ALTO: alto, BASS: bass}, voice_slot=SOP, n_voices=3,
    region=(sop[-1].on, u(19) + 14), key_track=KEY_TRACK, vrange=(64, 81),
    prior=free_prior, seed_notes=(sop[-1],),
    end_pitch=76, end_by_step=True, beam_width=240,
)
add(sop, sop_link, drop_before=sop[-1].off)
add(sop, str2_sop)

bass_link = compose(
    fixed_voices={ALTO: alto, SOP: sop},
    voice_slot=BASS, n_voices=3,
    region=(bass[-1].on, u(21)), key_track=KEY_TRACK, vrange=RANGES[BASS],
    prior=StylePrior(complement_bonus=1.0, tessitura_center=50, tessitura_width=9),
    seed_notes=(bass[-1],), start_pitch=53,  # F3 legalizes the head's E3 as passing
    end_pitch=46, end_by_step=True, beam_width=240,
)
add(bass, bass_link, drop_before=bass[-1].off)
log("bars 19-20: stretto II set")

# ============================================ bars 21-22: dominant pedal
sections.append({"name": "dominant pedal", "bar": 21, "t": u(21)})
PEDAL = Note(u(21), 32, 45)                                 # A2, 320-352
bass.append(PEDAL)
# the augmented bass is fully determined — put it in the world BEFORE the
# pedal soprano composes across bar 23, or it writes over false silence
aug_bass = [Note(u(23) + n.on * 2, n.dur * 2, n.pitch - 12) for n in SUBJECT]
add(bass, aug_bass)

sop_pedal = compose(
    fixed_voices={BASS: bass, ALTO: alto}, voice_slot=SOP, n_voices=3,
    region=(sop[-1].on, u(24)), key_track=KEY_TRACK, vrange=(62, 81),
    prior=StylePrior(complement_bonus=1.0, suspension_seek=3.5,
                     sustain_bonus=0.8, tessitura_center=72, tessitura_width=9,
                     prefer_dur={2: 0.2, 4: 0.6, 8: 0.4, 3: -0.4, 1: -1.2, 6: 0.3, 12: -0.3, 16: -1.5}),
    seed_notes=(sop[-1],), end_pitch=73, end_by_step=True, beam_width=260,
)
add(sop, sop_pedal, drop_before=sop[-1].off)

# ============================================ bars 23-26: FINALE (augmentation)
sections.append({"name": "finale — subject augmented (bass) with subject above", "bar": 23, "t": u(23)})
# The soprano's entry over the augmentation carries two ornamental 16ths —
# not decoration for its own sake: the plain statement makes parallel
# octaves with the augmented bass at both spots (the map's "1 hard").
# A passing C5 and an escape-tone G5 break the adjacency legally.
fin_plain = shift(transpose(SUBJECT, 12), u(23) + 16)      # 368-400
fin_sop = []
for n in fin_plain:
    rel = n.on - (u(23) + 16)
    if rel == 6:    # Bb4 8th -> Bb4 16th + C5 16th (passing into D5)
        fin_sop += [Note(n.on, 1, 70), Note(n.on + 1, 1, 72)]
    elif rel == 14:  # E5 8th -> E5 16th + G5 16th (échappée into F5)
        fin_sop += [Note(n.on, 1, 76), Note(n.on + 1, 1, 79)]
    else:
        fin_sop.append(n)
log(f"augmentation pair (ornamented): {pair_stats(aug_bass, fin_sop)}")

alto_pedal_fin = compose(
    fixed_voices={SOP: sop + fin_sop, BASS: bass},
    voice_slot=ALTO, n_voices=3,
    region=(alto[-1].on, u(24)), key_track=KEY_TRACK, vrange=RANGES[ALTO],
    prior=StylePrior(complement_bonus=1.2, suspension_seek=3.0,
                     tessitura_center=62, tessitura_width=10),
    seed_notes=(alto[-1],), beam_width=260,
)
add(alto, alto_pedal_fin, drop_before=alto[-1].off)
add(sop, fin_sop)

alto_fin = compose(
    fixed_voices={SOP: sop, BASS: bass},
    voice_slot=ALTO, n_voices=3,
    region=(alto[-1].on, u(27)), key_track=KEY_TRACK, vrange=RANGES[ALTO],
    prior=free_prior, seed_notes=(alto[-1],),
    end_pitch=62, end_by_step=True, beam_width=260,
)
add(alto, alto_fin, drop_before=alto[-1].off)

sop_fin = compose(
    fixed_voices={ALTO: alto, BASS: bass},
    voice_slot=SOP, n_voices=3,
    region=(sop[-1].on, u(27)), key_track=KEY_TRACK, vrange=(62, 81),
    prior=free_prior, seed_notes=(sop[-1],),
    end_pitch=70, end_by_step=True, beam_width=260,
)
add(sop, sop_fin, drop_before=sop[-1].off)
log("bars 23-26: finale composed")

# ============================================ bars 27-28: cadence (by hand)
# Draft 1 of this cadence had soprano and bass leaping a 4th into the final
# octave together — parallel octaves at the last chord, caught by my own
# checker. Rewritten: the soprano takes the 4-3 suspension over the
# dominant and resolves by half-step into the Picardy D.
sections.append({"name": "cadence — iv, 4-3 over V, Picardy", "bar": 27, "t": u(27)})
sop += [Note(u(27), 12, 74), Note(u(27)+12, 4, 73), Note(u(28), 16, 74)]
alto += [Note(u(27), 8, 58), Note(u(27)+8, 8, 57), Note(u(28), 16, 66)]
bass += [Note(u(27), 8, 43), Note(u(27)+8, 8, 45), Note(u(28), 16, 50)]

# ================================================================ verify
for v in (sop, alto, bass):
    v.sort(key=lambda n: n.on)
    for i in range(1, len(v)):
        assert v[i].on >= v[i-1].off, f"overlap at {v[i-1]} -> {v[i]}"

log("classifying full score...")
txt, events, problems = report(
    [sop, alto, bass], names=["S", "A", "B"],
    pedal_spans=[(BASS, u(21), u(23))], final_bar_start=u(28))
print("\n=== FULL-SCORE RULE REPORT ===")
print(txt)

# ================================================================= emit
music = {"sop": sop, "alto": alto, "bass": bass}
data = {
    "title": "RULES FOR A DEAF COMPOSER",
    "bars": 28, "end_units": END,
    "sections": sections,
    "voices": {k: [[n.on, n.dur, n.pitch] for n in v] for k, v in music.items()},
    "events": [{"t": e.t, "hi_v": e.hi_voice, "lo_v": e.lo_voice,
                "hi": e.hi, "lo": e.lo, "iv": e.interval,
                "kind": e.kind, "detail": e.detail} for e in events],
    "problems": [{"t": p.t, "kind": p.kind, "voices": list(p.voices),
                  "detail": p.detail, "hard": p.hard} for p in problems],
}
with open("fugue.json", "w") as f:
    json.dump(data, f)

ly = emit.emit(sop, alto, bass, KEY_TRACK, END,
               title="RULES FOR A DEAF COMPOSER",
               subtitle="fugue in three voices, D minor",
               date="2026-08-11", tempo_mark="Moderato, severo")
with open("fugue.ly", "w", encoding="utf-8") as f:
    f.write(ly)
log("wrote fugue.json + fugue.ly")
