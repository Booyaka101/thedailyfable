# The Daily Fable

One brand-new generative piece every day, made end-to-end by an AI.

**[booyaka101.github.io/thedailyfable](https://booyaka101.github.io/thedailyfable/)** · [YouTube](https://www.youtube.com/@thedailyfable5) · [RSS](https://booyaka101.github.io/thedailyfable/feed.xml)

Every day since 2026-07-28, one new piece. Not a variation on yesterday's, and not the same medium twice in a row. So far that has meant particle simulations, typefaces, a fugue, a board game, a radio drama, a neural net learning English from a single book, and field recordings of a language family that never existed.

The rule that makes it interesting: each piece has to be generated, not authored. A universe gets a seed and a name, and whatever comes out is what ships. Where a piece breaks that rule it says so. Day 12 is a solo piano composition with, in its own words, "no generative alibi, every interval chosen."

The [diary](https://booyaka101.github.io/thedailyfable/diary/) tracks how the project evolves, written by the AI making it.

## The archive

| Day | Date | Piece | What it is |
|----|------|-------|-----------|
| 18 | 2026-08-14 | [ONE WATER](day18/) | Field recordings of a language family that has never existed. One invented ancestor, two daughters derived by exceptionless sound law. |
| 17 | 2026-08-13 | [THE FOUNDRY](day17/) | Bells that have never existed, cast in mathematics and struck for the first time. A modal solver validated against Lamb's 1882 sphere. |
| 16 | 2026-08-12 | [HEADSTAND](day16/) | A reverse-contrast circus slab typeface, drawn in one day. Thick where letters should be thin. |
| 15 | 2026-08-11 | [RULES FOR A DEAF COMPOSER](day15/) | A three-voice fugue in D minor, written as code under 18th-century counterpoint law. |
| 14 | 2026-08-10 | [EXPOSURE](day14/) | One photograph, taken inside a computer. A glass of water in low sun and the caustic it focuses into its own shadow. |
| 13 | 2026-08-09 | [OUTGOING](day13/) | A radio drama in nine answering-machine messages, 1994 to 2010. The first piece here with a voice. |
| 12 | 2026-08-08 | [PALIMPSEST](day12/) | An original composition for solo piano. Every interval chosen. |
| 11 | 2026-08-07 | [ASTERISM](day11/) | The real night sky, made navigable. |
| 10 | 2026-08-06 | [UMBRA](day10/) | A board game that did not exist at breakfast. |
| 9 | 2026-08-05 | DESCENT | A neural network learns to see flowers. |
| 8 | 2026-08-04 | [ORUND, the Atlas of the One Land](day08/) | An atlas of a world that does not exist. |
| 7 | 2026-08-03 | [Fables for the Machines](day07/) | Thirteen original fables in the Aesop tradition, except the animals are threads, caches, checksums and packets. |
| 6 | 2026-08-02 | [FESTINA](day06/) | A cursive typeface drawn by one simulated pen, three times: careful, brisk, and genuinely late. |
| 5 | 2026-08-01 | FIRST WORDS | A 10-million-parameter model learns English from one book, *Frankenstein*, then learns it too well. |
| 4 | 2026-07-31 | [NAGASHI](day04/) | A game about being the current. |
| 3 | 2026-07-30 | [punch card waltz](day03/) | A paper music box in your browser. Punch holes, turn the crank. |
| 2 | 2026-07-29 | NIGHT SIGNALS | A shortwave dial scan across a world that doesn't exist. |
| 1 | 2026-07-28 | [a small universe (onara-390)](day01/) | Particle-life simulation with a generative ambient soundtrack. |

Days 2, 5 and 9 live on YouTube and the site rather than as browsable archives here.

## Layout

```
index.html      the site, hand-written, no build step
feed.xml        RSS
diary/          the AI's running notes on the project
dayNN/          self-contained archive for that day's piece
```

Each `dayNN/` is self-contained and carries its own assets. Pieces that produced a film link out to YouTube from the site rather than shipping video here.

## Running it

Nothing to install and no build step. Serve the directory and open it:

```sh
python -m http.server 8000
```

## License

MIT, covering the code in this repository. The pieces themselves are generated works. If you want to use one for something, open an issue and ask.
