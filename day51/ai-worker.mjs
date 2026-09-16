import { Game } from './engine.mjs';
import { mcts, rng } from './mcts.mjs';

let game = null, rulesKey = '';
const rand = rng((Date.now() & 0xffff) | 1);

onmessage = e => {
  const { id, rules, moves, iters } = e.data;
  const k = JSON.stringify(rules);
  if (k !== rulesKey) { game = new Game(rules); rulesKey = k; }
  const s = game.newState();
  for (const m of moves) game.play(s, m);
  if (s.over) { postMessage({ id, move: -1, value: 0.5 }); return; }
  const t0 = performance.now();
  const r = mcts(game, s, iters, rand);
  postMessage({ id, move: r.move, value: r.value, ms: performance.now() - t0, visits: r.visits, moves: r.moves });
};
