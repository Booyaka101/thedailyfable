// UCT with tactical playouts: a playout takes an immediate win when a sampled move
// gives one, and re-samples a few times to avoid an immediate loss. At the root every
// legal move is checked one ply deep (and the reply two plies) before search starts.

export function rng(seed) {
  let a = seed >>> 0 || 1;
  return () => { a ^= a << 13; a >>>= 0; a ^= a >> 17; a ^= a << 5; a >>>= 0; return a / 4294967296; };
}

// outcome of playing m from s for the mover: 1 win, -1 loss, 0 nothing decided
function peek(game, s, m) {
  const t = game.clone(s), mover = t.player;
  game.play(t, m);
  if (!t.over) return 0;
  return t.winner === mover ? 1 : t.winner === 0 ? 0 : -1;
}

export function playout(game, s, rand, tries = 4) {
  const cap = s.moveNo + 2 * game.N + 20;
  while (!s.over) {
    const moves = game.legalMoves(s);
    let pick = -1;
    for (let k = 0; k < tries; k++) {
      const m = moves[(rand() * moves.length) | 0];
      const o = peek(game, s, m);
      if (o === 1) { pick = m; break; }
      if (o === 0 && pick < 0) pick = m;
    }
    if (pick < 0) pick = moves[(rand() * moves.length) | 0];
    game.play(s, pick);
    if (s.moveNo > cap) return 0;
  }
  return s.winner;
}

export function mcts(game, root, iters, rand, C = 1.0, opts = {}) {
  const rootMoves = game.legalMoves(root);
  // one-ply: take a win, drop losses; two-ply: drop moves that hand the opponent a win
  if (opts.lookahead !== false) {
    const safe = [];
    for (const m of rootMoves) {
      const o = peek(game, root, m);
      if (o === 1) return { move: m, value: 1, visits: null, moves: rootMoves, forced: 'win' };
      if (o === 0) safe.push(m);
    }
    const safer = [];
    for (const m of safe) {
      const t = game.clone(root); game.play(t, m);
      if (t.over) { safer.push(m); continue; }
      let bad = false;
      for (const r of game.legalMoves(t)) { if (peek(game, t, r) === 1) { bad = true; break; } }
      if (!bad) safer.push(m);
    }
    const use = safer.length ? safer : safe.length ? safe : rootMoves;
    if (use.length === 1) return { move: use[0], value: 0.5, visits: null, moves: rootMoves, forced: 'only' };
    return search(game, root, iters, rand, C, use);
  }
  return search(game, root, iters, rand, C, rootMoves);
}

function search(game, root, iters, rand, C, rootMoves) {
  const rootNode = { moves: rootMoves, kids: new Array(rootMoves.length).fill(null), N: 0, W: 0, who: 3 - root.player };
  for (let it = 0; it < iters; it++) {
    let node = rootNode, s = game.clone(root);
    const path = [node];
    while (true) {
      if (s.over) break;
      if (node.moves === null) { node.moves = game.legalMoves(s); node.kids = new Array(node.moves.length).fill(null); }
      const n = node.moves.length; let bi = -1, bv = -1e9;
      const logN = Math.log(node.N + 1);
      for (let i = 0; i < n; i++) {
        const k = node.kids[i];
        const v = k === null ? 1e6 + rand() : k.W / k.N + C * Math.sqrt(logN / k.N);
        if (v > bv) { bv = v; bi = i; }
      }
      const mover = s.player;
      game.play(s, node.moves[bi]);
      let k = node.kids[bi];
      if (k === null) { k = node.kids[bi] = { moves: null, kids: null, N: 0, W: 0, who: mover }; path.push(k); break; }
      node = k; path.push(k);
    }
    const w = s.over ? s.winner : playout(game, s, rand);
    for (const nd of path) { nd.N++; nd.W += w === 0 ? 0.5 : w === nd.who ? 1 : 0; }
  }
  let bi = 0, bn = -1;
  for (let i = 0; i < rootNode.moves.length; i++) { const k = rootNode.kids[i]; if (k && k.N > bn) { bn = k.N; bi = i; } }
  const k = rootNode.kids[bi];
  return { move: rootNode.moves[bi], value: k.W / k.N, visits: rootNode.kids.map(x => x ? x.N : 0), moves: rootNode.moves };
}

export function randomMove(game, s, rand) {
  const m = game.legalMoves(s);
  return m[(rand() * m.length) | 0];
}
