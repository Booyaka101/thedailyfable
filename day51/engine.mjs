// Generic placement-game engine. A "rules" object picks a board, a placement
// constraint, a capture effect, win/lose conditions and a board-full rule.
// The same file runs the search in node and the opponent in the browser.

export const PLACE = ['any', 'adjAny', 'adjEnemy', 'notAdjOwn', 'gravity'];
export const CAPTURE = ['none', 'custodial', 'flip', 'surround'];
export const FULL = ['draw', 'majority', 'lastWins', 'lastLoses'];

export function makeBoard(spec) {
  const cells = []; // {x,y} for sq; {q,r} for hex
  const key = new Map();
  let dirs, adjDirs, sideOf, type = spec.type;
  if (type === 'sq') {
    const n = spec.n;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { key.set(x + ',' + y, cells.length); cells.push({ x, y }); }
    dirs = [[1, 0], [0, 1], [1, 1], [1, -1]];
    adjDirs = spec.adj === 8 ? [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]] : [[1, 0], [-1, 0], [0, 1], [0, -1]];
    // sides: 0 left 1 right 2 top 3 bottom
    sideOf = c => (c.x === 0 ? 1 : 0) | (c.x === n - 1 ? 2 : 0) | (c.y === 0 ? 4 : 0) | (c.y === n - 1 ? 8 : 0);
  } else if (type === 'rhombus') {
    const n = spec.n;
    for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) { key.set(q + ',' + r, cells.length); cells.push({ x: q, y: r }); }
    dirs = [[1, 0], [0, 1], [1, -1]];
    adjDirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];
    sideOf = c => (c.x === 0 ? 1 : 0) | (c.x === n - 1 ? 2 : 0) | (c.y === 0 ? 4 : 0) | (c.y === n - 1 ? 8 : 0);
  } else if (type === 'hexhex') {
    const R = spec.r;
    for (let r = -R; r <= R; r++) for (let q = -R; q <= R; q++) {
      if (Math.abs(q + r) > R) continue;
      key.set(q + ',' + r, cells.length); cells.push({ x: q, y: r });
    }
    dirs = [[1, 0], [0, 1], [1, -1]];
    adjDirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];
    // six sides, opposite pairs (0,1) (2,3) (4,5)
    sideOf = c => { const s = -c.x - c.y; return (c.x === R ? 1 : 0) | (c.x === -R ? 2 : 0) | (c.y === R ? 4 : 0) | (c.y === -R ? 8 : 0) | (s === R ? 16 : 0) | (s === -R ? 32 : 0); };
  } else throw new Error('board type ' + type);
  const N = cells.length;
  const at = (x, y) => { const i = key.get(x + ',' + y); return i === undefined ? -1 : i; };
  const nbr = [], rays = [], side = new Uint8Array(N), col = new Int16Array(N), below = new Int16Array(N);
  for (let i = 0; i < N; i++) {
    const c = cells[i];
    nbr.push(Int16Array.from(adjDirs.map(d => at(c.x + d[0], c.y + d[1])).filter(j => j >= 0)));
    const rr = [];
    for (const d of dirs) for (const s of [1, -1]) {
      const ray = []; let x = c.x, y = c.y;
      for (; ;) { x += d[0] * s; y += d[1] * s; const j = at(x, y); if (j < 0) break; ray.push(j); }
      rr.push(Int16Array.from(ray));
    }
    rays.push(rr);
    side[i] = sideOf(c);
    col[i] = c.x;
    below[i] = type === 'sq' ? at(c.x, c.y + 1) : -1;
  }
  return { type, spec, N, cells, nbr, rays, side, col, below, D: dirs.length };
}

export class Game {
  constructor(rules) {
    this.rules = rules;
    this.b = makeBoard(rules.board);
    this.N = this.b.N;
    this.mark = new Int32Array(this.N);
    this.stamp = 0;
    this.stack = new Int16Array(this.N);
    this.grp = new Int16Array(this.N);
    this.turn = rules.turn || 1;
    this.win = rules.win || [];
    this.lose = rules.lose || [];
    this.place = rules.place || 'any';
    this.capture = rules.capture || 'none';
    this.full = rules.full || 'draw';
    this.connectMode = rules.connect || 'own'; // own | any | opposite (hexhex)
  }
  newState() {
    const N = this.N;
    const s = { board: new Int8Array(N), empt: new Int16Array(N), pos: new Int16Array(N), ne: N, player: 1, left: 1, over: false, winner: 0, moveNo: 0, caps: new Int16Array(3), last: -1, how: '' };
    for (let i = 0; i < N; i++) { s.empt[i] = i; s.pos[i] = i; }
    return s;
  }
  clone(s) {
    return { board: s.board.slice(), empt: s.empt.slice(), pos: s.pos.slice(), ne: s.ne, player: s.player, left: s.left, over: s.over, winner: s.winner, moveNo: s.moveNo, caps: s.caps.slice(), last: s.last, how: s.how };
  }
  _rm(s, c) { const p = s.pos[c], l = s.empt[--s.ne]; s.empt[p] = l; s.pos[l] = p; }
  _add(s, c) { s.empt[s.ne] = c; s.pos[c] = s.ne; s.ne++; }
  legalMoves(s) {
    const out = [];
    if (s.over) return out;
    const b = this.b, bd = s.board, p = s.player, mode = this.place;
    if (mode === 'any') { for (let i = 0; i < s.ne; i++) out.push(s.empt[i]); return out; }
    if (mode === 'gravity') {
      for (let i = 0; i < s.ne; i++) { const c = s.empt[i]; const d = b.below[c]; if (d < 0 || bd[d] !== 0) out.push(c); }
      return out;
    }
    if (s.moveNo === 0) { for (let i = 0; i < s.ne; i++) out.push(s.empt[i]); return out; }
    for (let i = 0; i < s.ne; i++) {
      const c = s.empt[i], nb = b.nbr[c]; let ok;
      if (mode === 'adjAny') { ok = false; for (let k = 0; k < nb.length; k++) if (bd[nb[k]] !== 0) { ok = true; break; } }
      else if (mode === 'adjEnemy') { ok = false; for (let k = 0; k < nb.length; k++) if (bd[nb[k]] === 3 - p) { ok = true; break; } }
      else { ok = true; for (let k = 0; k < nb.length; k++) if (bd[nb[k]] === p) { ok = false; break; } }
      if (ok) out.push(c);
    }
    if (out.length === 0) for (let i = 0; i < s.ne; i++) out.push(s.empt[i]);
    return out;
  }
  // flood own group from c; returns size, sets sideMask & liberties
  _flood(s, c, p) {
    const b = this.b, bd = s.board, mark = this.mark, st = ++this.stamp, stack = this.stack, grp = this.grp;
    let n = 0, top = 0, sides = 0, libs = 0;
    stack[top++] = c; mark[c] = st;
    while (top) {
      const x = stack[--top]; grp[n++] = x; sides |= b.side[x];
      const nb = b.nbr[x];
      for (let k = 0; k < nb.length; k++) {
        const y = nb[k];
        if (bd[y] === p) { if (mark[y] !== st) { mark[y] = st; stack[top++] = y; } }
        else if (bd[y] === 0) libs++;
      }
    }
    this.fsides = sides; this.flibs = libs;
    return n;
  }
  _maxRun(s, c, p) {
    const rays = this.b.rays[c], bd = s.board; let best = 1;
    for (let d = 0; d < rays.length; d += 2) {
      let run = 1;
      const r1 = rays[d], r2 = rays[d + 1];
      for (let k = 0; k < r1.length && bd[r1[k]] === p; k++) run++;
      for (let k = 0; k < r2.length && bd[r2[k]] === p; k++) run++;
      if (run > best) best = run;
    }
    return best;
  }
  _connected(sides, p) {
    const m = this.connectMode, t = this.b.type;
    if (t === 'hexhex') {
      return ((sides & 3) === 3) || ((sides & 12) === 12) || ((sides & 48) === 48);
    }
    const lr = (sides & 3) === 3, tb = (sides & 12) === 12;
    if (m === 'any') return lr || tb;
    return p === 1 ? lr : tb;
  }
  _conds(list, s, c, p, maxRun, gsize, sides) {
    for (const w of list) {
      if (w.k === 'row') { if (maxRun >= w.n) return w.k; }
      else if (w.k === 'connect') { if (this._connected(sides, p)) return w.k; }
      else if (w.k === 'group') { if (gsize >= w.n) return w.k; }
      else if (w.k === 'captures') { if (s.caps[p] >= w.n) return w.k; }
    }
    return null;
  }
  play(s, c) {
    const b = this.b, bd = s.board, p = s.player, e = 3 - p;
    bd[c] = p; this._rm(s, c); s.last = c; s.moveNo++;
    // capture effects
    const cap = this.capture;
    if (cap === 'custodial' || cap === 'flip') {
      const rays = b.rays[c];
      for (let d = 0; d < rays.length; d++) {
        const r = rays[d]; let k = 0;
        while (k < r.length && bd[r[k]] === e) k++;
        if (k > 0 && k < r.length && bd[r[k]] === p) {
          for (let j = 0; j < k; j++) { const x = r[j]; if (cap === 'flip') bd[x] = p; else { bd[x] = 0; this._add(s, x); } s.caps[p]++; }
        }
      }
    } else if (cap === 'surround') {
      const nb = b.nbr[c], st0 = this.stamp;
      for (let k = 0; k < nb.length; k++) {
        const y = nb[k];
        if (bd[y] !== e || this.mark[y] > st0) continue;
        const n = this._flood(s, y, e);
        if (this.flibs === 0) { for (let j = 0; j < n; j++) { const x = this.grp[j]; bd[x] = 0; this._add(s, x); } s.caps[p] += n; }
      }
    }
    // outcome checks
    const needRun = this._needs.row, needFlood = this._needs.flood;
    const maxRun = needRun ? this._maxRun(s, c, p) : 0;
    let gsize = 0, sides = 0;
    if (needFlood) { gsize = this._flood(s, c, p); sides = this.fsides; }
    let how;
    if ((how = this._conds(this.win, s, c, p, maxRun, gsize, sides))) { s.over = true; s.winner = p; s.how = 'win:' + how; return s; }
    if ((how = this._conds(this.lose, s, c, p, maxRun, gsize, sides))) { s.over = true; s.winner = e; s.how = 'lose:' + how; return s; }
    if (s.ne === 0) {
      s.over = true; s.how = 'full';
      const f = this.full;
      if (f === 'draw') s.winner = 0;
      else if (f === 'lastWins') s.winner = p;
      else if (f === 'lastLoses') s.winner = e;
      else { let a = 0, bb = 0; for (let i = 0; i < this.N; i++) { if (bd[i] === 1) a++; else if (bd[i] === 2) bb++; } s.winner = a > bb ? 1 : bb > a ? 2 : 0; }
      return s;
    }
    if (--s.left <= 0) { s.player = e; s.left = this.turn; }
    return s;
  }
  get _needs() {
    if (!this.__needs) {
      const all = [...this.win, ...this.lose];
      this.__needs = { row: all.some(w => w.k === 'row'), flood: all.some(w => w.k === 'connect' || w.k === 'group') };
    }
    return this.__needs;
  }
}

// ---- named rule sets (controls) ----
export const CONTROLS = {
  tictactoe: { board: { type: 'sq', n: 3, adj: 8 }, win: [{ k: 'row', n: 3 }] },
  gomoku9: { board: { type: 'sq', n: 9, adj: 8 }, win: [{ k: 'row', n: 5 }] },
  hex7: { board: { type: 'rhombus', n: 7 }, win: [{ k: 'connect' }], full: 'draw' },
  yavalath: { board: { type: 'hexhex', r: 4 }, win: [{ k: 'row', n: 4 }], lose: [{ k: 'row', n: 3 }] },
  othello6: { board: { type: 'sq', n: 6, adj: 8 }, place: 'adjEnemy', capture: 'flip', full: 'majority' },
  connect6_9: { board: { type: 'sq', n: 9, adj: 8 }, turn: 2, win: [{ k: 'row', n: 6 }] },
  connect4: { board: { type: 'sq', n: 7, adj: 8 }, place: 'gravity', win: [{ k: 'row', n: 4 }] },
  pente7: { board: { type: 'sq', n: 7, adj: 8 }, capture: 'custodial', win: [{ k: 'row', n: 5 }, { k: 'captures', n: 6 }] },
  atari9: { board: { type: 'sq', n: 7, adj: 4 }, capture: 'surround', win: [{ k: 'captures', n: 1 }] },
};

export function ruleText(r) {
  const b = r.board, parts = [];
  parts.push(b.type === 'sq' ? `${b.n}x${b.n} square board (${b.adj}-connected)` : b.type === 'rhombus' ? `${b.n}x${b.n} hex rhombus` : `hexagonal board, ${b.r + 1} on a side`);
  parts.push(r.turn === 2 ? 'players place two stones per turn (first turn one)' : 'players alternate placing one stone');
  const pl = { any: null, adjAny: 'a stone must touch an existing stone', adjEnemy: 'a stone must touch an enemy stone', notAdjOwn: 'a stone may not touch a friendly stone', gravity: 'stones drop to the lowest free cell in their column' }[r.place || 'any'];
  if (pl) parts.push(pl);
  const cp = { none: null, custodial: 'enemy stones sandwiched in a line are removed', flip: 'enemy stones sandwiched in a line are flipped', surround: 'enemy groups with no free neighbour are removed' }[r.capture || 'none'];
  if (cp) parts.push(cp);
  const cond = w => w.k === 'row' ? `${w.n} in a row` : w.k === 'connect' ? (b.type === 'hexhex' ? 'connecting two opposite sides' : r.connect === 'any' ? 'connecting either pair of opposite sides' : 'connecting your two sides') : w.k === 'group' ? `a group of ${w.n}` : `${w.n} captures`;
  if (r.win && r.win.length) parts.push('win: ' + r.win.map(cond).join(' or '));
  if (r.lose && r.lose.length) parts.push('lose: ' + r.lose.map(cond).join(' or '));
  const f = { draw: 'full board is a draw', majority: 'full board: most stones wins', lastWins: 'full board: last to place wins', lastLoses: 'full board: last to place loses' }[r.full || 'draw'];
  parts.push(f);
  return parts.join('; ');
}
