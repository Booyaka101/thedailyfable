// A playable hex-hex board bound to the wasm engine in worker.js. One factory, one board:
//   const game = createGame({ canvas, status, meter, hint, rules, side, level, wasmUrl });
//   game.setCode('s6w0l3c0k0K0a2p1');   // switches rules, rebuilds geometry and rules text, starts a game
// The rule genes mirror engine/src/game.rs; the geometry mirrors Geom there.
'use strict';

function parseCode(c) {
  const m = String(c || '').trim().match(/^s([456])w([03456])l([0345])c([012])k([012])K([0358])a([012])p([12])$/);
  return m ? m.slice(1, 9).map(Number) : null;
}

function rulesText(R, N) {
  const [, WIN_N, LOSE_N, CONNECT, CAPTURE, CAPTURE_WIN, ADJ, PER_TURN] = R;
  const li = [];
  li.push(`Two players, light and dark, take turns placing a stone on an empty cell of the ${N}-cell board. Light starts.`);
  if (PER_TURN === 2) li.push('Light\'s first turn is a single stone. After that every turn is two stones.');
  if (ADJ === 1) li.push('A stone must be placed touching at least one stone already on the board (the first stone may go anywhere).');
  if (ADJ === 2) li.push('A stone must be placed touching one of your own stones (your first stone may go anywhere).');
  if (CAPTURE === 1) li.push('Custodial capture: if your stone lands so that a single enemy stone is sandwiched in a straight line between it and another of your stones, the enemy stone is removed. Landing between two enemy stones is safe.');
  if (CAPTURE === 2) li.push('Custodial capture: if your stone lands so that an unbroken straight line of enemy stones is sandwiched between it and another of your stones, all of them are removed. Landing between enemy stones is safe.');
  if (WIN_N) li.push(`Make a straight line of ${WIN_N} or more of your stones and you win at once.`);
  if (LOSE_N) li.push(WIN_N ? `But make a straight line of exactly ${LOSE_N} (with no ${WIN_N}) and you lose at once.` : `Make a straight line of exactly ${LOSE_N} of your stones and you lose at once. Longer lines are safe, but you can only get one by filling a gap.`);
  if (CONNECT === 1) li.push('Connect any two opposite sides of the board with an unbroken chain of your stones and you win. Corner cells count for both sides they touch.');
  if (CONNECT === 2) li.push('Connect any three sides of the board with one unbroken chain of your stones and you win. Corner cells count for both sides they touch.');
  if (CAPTURE_WIN) li.push(`Capture ${CAPTURE_WIN} stones in total and you win.`);
  li.push('If the board fills, or neither player can place, the game is a draw.');
  return li;
}

function geometry(SIZE) {
  const s = SIZE - 1, coords = [];
  for (let q = -s; q <= s; q++) for (let r = -s; r <= s; r++) if (Math.abs(q + r) <= s) coords.push([q, r]);
  const sideMask = coords.map(([q, r]) => {
    const sv = -(q + r); let m = 0;
    if (q === s) m |= 1; if (r === -s) m |= 2; if (sv === s) m |= 4; if (q === -s) m |= 8; if (r === s) m |= 16; if (sv === -s) m |= 32;
    return m;
  });
  return { coords, sideMask, N: coords.length, cellsAcross: 2 * SIZE - 1 };
}

const SIDE_COL = ['#c96a3c', '#5b8fd6', '#7cb26b', '#c96a3c', '#5b8fd6', '#7cb26b'];

function createGame(el) {
  const canvas = el.canvas, ctx = canvas.getContext('2d');
  const coarse = matchMedia('(pointer: coarse)').matches;
  let R = null, G = null, code = null;
  let hexR = 24, ox = 0, oy = 0, DPR = 1, cssW = 0, cssH = 0;
  let state = null, busy = false, humanSide = 1, lastCapCells = [], ready = false, pendingCode = null;
  const listeners = [];

  const worker = new Worker(el.workerUrl || 'worker.js');
  let msgId = 0; const pending = new Map();
  const send = m => new Promise(res => { m.id = ++msgId; pending.set(m.id, res); worker.postMessage(m); });
  worker.onmessage = e => {
    const m = e.data;
    if (m.op === 'ready') { ready = true; if (pendingCode) setCode(pendingCode); return; }
    const res = pending.get(m.id); pending.delete(m.id); res && res(m);
  };
  worker.postMessage({ op: 'init', wasmUrl: el.wasmUrl || 'ludus.wasm' });

  function layout() {
    if (!G) return;
    cssW = canvas.clientWidth || Math.min(720, window.innerWidth - 24);
    const margin = 10;
    hexR = Math.max(17, Math.min(34, (cssW - 2 * margin) / (G.cellsAcross * Math.sqrt(3) + 0.2)));
    cssH = Math.ceil(hexR * (1.5 * (G.cellsAcross - 1) + 2) + 2 * margin);
    DPR = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * DPR); canvas.height = Math.round(cssH * DPR);
    canvas.style.height = cssH + 'px';
    ox = cssW / 2; oy = cssH / 2;
    draw();
  }
  const center = i => { const [q, r] = G.coords[i]; return [ox + hexR * Math.sqrt(3) * (q + r / 2), oy + hexR * 1.5 * r]; };
  function hexPath(x, y, rad) {
    ctx.beginPath();
    for (let k = 0; k < 6; k++) { const a = Math.PI / 180 * (60 * k - 30); const px = x + rad * Math.cos(a), py = y + rad * Math.sin(a); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.closePath();
  }
  function draw() {
    if (!G) return;
    const CONNECT = R[3];
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    for (let i = 0; i < G.N; i++) {
      const [x, y] = center(i);
      hexPath(x, y, hexR - 1.2);
      ctx.fillStyle = '#3a3229';
      ctx.fill();
      if (CONNECT) {
        const m = G.sideMask[i];
        for (let sd = 0; sd < 6; sd++) if (m & (1 << sd)) {
          const a = Math.PI / 180 * [-30, -90, -150, 150, 90, 30][sd];
          ctx.beginPath();
          ctx.arc(x + Math.cos(a) * hexR * 0.62, y + Math.sin(a) * hexR * 0.62, hexR * 0.16, 0, Math.PI * 2);
          ctx.fillStyle = CONNECT === 1 ? SIDE_COL[sd] : '#b8a27a';
          ctx.fill();
        }
      }
      const v = state ? state.cells[i] : 0;
      if (v) {
        ctx.beginPath(); ctx.arc(x, y, hexR * 0.66, 0, Math.PI * 2);
        ctx.fillStyle = v === 1 ? '#efe9dc' : '#1b1815'; ctx.fill();
        ctx.lineWidth = 1.2; ctx.strokeStyle = v === 1 ? '#bdb4a3' : '#4a423a'; ctx.stroke();
      }
      if (state && state.lastMove === i && v) {
        ctx.beginPath(); ctx.arc(x, y, hexR * 0.2, 0, Math.PI * 2);
        ctx.fillStyle = v === 1 ? '#c98d3a' : '#d8a24a'; ctx.fill();
      }
    }
    for (const i of lastCapCells) { const [x, y] = center(i); ctx.beginPath(); ctx.arc(x, y, hexR * 0.3, 0, Math.PI * 2); ctx.strokeStyle = '#d8a24a'; ctx.lineWidth = 2; ctx.stroke(); }
  }
  function cellAt(px, py) {
    let best = -1, bd = 1e9;
    for (let i = 0; i < G.N; i++) { const [x, y] = center(i); const d = (x - px) ** 2 + (y - py) ** 2; if (d < bd) { bd = d; best = i; } }
    return bd <= (hexR * 0.95) ** 2 ? best : -1;
  }

  function setStatus() {
    if (!state || !el.status) return;
    const who = p => (p === 1 ? 'Light' : 'Dark');
    const CAPTURE = R[4], PER_TURN = R[7];
    let t;
    if (state.outcome === 3) t = 'Draw.';
    else if (state.outcome) t = who(state.outcome) + ' wins' + (state.outcome === humanSide ? '. That is you.' : '.');
    else t = (state.toMove === humanSide ? 'Your move' : 'AI thinking…') + (PER_TURN === 2 && state.stonesLeft === 2 && state.moves > 0 ? ' (two stones)' : '');
    if (CAPTURE) t += `  ·  captures ${state.cap1} : ${state.cap2}`;
    el.status.textContent = t;
    canvas.dataset.moves = state.moves; canvas.dataset.outcome = state.outcome;
    if (el.meter && typeof state.aiValue === 'number') {
      const light = (3 - humanSide) === 1 ? state.aiValue : 1 - state.aiValue;
      el.meter.style.width = (light * 100).toFixed(0) + '%';
      el.meter.title = 'AI estimate of Light\'s chances: ' + (light * 100).toFixed(0) + '%';
    }
    for (const f of listeners) f(state);
  }
  function markCaptures(prev, next) {
    lastCapCells = [];
    if (!prev) return;
    for (let i = 0; i < G.N; i++) if (prev.cells[i] && !next.cells[i]) lastCapCells.push(i);
  }
  async function apply(promise) {
    const prev = state; const m = await promise; if (m.state) { markCaptures(prev, m.state); state = m.state; }
    draw(); setStatus();
  }
  const level = () => (el.level ? +el.level.value : 3000);
  let gen = 0;  // bumped on every new game so a stale AI loop stops touching the board
  async function aiTurn() {
    const g = gen;
    while (g === gen && state && !state.outcome && state.toMove !== humanSide) {
      busy = true; setStatus();
      await new Promise(r => setTimeout(r, 30));
      if (g !== gen) break;
      await apply(send({ op: 'ai', iters: level() }));
      busy = false;
    }
    if (g === gen && state && !state.outcome && state.toMove === humanSide && !state.hasLegal) { await apply(send({ op: 'pass' })); return aiTurn(); }
  }
  async function newGame() {
    if (!R) return;
    gen++; busy = false;
    humanSide = el.side ? +el.side.value : 1;
    lastCapCells = []; state = null;
    await apply(send({ op: 'rules', r: R, seed: (Math.random() * 1e9) | 0 }));
    aiTurn();
  }
  function setCode(c) {
    const r = parseCode(c);
    if (!r) return false;
    if (!ready) { pendingCode = c; return true; }
    pendingCode = null;
    code = c; R = r; G = geometry(R[0]);
    if (el.rules) el.rules.innerHTML = '<h2>Rules</h2><ol>' + rulesText(R, G.N).map(t => `<li>${t}</li>`).join('') + '</ol>';
    if (el.hint) el.hint.textContent = (coarse ? 'Tap' : 'Click') + ' a cell to place a stone.' + (R[4] ? ' Captured stones flash gold.' : '');
    newGame().then(layout);
    layout();
    return true;
  }
  async function undo() {
    if (busy || !state) return;
    let guard = 0;
    do { await apply(send({ op: 'undo' })); guard++; } while (state.moves > 0 && state.toMove !== humanSide && guard < 8);
  }

  if (el.newBtn) el.newBtn.onclick = newGame;
  if (el.side) el.side.onchange = newGame;
  if (el.undoBtn) el.undoBtn.onclick = undo;
  canvas.addEventListener('pointerdown', async e => {
    if (busy || !state || state.outcome || state.toMove !== humanSide) return;
    const rect = canvas.getBoundingClientRect();
    const c = cellAt(e.clientX - rect.left, e.clientY - rect.top);
    if (c < 0) return;
    const m = await send({ op: 'play', c });
    if (m.state.captured === -1) return;
    markCaptures(state, m.state); state = m.state; draw(); setStatus();
    aiTurn();
  });
  window.addEventListener('resize', layout);
  window.addEventListener('orientationchange', layout);

  return { setCode, newGame, undo, onState: f => listeners.push(f), get code() { return code; }, get state() { return state; }, get rules() { return R; } };
}
