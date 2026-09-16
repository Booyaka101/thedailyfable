// Mounts a playable board for any rule set. AI moves come from ai-worker.mjs.
import { Game, ruleText } from './engine.mjs';
import { layout, boardSVG } from './draw.mjs';

export function mount(root, rules, opts = {}) {
  const game = new Game(rules);
  const colors = opts.colors || { bg: '#f3ead8', cell: '#e5d9bf', line: '#8a7a5a', p1: '#1a1a1a', p2: '#f6f1e6', p2s: '#333', hi: '#c8412b' };
  const worker = new Worker(new URL('./ai-worker.mjs', import.meta.url), { type: 'module' });
  let moves = [], state = game.newState(), human = 1, iters = opts.iters || 3000, thinking = false, reqId = 0, lastValue = null;
  const svgHolder = document.createElement('div');
  svgHolder.className = 'board';
  root.appendChild(svgHolder);
  const status = opts.status || document.createElement('div');
  if (!opts.status) { status.className = 'status'; root.appendChild(status); }

  function render() {
    const size = 100;
    svgHolder.innerHTML = boardSVG(game.b, state.board, { size, last: state.last, colors, sides: game.win.some(w => w.k === 'connect') ? 'yes' : 'none' });
    const svg = svgHolder.firstChild;
    svg.removeAttribute('width'); svg.removeAttribute('height');
    svg.style.width = '100%'; svg.style.height = 'auto'; svg.style.display = 'block'; svg.style.touchAction = 'manipulation';
    svg.addEventListener('pointerdown', onTap);
    const who = state.player === human ? 'your move' : 'thinking';
    let txt;
    if (state.over) txt = state.winner === 0 ? 'draw' : state.winner === human ? 'you win' : 'the machine wins';
    else txt = who;
    if (game.turn === 2 && !state.over) txt += state.left === 2 ? ' (1 of 2)' : ' (2 of 2)';
    if (game.capture !== 'none' && (state.caps[1] || state.caps[2])) txt += ` · captures ${state.caps[human]}:${state.caps[3 - human]}`;
    status.textContent = txt;
    if (opts.onState) opts.onState({ state, moves, human, lastValue, game });
  }
  function onTap(ev) {
    if (thinking || state.over || state.player !== human) return;
    const svg = ev.currentTarget, rect = svg.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / rect.width * 100, y = (ev.clientY - rect.top) / rect.height * 100;
    const L = layout(game.b, 100); let best = -1, bd = 1e9;
    for (let i = 0; i < game.N; i++) { const d = Math.hypot(L.pts[i][0] - x, L.pts[i][1] - y); if (d < bd) { bd = d; best = i; } }
    if (best < 0 || bd > L.s * 0.55) return;
    const legal = game.legalMoves(state);
    if (!legal.includes(best)) { flash(best); return; }
    doMove(best);
    if (!state.over && state.player !== human) aiMove();
  }
  function flash(cell) { status.textContent = 'not allowed there'; setTimeout(render, 700); }
  function doMove(m) { moves.push(m); game.play(state, m); render(); }
  function aiMove() {
    thinking = true; const id = ++reqId; render();
    worker.postMessage({ id, rules, moves: moves.slice(), iters });
  }
  worker.onmessage = e => {
    if (e.data.id !== reqId) return;
    thinking = false;
    if (e.data.move >= 0) { lastValue = e.data.value; doMove(e.data.move); if (!state.over && state.player !== human) aiMove(); }
    else render();
  };
  function newGame(humanFirst = true) {
    reqId++; thinking = false; moves = []; state = game.newState(); human = humanFirst ? 1 : 2; lastValue = null; render();
    if (!humanFirst) aiMove();
  }
  function undo() {
    if (thinking) return;
    // undo back to the last position where the human was to move
    const keep = moves.slice();
    do { keep.pop(); } while (keep.length && replay(keep).player !== human);
    moves = keep; state = replay(moves); reqId++; render();
  }
  function replay(ms) { const s = game.newState(); for (const m of ms) game.play(s, m); return s; }
  function setIters(n) { iters = n; }
  render();
  return { newGame, undo, setIters, get state() { return state; }, get moves() { return moves; }, game, text: ruleText(rules) };
}
