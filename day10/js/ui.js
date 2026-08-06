// UMBRA — page logic. Board is SVG; net runs in-browser (see ai.js / mcts.js).

import * as E from "./engine.js";
import { MCTS, bestMove } from "./mcts.js";
import { loadLadder, getSession, makeEvalFn } from "./ai.js";

// ---------- geometry (pointy-top hexes, axial q,r) ----------
const S = 22, SQ3 = Math.sqrt(3);
const cx = (q, r) => SQ3 * S * (q + r / 2);
const cy = (q, r) => 1.5 * S * r;
const hexPts = (x, y, s = S) => {
  const p = [];
  for (let a = 0; a < 6; a++) {
    const t = Math.PI / 180 * (60 * a - 30);
    p.push(`${(x + s * Math.cos(t)).toFixed(2)},${(y + s * Math.sin(t)).toFixed(2)}`);
  }
  return p.join(" ");
};
const centers = E.IJ_OF_IDX.map(([i, j]) => [cx(i - 4, j - 4), cy(i - 4, j - 4)]);

// ---------- state ----------
const st = {
  board: E.newBoard(), ply: 0, humanColor: 1, rung: 3, ladder: [],
  busy: false, over: false, preview: -1, insight: null, insightOn: false,
  lastMove: -1,
  soundOn: localStorage.getItem("umbra-sound") !== "off",
};

const $ = (id) => document.getElementById(id);
const coarse = matchMedia("(pointer: coarse)").matches;
const SIMS = 128;

// ---------- svg construction ----------
const NS = "http://www.w3.org/2000/svg";
let cellEls = [], lanternG, rayG, previewG, insightG;

function buildBoard() {
  const svg = $("board");
  const xs = centers.map(c => c[0]), ys = centers.map(c => c[1]);
  const m = S * 1.35;
  const x0 = Math.min(...xs) - m, y0 = Math.min(...ys) - m;
  const w = Math.max(...xs) - Math.min(...xs) + 2 * m;
  const h = Math.max(...ys) - Math.min(...ys) + 2 * m;
  svg.setAttribute("viewBox", `${x0.toFixed(1)} ${y0.toFixed(1)} ${w.toFixed(1)} ${h.toFixed(1)}`);

  const tint = document.createElementNS(NS, "g");
  rayG = document.createElementNS(NS, "g");
  insightG = document.createElementNS(NS, "g");
  lanternG = document.createElementNS(NS, "g");
  previewG = document.createElementNS(NS, "g");
  for (const g of [tint, rayG, insightG, lanternG, previewG]) svg.appendChild(g);

  for (let k = 0; k < E.N_CELLS; k++) {
    const [x, y] = centers[k];
    const poly = document.createElementNS(NS, "polygon");
    poly.setAttribute("points", hexPts(x, y, S - 0.8));
    poly.setAttribute("class", "cell");
    poly.dataset.k = k;
    tint.appendChild(poly);
    cellEls.push(poly);
  }

  svg.addEventListener("pointerdown", onTap);
  if (!coarse) {
    svg.addEventListener("pointermove", onHover);
    svg.addEventListener("pointerleave", () => setPreview(-1));
  }
}

function cellFromEvent(ev) {
  const t = ev.target.closest?.(".cell");
  return t ? +t.dataset.k : -1;
}

// ---------- rendering ----------
function lanternEl(k, color, ghost = false) {
  const [x, y] = centers[k];
  const g = document.createElementNS(NS, "g");
  g.setAttribute("class", `lantern ${color === 1 ? "gold" : "silver"}${ghost ? " ghost" : ""}`);
  const halo = document.createElementNS(NS, "circle");
  halo.setAttribute("cx", x); halo.setAttribute("cy", y); halo.setAttribute("r", S * 0.78);
  halo.setAttribute("class", "halo");
  const core = document.createElementNS(NS, "circle");
  core.setAttribute("cx", x); core.setAttribute("cy", y); core.setAttribute("r", S * 0.34);
  core.setAttribute("class", "core");
  g.append(halo, core);
  return g;
}

function raysFor(board, k, color) {
  // segments from lantern k to each lit cell (stop before blockers)
  const segs = [];
  for (let d = 0; d < 6; d++) {
    let last = null;
    for (const c of E.RAYS[k][d]) {
      const [i, j] = E.IJ_OF_IDX[c];
      if (board[i * 9 + j] !== 0) break;
      last = c;
    }
    if (last !== null) segs.push([centers[k], centers[last]]);
  }
  return segs;
}

function drawRays(g, board, k, color, cls) {
  for (const [[x1, y1], [x2, y2]] of raysFor(board, k, color)) {
    const ln = document.createElementNS(NS, "line");
    ln.setAttribute("x1", x1); ln.setAttribute("y1", y1);
    ln.setAttribute("x2", x2); ln.setAttribute("y2", y2);
    ln.setAttribute("class", cls);
    g.appendChild(ln);
  }
}

function tintFor(owner, lg, ls) {
  if (!owner) return "transparent";
  const margin = Math.min(Math.abs(lg - ls), 4);
  const a = 0.11 + 0.06 * margin;
  return owner === 1 ? `rgba(255,183,84,${a})` : `rgba(158,180,255,${a})`;
}

function render() {
  const [lg, ls] = E.lightCounts(st.board);
  const own = E.ownership(st.board, [lg, ls]);
  for (let k = 0; k < E.N_CELLS; k++) {
    const [i, j] = E.IJ_OF_IDX[k];
    const v = st.board[i * 9 + j];
    const el = cellEls[k];
    el.classList.toggle("occupied", v !== 0);
    el.style.fill = v === 0 ? tintFor(own[k], lg[k], ls[k]) : "transparent";
  }
  lanternG.replaceChildren();
  for (let k = 0; k < E.N_CELLS; k++) {
    const [i, j] = E.IJ_OF_IDX[k];
    const v = st.board[i * 9 + j];
    if (v !== 0) lanternG.appendChild(lanternEl(k, v));
  }
  if (st.lastMove >= 0) {
    const [x, y] = centers[st.lastMove];
    const ring = document.createElementNS(NS, "circle");
    ring.setAttribute("cx", x); ring.setAttribute("cy", y);
    ring.setAttribute("r", S * 0.56);
    ring.setAttribute("class", "last-ring");
    lanternG.appendChild(ring);
  }
  renderScore(own);
  renderInsight();
}

function counts(own) {
  let g = 0, s = 0, sh = 0;
  for (let k = 0; k < E.N_CELLS; k++) {
    const [i, j] = E.IJ_OF_IDX[k];
    if (st.board[i * 9 + j] !== 0) continue;
    if (own[k] === 1) g++; else if (own[k] === 2) s++; else sh++;
  }
  return { g, s, sh };
}

function renderScore(own) {
  const { g, s, sh } = counts(own ?? E.ownership(st.board));
  $("score-gold").textContent = g;
  $("score-silver").textContent = s;
  $("score-shadow").textContent = sh;
  // lanterns remaining as dots (gold moves on even plies, silver on odd)
  const gRem = E.STONES_EACH - Math.floor((st.ply + 1) / 2);
  const sRem = E.STONES_EACH - Math.floor(st.ply / 2);
  $("dots-gold").textContent = "●".repeat(Math.max(gRem, 0));
  $("dots-silver").textContent = "●".repeat(Math.max(sRem, 0));
}

function status(msg) { $("status").innerHTML = msg; }

// ---------- preview ----------
function setPreview(k) {
  st.preview = k;
  previewG.replaceChildren();
  if (k < 0 || st.busy || st.over) { render(); return; }
  const me = E.playerToMove(st.ply);
  const b2 = st.board.slice();
  E.play(b2, k, me);
  // tint preview: render future ownership onto cells
  const [lg, ls] = E.lightCounts(b2);
  const own = E.ownership(b2, [lg, ls]);
  for (let kk = 0; kk < E.N_CELLS; kk++) {
    const [i, j] = E.IJ_OF_IDX[kk];
    const v = b2[i * 9 + j];
    cellEls[kk].style.fill = v === 0 ? tintFor(own[kk], lg[kk], ls[kk]) : "transparent";
  }
  drawRays(previewG, b2, k, me, `ray preview ${me === 1 ? "gold" : "silver"}`);
  previewG.appendChild(lanternEl(k, me, true));
  const { g, s } = counts(own);
  const cur = counts(E.ownership(st.board));
  const dg = g - cur.g, ds = s - cur.s;
  status(me === 1
    ? `gold ${dg >= 0 ? "+" + dg : dg} · silver ${ds >= 0 ? "+" + ds : ds} — ${coarse ? "tap again to place" : "click to place"}`
    : `silver ${ds >= 0 ? "+" + ds : ds} · gold ${dg >= 0 ? "+" + dg : dg} — ${coarse ? "tap again to place" : "click to place"}`);
}

function onHover(ev) {
  if (st.busy || st.over) return;
  if (E.playerToMove(st.ply) !== st.humanColor) return;
  const k = cellFromEvent(ev);
  if (k < 0 || !isLegal(k)) { if (st.preview >= 0) setPreview(-1); return; }
  if (k !== st.preview) setPreview(k);
}

function isLegal(k) {
  const [i, j] = E.IJ_OF_IDX[k];
  return st.board[i * 9 + j] === 0;
}

function onTap(ev) {
  if (st.busy || st.over) return;
  if (E.playerToMove(st.ply) !== st.humanColor) return;
  const k = cellFromEvent(ev);
  if (k < 0 || !isLegal(k)) { setPreview(-1); defaultStatus(); return; }
  if (coarse && st.preview !== k) { setPreview(k); return; }
  commit(k);
}

function defaultStatus() {
  if (st.over || st.busy) return;
  const mine = E.playerToMove(st.ply) === st.humanColor;
  status(mine ? (st.ply === 0 ? "tap any cell — nearest light claims territory" : "your move") : "…");
}

// ---------- moves ----------
async function commit(k) {
  setPreviewNone();
  const me = E.playerToMove(st.ply);
  E.play(st.board, k, me);
  st.lastMove = k;
  st.ply++;
  ping(me);
  flashMove(k, me);
  render();
  if (st.ply >= E.PLIES) return endGame();
  if (E.playerToMove(st.ply) !== st.humanColor) await aiTurn();
}

function setPreviewNone() { st.preview = -1; previewG.replaceChildren(); }

function flashMove(k, color) {
  rayG.replaceChildren();
  drawRays(rayG, st.board, k, color, `ray flash ${color === 1 ? "gold" : "silver"}`);
  setTimeout(() => rayG.replaceChildren(), 1400);
}

async function aiTurn() {
  st.busy = true;
  const rung = st.ladder[st.rung];
  status(`<span class="think"></span> ${rung.label} is thinking`);
  try {
    const sess = await getSession(rung.file, status);
    const evalFn = makeEvalFn(sess);
    let tick = 0;
    const wrapped = async (f, l) => {
      if ((++tick & 7) === 0) await new Promise(r => setTimeout(r));
      return evalFn(f, l);
    };
    const searcher = new MCTS(wrapped, { sims: SIMS });
    const t0 = performance.now();
    const res = await searcher.run(st.board.slice(), st.ply);
    // gentle variety in the opening so rematches differ; argmax after ply 4
    let a;
    if (st.ply < 4) {
      const w = [...res.visits].map(v => v * v);
      const sum = w.reduce((x, y) => x + y, 0);
      let r = Math.random() * sum;
      a = w.findIndex(v => (r -= v) <= 0);
      if (a < 0) a = bestMove(res.visits);
    } else {
      a = bestMove(res.visits);
    }
    st.insight = { ...res, move: a, color: E.playerToMove(st.ply), ms: performance.now() - t0 };
    const me = E.playerToMove(st.ply);
    E.play(st.board, a, me);
    st.lastMove = a;
    st.ply++;
    ping(me);
    flashMove(a, me);
    render();
    st.busy = false;
    if (st.ply >= E.PLIES) return endGame();
    defaultStatus();
  } catch (err) {
    st.busy = false;
    status(`the lantern-keeper failed to answer (${err.message}). <a href="#" onclick="location.reload()">reload</a>`);
    console.error(err);
  }
}

// ---------- insight ----------
function renderInsight() {
  insightG.replaceChildren();
  $("insight-text").textContent = "";
  if (!st.insightOn || !st.insight) return;
  const { visits, rootQ, move, color, ms } = st.insight;
  let total = 0, mx = 0;
  for (const v of visits) { total += v; if (v > mx) mx = v; }
  if (!total) return;
  // only the top candidates — a spread-out search paints noise otherwise
  const top = [...visits.keys()].filter(k => visits[k] > 0)
    .sort((a, b) => visits[b] - visits[a]).slice(0, 10);
  for (const k of top) {
    const [x, y] = centers[k];
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", x); c.setAttribute("cy", y);
    c.setAttribute("r", 3 + 8 * Math.sqrt(visits[k] / mx));
    c.setAttribute("class", `insight-ring ${color === 1 ? "gold" : "silver"}${k === move ? " chosen" : ""}`);
    insightG.appendChild(c);
  }
  const conf = Math.round(50 * (rootQ + 1));
  $("insight-text").textContent =
    `${st.ladder[st.rung].label} searched ${Math.round(total)} playouts in ${(ms / 1000).toFixed(1)}s — it rates its position ${conf}%. Ring size = attention.`;
}

// ---------- end ----------
function endGame() {
  st.over = true;
  const diff = E.scoreDiff(st.board);
  const goldWins = diff >= E.KOMI_THRESHOLD;
  const { g, s, sh } = counts(E.ownership(st.board));
  $("end-title").textContent = goldWins ? "GOLD PREVAILS" : "SILVER PREVAILS";
  $("end-title").className = goldWins ? "gold-text" : "silver-text";
  const humanWon = (st.humanColor === 1) === goldWins;
  $("end-detail").innerHTML =
    `gold <b>${g}</b> · silver <b>${s}</b> · shadow ${sh}` +
    (g === s ? ` — tie goes to silver<br>` : `<br>`) +
    (humanWon ? `you beat ${st.ladder[st.rung].label}.` : `${st.ladder[st.rung].label} takes it.`) +
    (humanWon && st.rung < st.ladder.length - 1
      ? ` a stronger one is waiting.` : humanWon ? ` that was the strongest one.` : ``);
  $("endcard").classList.add("show");
  chord(goldWins ? 1 : 2);
  status(goldWins ? "gold prevails" : "silver prevails");
}

// ---------- sound (tiny WebAudio synth) ----------
let actx = null;
function audio() {
  if (!st.soundOn) return null;
  if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
  if (actx.state === "suspended") actx.resume();
  return actx;
}
function ping(color) {
  const ac = audio(); if (!ac) return;
  const t = ac.currentTime;
  const o = ac.createOscillator(), g = ac.createGain(), f = ac.createBiquadFilter();
  o.type = color === 1 ? "triangle" : "sine";
  o.frequency.value = color === 1 ? 523.25 : 783.99;
  f.type = "lowpass"; f.frequency.value = 2200;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.14, t + 0.012);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.9);
  o.connect(f).connect(g).connect(ac.destination);
  o.start(t); o.stop(t + 1);
}
function chord(winner) {
  const ac = audio(); if (!ac) return;
  const base = winner === 1 ? [261.6, 329.6, 392.0, 523.3] : [293.7, 370.0, 440.0, 587.3];
  base.forEach((fr, i) => {
    const t = ac.currentTime + i * 0.09;
    const o = ac.createOscillator(), g = ac.createGain();
    o.type = "sine"; o.frequency.value = fr;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.09, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 1.6);
    o.connect(g).connect(ac.destination);
    o.start(t); o.stop(t + 1.7);
  });
}

// ---------- controls ----------
function newGame() {
  st.board = E.newBoard(); st.ply = 0; st.over = false; st.busy = false;
  st.insight = null; st.lastMove = -1;
  $("endcard").classList.remove("show");
  rayG.replaceChildren(); setPreviewNone();
  render();
  defaultStatus();
  if (st.humanColor !== 1) aiTurn();
}

function bindControls() {
  $("btn-new").addEventListener("click", newGame);
  $("btn-rematch").addEventListener("click", () => { $("endcard").classList.remove("show"); newGame(); });
  $("btn-swap").addEventListener("click", () => {
    st.humanColor = 3 - st.humanColor;
    $("btn-color").textContent = st.humanColor === 1 ? "you play gold" : "you play silver";
    $("endcard").classList.remove("show");
    newGame();
  });
  $("btn-color").addEventListener("click", () => {
    st.humanColor = 3 - st.humanColor;
    $("btn-color").textContent = st.humanColor === 1 ? "you play gold" : "you play silver";
    newGame();
  });
  $("btn-insight").addEventListener("click", () => {
    st.insightOn = !st.insightOn;
    $("btn-insight").classList.toggle("on", st.insightOn);
    renderInsight();
  });
  $("btn-sound").addEventListener("click", () => {
    st.soundOn = !st.soundOn;
    localStorage.setItem("umbra-sound", st.soundOn ? "on" : "off");
    $("btn-sound").textContent = st.soundOn ? "sound on" : "sound off";
  });
  $("btn-sound").textContent = st.soundOn ? "sound on" : "sound off";
  $("rung-select").addEventListener("change", (e) => {
    st.rung = +e.target.value;
    $("opp-story").textContent = st.ladder[st.rung].story;
    newGame();
  });
}

// ---------- boot ----------
async function boot() {
  buildBoard();
  render();
  status("waking the lanterns…");
  try {
    st.ladder = await loadLadder();
  } catch {
    status("could not load the opponents (models missing)");
    return;
  }
  const sel = $("rung-select");
  st.ladder.forEach((r, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = r.label;
    sel.appendChild(o);
  });
  st.rung = st.ladder.length - 1;
  sel.value = st.rung;
  $("opp-story").textContent = st.ladder[st.rung].story;
  bindControls();
  defaultStatus();
  // warm the default session in the background
  getSession(st.ladder[st.rung].file, () => {});
}

boot();
