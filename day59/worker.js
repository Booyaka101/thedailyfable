// Owns the wasm game. Messages: {op:'init', wasmUrl}, {op:'rules', r:[...8], seed}, {op:'reset'},
// {op:'play', c}, {op:'pass'}, {op:'undo'}, {op:'ai', iters}. Every reply carries a state snapshot.
let W = null;

function snapshot(extra) {
  const n = W.n_cells();
  const ptr = W.cells_ptr();
  const cells = Array.from(new Uint8Array(W.memory.buffer, ptr, n));
  return Object.assign({
    cells, toMove: W.to_move(), stonesLeft: W.stones_left(), outcome: W.outcome(),
    cap1: W.captured(1), cap2: W.captured(2), moves: W.move_count(), lastMove: W.last_move(),
    hasLegal: W.has_legal(), value: W.ai_value() / 1000,
  }, extra || {});
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.op === 'init') {
    const buf = await (await fetch(m.wasmUrl)).arrayBuffer();
    const { instance } = await WebAssembly.instantiate(buf, {});
    W = instance.exports;
    self.postMessage({ op: 'ready' });
    return;
  }
  if (!W) return;
  if (m.op === 'rules') {
    const ok = W.set_rules(...m.r, m.seed >>> 0);
    self.postMessage({ op: 'state', id: m.id, ok, state: ok ? snapshot() : null });
  } else if (m.op === 'reset') {
    W.reset();
    self.postMessage({ op: 'state', id: m.id, state: snapshot() });
  } else if (m.op === 'play') {
    const cap = W.play(m.c);
    self.postMessage({ op: 'state', id: m.id, state: snapshot({ captured: cap === 255 ? -1 : cap }) });
  } else if (m.op === 'pass') {
    W.pass();
    self.postMessage({ op: 'state', id: m.id, state: snapshot() });
  } else if (m.op === 'undo') {
    W.undo();
    self.postMessage({ op: 'state', id: m.id, state: snapshot() });
  } else if (m.op === 'ai') {
    const t0 = performance.now();
    const mv = W.ai_move(m.iters);
    const value = W.ai_value() / 1000;
    let cap = 0;
    if (mv === 65535) { if (!W.has_legal()) W.pass(); } else { cap = W.play(mv); }
    self.postMessage({ op: 'state', id: m.id, state: snapshot({ aiMove: mv, aiValue: value, captured: cap, ms: performance.now() - t0 }) });
  }
};
