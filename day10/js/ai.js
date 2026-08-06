// ONNX-backed evaluation + the opponent ladder.
// ladder.json (written at export time) lists the shipped checkpoints:
//   [{key, file, label, story, games}] ordered newborn -> final.

import * as E from "./engine.js";

export async function loadLadder() {
  const res = await fetch("models/ladder.json");
  return res.json();
}

const sessions = new Map();

export async function getSession(file, onStatus) {
  if (sessions.has(file)) return sessions.get(file);
  onStatus?.("summoning…");
  ort.env.wasm.numThreads = 1;             // GitHub Pages: no cross-origin isolation
  ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
  const sess = await ort.InferenceSession.create(file, {
    executionProviders: ["wasm"],
  });
  sessions.set(file, sess);
  return sess;
}

export function makeEvalFn(sess) {
  return async (feats, legal) => {
    const input = new ort.Tensor("float32", feats, [1, E.N_PLANES, 9, 9]);
    const out = await sess.run({ planes: input });
    const logits = out.policy.data;        // Float32Array(61)
    const value = out.value.data[0];
    let mx = -Infinity;
    for (let a = 0; a < E.N_CELLS; a++) if (legal[a] && logits[a] > mx) mx = logits[a];
    const policy = new Float32Array(E.N_CELLS);
    let sum = 0;
    for (let a = 0; a < E.N_CELLS; a++) {
      if (legal[a]) { policy[a] = Math.exp(logits[a] - mx); sum += policy[a]; }
    }
    for (let a = 0; a < E.N_CELLS; a++) policy[a] /= sum;
    return { policy, value };
  };
}
