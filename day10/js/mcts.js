// PUCT MCTS over UMBRA — value conventions identical to ../../mcts.py:
// W[node] accumulates values from the perspective of the mover AT that node;
// a parent viewing a child negates. evalFn is async (ONNX session behind it).

import * as E from "./engine.js";

export class MCTS {
  // evalFn(features Float32Array(7*81), legal Uint8Array(61))
  //   -> Promise<{policy: Float32Array(61) legal-masked+normalized, value: number}>
  constructor(evalFn, opts = {}) {
    this.evalFn = evalFn;
    this.sims = opts.sims ?? 128;
    this.cPuct = opts.cPuct ?? 1.5;
    // First-play urgency. Scoring an unvisited move as Q=0 only works while values
    // are near 0; this net is certain enough (|v| ~ 1) that a single visited child
    // outscored every exploration bonus and all 128 playouts piled onto one move —
    // a 1-ply policy net wearing a search. Unvisited moves inherit the parent's own
    // value, minus a penalty scaled by how much prior has already been tried.
    this.fpu = opts.fpu ?? 0.25;
  }

  async run(board, ply, onProgress) {
    const M = this.sims + 2;
    const boards = new Array(M);
    const plies = new Int16Array(M);
    const N = new Float32Array(M);
    const W = new Float32Array(M);
    const P = new Array(M);
    const child = new Array(M);
    const legal = new Array(M);
    const isTerm = new Uint8Array(M);
    const termV = new Float32Array(M);
    let nNodes = 1;

    const legalArr = (b) => {
      const l = new Uint8Array(E.N_CELLS);
      for (const k of E.legalMoves(b)) l[k] = 1;
      return l;
    };

    boards[0] = board.slice();
    plies[0] = ply;
    legal[0] = legalArr(board);
    child[0] = new Int32Array(E.N_CELLS).fill(-1);
    {
      const { policy, value } = await this.evalFn(E.features(board, ply), legal[0]);
      P[0] = policy;
      N[0] = 1; W[0] = value;
    }

    const path = new Int32Array(E.PLIES + 2);

    for (let sim = 0; sim < this.sims; sim++) {
      let cur = 0, depth = 0;
      path[depth++] = 0;
      let leafVal = 0;

      for (;;) {
        if (isTerm[cur]) { leafVal = termV[cur]; break; }
        // select
        const ch = child[cur], p = P[cur], lg = legal[cur];
        const sqrtN = Math.sqrt(Math.max(N[cur], 1));
        let triedP = 0;
        for (let a = 0; a < E.N_CELLS; a++) if (lg[a] && ch[a] >= 0) triedP += p[a];
        const qUnseen = W[cur] / Math.max(N[cur], 1) - this.fpu * Math.sqrt(triedP);
        let best = -Infinity, bestA = -1;
        for (let a = 0; a < E.N_CELLS; a++) {
          if (!lg[a]) continue;
          const c = ch[a];
          let q = qUnseen, n = 0;
          if (c >= 0) { n = N[c]; q = -W[c] / Math.max(n, 1e-9); }
          const s = q + this.cPuct * p[a] * sqrtN / (1 + n);
          if (s > best) { best = s; bestA = a; }
        }
        const c = ch[bestA];
        if (c >= 0) { cur = c; path[depth++] = c; continue; }
        // expand
        const nb = boards[cur].slice();
        E.play(nb, bestA, E.playerToMove(plies[cur]));
        const npl = plies[cur] + 1;
        const id = nNodes++;
        boards[id] = nb; plies[id] = npl;
        child[id] = new Int32Array(E.N_CELLS).fill(-1);
        ch[bestA] = id;
        path[depth++] = id;
        if (npl >= E.PLIES) {
          isTerm[id] = 1;
          termV[id] = E.resultForGold(nb); // mover parity at ply 20 = gold
          leafVal = termV[id];
        } else {
          legal[id] = legalArr(nb);
          const { policy, value } = await this.evalFn(E.features(nb, npl), legal[id]);
          P[id] = policy;
          leafVal = value;
        }
        break;
      }

      // backup: leaf value is from leaf-mover's perspective; alternate signs up
      for (let d = depth - 1, sign = 1; d >= 0; d--, sign = -sign) {
        N[path[d]] += 1;
        W[path[d]] += sign * leafVal;
      }
      if (onProgress && (sim & 15) === 0) onProgress(sim / this.sims);
    }

    const visits = new Float32Array(E.N_CELLS);
    const ch = child[0];
    for (let a = 0; a < E.N_CELLS; a++)
      if (legal[0][a] && ch[a] >= 0) visits[a] = N[ch[a]];
    const rootQ = W[0] / Math.max(N[0], 1);
    // childQ for insight display: mover's win estimate after each move
    const childQ = new Float32Array(E.N_CELLS).fill(NaN);
    for (let a = 0; a < E.N_CELLS; a++)
      if (ch[a] >= 0 && N[ch[a]] > 0) childQ[a] = -W[ch[a]] / N[ch[a]];
    return { visits, rootQ, childQ, prior: P[0] };
  }
}

export function bestMove(visits) {
  let best = -1, ba = -1;
  for (let a = 0; a < visits.length; a++)
    if (visits[a] > best) { best = visits[a]; ba = a; }
  return ba;
}
