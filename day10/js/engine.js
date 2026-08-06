// UMBRA rules — exact JS mirror of ../../engine.py (see its docstring).
// Board: Int8Array(81), 9x9 row-major [i*9+j], i=q+4, j=r+4. 0 empty 1 gold 2 silver.

export const R = 4, SIZE = 9, N_CELLS = 61, PLIES = 20, STONES_EACH = 10;
export const N_PLANES = 7;
export const KOMI_THRESHOLD = 1; // gold wins iff scoreDiff >= 1 (ties go to silver)
export const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];

export const VALID = new Uint8Array(81);
export const IDX_OF_IJ = new Int16Array(81).fill(-1);  // [i*9+j] -> cell idx
export const IJ_OF_IDX = [];           // idx -> [i, j], same row-major order as python
for (let i = 0; i < 9; i++) {
  for (let j = 0; j < 9; j++) {
    if (Math.abs((i - 4) + (j - 4)) <= R) {
      VALID[i * 9 + j] = 1;
      IDX_OF_IJ[i * 9 + j] = IJ_OF_IDX.length;
      IJ_OF_IDX.push([i, j]);
    }
  }
}

// RAYS[cell][dir] = array of cell indices walking outward from cell in dir
export const RAYS = [];
for (let k = 0; k < N_CELLS; k++) {
  const [i0, j0] = IJ_OF_IDX[k];
  const rays = [];
  for (const [dq, dr] of DIRS) {
    const ray = [];
    let i = i0 + dq, j = j0 + dr;
    while (i >= 0 && i < 9 && j >= 0 && j < 9 && VALID[i * 9 + j]) {
      ray.push(IDX_OF_IJ[i * 9 + j]);
      i += dq; j += dr;
    }
    rays.push(ray);
  }
  RAYS.push(rays);
}

export function newBoard() { return new Int8Array(81); }
export function playerToMove(ply) { return 1 + (ply % 2); }

export function legalMoves(board) {
  const out = [];
  for (let k = 0; k < N_CELLS; k++) {
    const [i, j] = IJ_OF_IDX[k];
    if (board[i * 9 + j] === 0) out.push(k);
  }
  return out;
}

export function play(board, cellIdx, player) {
  const [i, j] = IJ_OF_IDX[cellIdx];
  board[i * 9 + j] = player;
}

// lightCounts: per cell-idx, [goldUnits, silverUnits] from nearest lantern per direction.
// Occupied cells receive no light (they score for nobody).
export function lightCounts(board) {
  const lg = new Uint8Array(N_CELLS), ls = new Uint8Array(N_CELLS);
  for (let k = 0; k < N_CELLS; k++) {
    const [i, j] = IJ_OF_IDX[k];
    if (board[i * 9 + j] !== 0) continue;
    for (let d = 0; d < 6; d++) {
      const ray = RAYS[k][d];
      for (let s = 0; s < ray.length; s++) {
        const [ri, rj] = IJ_OF_IDX[ray[s]];
        const v = board[ri * 9 + rj];
        if (v !== 0) { if (v === 1) lg[k]++; else ls[k]++; break; }
      }
    }
  }
  return [lg, ls];
}

// ownership per cell-idx: 1 gold-lit, 2 silver-lit, 0 shadow or occupied
export function ownership(board, lc) {
  const [lg, ls] = lc || lightCounts(board);
  const own = new Uint8Array(N_CELLS);
  for (let k = 0; k < N_CELLS; k++) {
    const [i, j] = IJ_OF_IDX[k];
    if (board[i * 9 + j] !== 0) continue;
    if (lg[k] > ls[k]) own[k] = 1;
    else if (ls[k] > lg[k]) own[k] = 2;
  }
  return own;
}

export function scoreDiff(board) {
  const own = ownership(board);
  let d = 0;
  for (let k = 0; k < N_CELLS; k++) d += (own[k] === 1) - (own[k] === 2);
  return d;
}

export function resultForGold(board) {
  return scoreDiff(board) >= KOMI_THRESHOLD ? 1.0 : -1.0;
}

// features: Float32Array(7*81), canonical to side-to-move — mirrors engine.features()
export function features(board, ply) {
  const me = playerToMove(ply), opp = 3 - me;
  const [lg, ls] = lightCounts(board);
  const own = ownership(board, [lg, ls]);
  const f = new Float32Array(N_PLANES * 81);
  const lMine = me === 1 ? lg : ls, lOpp = me === 1 ? ls : lg;
  for (let k = 0; k < N_CELLS; k++) {
    const [i, j] = IJ_OF_IDX[k];
    const p = i * 9 + j, v = board[p];
    if (v === me) f[0 * 81 + p] = 1;
    if (v === opp) f[1 * 81 + p] = 1;
    f[2 * 81 + p] = lMine[k] / 6;
    f[3 * 81 + p] = lOpp[k] / 6;
    if (own[k] === me) f[4 * 81 + p] = 1;
    if (own[k] === opp) f[5 * 81 + p] = 1;
    f[6 * 81 + p] = ply / PLIES;
  }
  return f;
}
