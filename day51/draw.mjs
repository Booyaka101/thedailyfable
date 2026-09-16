// Board geometry for drawing (shared by node renders and the web page).
export function layout(b, size = 400) {
  const pts = [];
  let minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
  for (const c of b.cells) {
    let x, y;
    if (b.type === 'sq') { x = c.x; y = c.y; }
    else { x = c.x + c.y / 2; y = c.y * Math.sqrt(3) / 2; } // axial -> pixel, pointy-top
    pts.push([x, y]); minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y);
  }
  const span = Math.max(maxx - minx, maxy - miny) + 1.3;
  const s = size / span, ox = size / 2 - s * (minx + maxx) / 2, oy = size / 2 - s * (miny + maxy) / 2;
  return { pts: pts.map(([x, y]) => [ox + s * x, oy + s * y]), s, hex: b.type !== 'sq' };
}

export function hexPath(cx, cy, r) {
  const p = [];
  for (let i = 0; i < 6; i++) { const a = Math.PI / 6 + i * Math.PI / 3; p.push((cx + r * Math.cos(a)).toFixed(2) + ',' + (cy + r * Math.sin(a)).toFixed(2)); }
  return 'M' + p.join('L') + 'Z';
}

export function boardSVG(b, board, opts = {}) {
  const size = opts.size || 400, L = layout(b, size), last = opts.last ?? -1;
  const col = opts.colors || { bg: '#f3ead8', cell: '#e5d9bf', line: '#8a7a5a', p1: '#1a1a1a', p2: '#f6f1e6', p2s: '#333' };
  let out = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">`;
  out += `<rect width="${size}" height="${size}" fill="${col.bg}"/>`;
  const r = L.s * 0.5;
  for (let i = 0; i < b.N; i++) {
    const [x, y] = L.pts[i];
    if (L.hex) out += `<path d="${hexPath(x, y, r * 1.13)}" fill="${col.cell}" stroke="${col.line}" stroke-width="${(L.s * 0.05).toFixed(2)}"/>`;
    else out += `<rect x="${(x - r).toFixed(2)}" y="${(y - r).toFixed(2)}" width="${(2 * r).toFixed(2)}" height="${(2 * r).toFixed(2)}" fill="${col.cell}" stroke="${col.line}" stroke-width="${(L.s * 0.05).toFixed(2)}"/>`;
  }
  // side markers for connection games
  if (opts.sides) {
    for (let i = 0; i < b.N; i++) {
      const sd = b.side[i]; if (!sd) continue; const [x, y] = L.pts[i];
      const c1 = (sd & 3) ? col.p1 : null, c2 = (sd & 12) ? col.p2s : null;
      if (c1 && opts.sides !== 'none') out += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(r * 0.12).toFixed(1)}" fill="${c1}" opacity="0.5"/>`;
      if (c2 && opts.sides !== 'none') out += `<circle cx="${(x + r * 0.3).toFixed(1)}" cy="${(y + r * 0.3).toFixed(1)}" r="${(r * 0.12).toFixed(1)}" fill="${c2}" opacity="0.5"/>`;
    }
  }
  for (let i = 0; i < b.N; i++) {
    const v = board[i]; if (!v) continue; const [x, y] = L.pts[i];
    out += `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${(r * 0.8).toFixed(2)}" fill="${v === 1 ? col.p1 : col.p2}" stroke="${v === 1 ? col.p1 : col.p2s}" stroke-width="${(L.s * 0.06).toFixed(2)}"/>`;
    if (i === last) out += `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${(r * 0.95).toFixed(2)}" fill="none" stroke="${opts.colors?.hi || '#c8412b'}" stroke-width="${(L.s * 0.07).toFixed(2)}"/>`;
  }
  if (opts.numbers) for (const [i, n] of Object.entries(opts.numbers)) { const [x, y] = L.pts[+i]; const v = board[+i]; out += `<text x="${x.toFixed(1)}" y="${(y + r * 0.3).toFixed(1)}" font-size="${(r * 0.85).toFixed(1)}" text-anchor="middle" font-family="system-ui" fill="${v === 1 ? col.p2 : col.p1}">${n}</text>`; }
  return out + '</svg>';
}
