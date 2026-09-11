// The writing machine, in the browser. Same equations and integrator as the torch solver:
// double pendulum, absolute angles from the downward vertical, RK4 at the solver's DT.
// The shoulder is driven. The elbow gets at most elbow_max of torque (zero on the free rung).
export class Machine {
  constructor(m) {
    this.m = m;
    this.q = [0, 0]; this.qd = [0, 0];
    this.t = 0;            // machine time [s]
    this.cx = 0;           // carriage x [m]
    this.vx = m.V;         // carriage speed [m/s]
  }
  accel(q, qd, tau, tau2, ac) {
    const { L1, L2, M1, M2, G, C_ELBOW, C_SHOULDER } = this.m;
    const K = this.m.K_ELBOW || 0;
    const d = q[0] - q[1], sd = Math.sin(d), cd = Math.cos(d);
    const m11 = (M1 + M2) * L1 * L1, m12 = M2 * L1 * L2 * cd, m22 = M2 * L2 * L2;
    const c1 = M2 * L1 * L2 * sd * qd[1] * qd[1];
    const c2 = -M2 * L1 * L2 * sd * qd[0] * qd[0];
    const g1 = (M1 + M2) * G * L1 * Math.sin(q[0]);
    const g2 = M2 * G * L2 * Math.sin(q[1]);
    const rel = qd[1] - qd[0];
    const passive = C_ELBOW * rel + K * (q[1] - q[0]);
    const r1 = tau - tau2 + passive - C_SHOULDER * qd[0] - c1 - g1 - (M1 + M2) * ac * L1 * Math.cos(q[0]);
    const r2 = tau2 - passive - c2 - g2 - M2 * ac * L2 * Math.cos(q[1]);
    const det = m11 * m22 - m12 * m12;
    return [(m22 * r1 - m12 * r2) / det, (m11 * r2 - m12 * r1) / det];
  }
  step(tau, tau2 = 0, ac = 0) {
    const dt = this.m.DT, q = this.q, qd = this.qd;
    const k1v = this.accel(q, qd, tau, tau2, ac), k1q = qd;
    const q2 = [q[0] + 0.5 * dt * k1q[0], q[1] + 0.5 * dt * k1q[1]];
    const v2 = [qd[0] + 0.5 * dt * k1v[0], qd[1] + 0.5 * dt * k1v[1]];
    const k2v = this.accel(q2, v2, tau, tau2, ac), k2q = v2;
    const q3 = [q[0] + 0.5 * dt * k2q[0], q[1] + 0.5 * dt * k2q[1]];
    const v3 = [qd[0] + 0.5 * dt * k2v[0], qd[1] + 0.5 * dt * k2v[1]];
    const k3v = this.accel(q3, v3, tau, tau2, ac), k3q = v3;
    const q4 = [q[0] + dt * k3q[0], q[1] + dt * k3q[1]];
    const v4 = [qd[0] + dt * k3v[0], qd[1] + dt * k3v[1]];
    const k4v = this.accel(q4, v4, tau, tau2, ac), k4q = v4;
    for (let i = 0; i < 2; i++) {
      q[i] += dt / 6 * (k1q[i] + 2 * k2q[i] + 2 * k3q[i] + k4q[i]);
      qd[i] += dt / 6 * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]);
    }
    this.cx += (this.vx + 0.5 * ac * dt) * dt; this.vx += ac * dt;
    this.t += dt;
  }
  elbow() { const { L1 } = this.m; return [this.cx + L1 * Math.sin(this.q[0]), -L1 * Math.cos(this.q[0])]; }
  tip() {
    const { L1, L2 } = this.m;
    return [this.cx + L1 * Math.sin(this.q[0]) + L2 * Math.sin(this.q[1]),
            -L1 * Math.cos(this.q[0]) - L2 * Math.cos(this.q[1])];
  }
}

// Plays glyph torque programs one after another. Each program was solved from rest, so a
// small shoulder-only feedback term keeps the real state near the plan when residue from the
// previous letter would otherwise compound. The elbow gets nothing but physics.
export class Writer {
  constructor(machine, glyphs, opts = {}) {
    this.mc = machine; this.glyphs = glyphs;
    this.queue = []; this.cur = null; this.k = 0;
    this.kp = opts.kp ?? 60; this.kd = opts.kd ?? 6;
    this.onInk = opts.onInk || (() => {});
    this.spaceSteps = Math.round(0.35 / machine.m.V / machine.m.DT);
    this.lastTau = 0; this.lastTau2 = 0; this.penDown = false;
  }
  type(text) { for (const ch of text) this.queue.push(ch); }
  clear() { this.queue.length = 0; }
  busy() { return this.cur !== null || this.queue.length > 0; }
  // advance the machine by one DT
  step() {
    if (!this.cur) {
      if (!this.queue.length) { this.free(); return; }
      const ch = this.queue.shift();
      const g = this.glyphs[ch];
      if (!g) { this.cur = { space: true, steps: this.spaceSteps }; this.k = 0; }
      else { this.cur = g; this.k = 0; this.x0 = this.mc.cx; }
    }
    const g = this.cur;
    if (g.space) { this.free(); this.k++; if (this.k >= g.steps) this.cur = null; return; }
    const k = this.k;
    let tau = g.tau[k];
    const qp = g.q[k], qn = g.q[k + 1];
    const qdp = (qn[0] - qp[0]) / this.mc.m.DT;
    tau += this.kp * (qp[0] - this.mc.q[0]) + this.kd * (qdp - this.mc.qd[0]);
    const lim = this.mc.m.TAU_MAX;
    tau = Math.max(-lim, Math.min(lim, tau));
    this.lastTau = tau;
    let tau2 = 0;
    const lim2 = this.mc.m.elbow_max || 0;
    if (lim2 > 0) {
      tau2 = g.tau2[k] + this.kp * (qp[1] - this.mc.q[1]) + this.kd * ((qn[1] - qp[1]) / this.mc.m.DT - this.mc.qd[1]);
      tau2 = Math.max(-lim2, Math.min(lim2, tau2));
    }
    this.lastTau2 = tau2;
    const ac = g.ac ? g.ac[k] : 0;
    const p0 = this.mc.tip();
    this.mc.step(tau, tau2, ac);
    const p1 = this.mc.tip();
    const down = g.ink[k] > 0.5;
    if (down) this.onInk(p0, p1, this.mc.qd);
    this.penDown = down;
    this.k++;
    if (this.k >= g.steps) this.cur = null;
  }
  free() {
    this.lastTau = 0; this.lastTau2 = 0; this.penDown = false;
    // between glyphs the carriage servo pulls speed back to nominal
    const ac = Math.max(-2, Math.min(2, 8 * (this.mc.m.V - this.mc.vx)));
    this.mc.step(0, 0, ac);
  }
}
