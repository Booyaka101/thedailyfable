"""The controller: a policy network driving the arm's muscles to follow a pen-tip target.

At every control tick (20 ms) the policy sees the arm's joint state as it was DELAY ticks ago (a
proprioceptive delay), the commands it sent since, the current pen pressure, and the next H target points
relative to the estimated tip position, in millimetres. It outputs eight muscle excitations and a pressure
command, held for the tick. It is trained by backpropagation through the physics in arm.py.

Underneath the network sits a fixed reflex. A coarse internal copy of the arm is run forward from the
delayed state with the commands sent since (a forward model, the cerebellum's job), and a Cartesian
spring-damper toward the target a few ticks ahead is evaluated on that estimate, plus exact cancellation of
the arm's passive elasticity. The torque is split into agonist and antagonist activations with a little
co-contraction. The network adds a bounded force at the pen tip (a few newtons) and a pressure correction on
top of the reflex, so an untrained network writes with the reflex alone (late, rounding the corners) and
training sharpens it. A bounded tip force cannot fling the arm, which keeps training from diverging.
"""
import torch
import torch.nn as nn
import arm as A

TICK = 5              # physics steps per control tick (5 x 4 ms)
H = 10                # target horizon, ticks
DELAY = 2             # ticks of sensory delay
Q_MEAN = torch.tensor([0.5, 1.6, 0.25])
IN_DIM = 3 + 3 + A.NM + 1 + 2 * H + H + 2
KP_X, KD_X = 300.0, 25.0          # Cartesian reflex gains, N/m and N s/m
KI_X, I_LEAK = 0.0, 0.3           # leaky integral on tip error, N/(m s), and its time constant
KP_W, KD_W = 1.5, 0.05            # wrist joint reflex toward Q_REF[2], N m/rad
TONE = 0.01                       # co-contraction floor
MODEL_SUB = 2                     # internal model steps per tick
LOOK = 1                          # reflex aims at the target this many ticks ahead
LEAD = 1                          # extra ticks of lead on the feedforward, for the activation lag
FRES = 3.0                        # bound on the network's tip force residual, N
PRES = 0.3                        # bound on the network's pressure residual
AGON = [0, 2, 6]                  # muscle that produces +torque at joint j, its antagonist is +1


class Policy(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IN_DIM, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 3))
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)      # start as the pure reflex

    def forward(self, x):
        """Tip force residual (B, 2) in N and pressure residual (B,)."""
        y = torch.tanh(self.net(x))
        return FRES * y[:, :2], PRES * y[:, 2]


class Sensor:
    """Delayed proprioception plus efference copy: what the controller knows at each tick."""

    def __init__(self, arm):
        self.arm = arm
        self.hist = [(arm.q, arm.qd, arm.a)] * (DELAY + 1)
        self.us = [torch.full((arm.B, A.NM), TONE, device=arm.dev)] * max(DELAY, 1)
        self.model = A.Arm(arm.B, dt=TICK * arm.dt / MODEL_SUB, noise=False, fatigue=False, device=arm.dev)
        self.ierr = torch.zeros(arm.B, 2, device=arm.dev)

    def estimate(self):
        """Current joint state estimated by rolling the internal model over the delay with the sent commands."""
        q, qd, a = self.hist[-DELAY - 1]
        m = self.model
        m.q, m.qd, m.a = q, qd, a
        m.fat = torch.zeros_like(a)
        for u in (self.us[-DELAY:] if DELAY else []):
            for _ in range(MODEL_SUB):
                m.step(u, m.p)
        return m.q, m.qd

    def push(self, u):
        self.hist.append((self.arm.q, self.arm.qd, self.arm.a)); self.hist = self.hist[-DELAY - 1:]
        self.us.append(u); self.us = self.us[-max(DELAY, 1):]

    def detach(self):
        self.hist = [tuple(t.detach() for t in h) for h in self.hist]
        self.us = [u.detach() for u in self.us]
        self.ierr = self.ierr.detach()


def reflex(arm, q, qd, targets, tp, sensor=None):
    """Joint torque (B, 3), tip force (B, 2), Jacobian, muscle gains and pressure command from the estimated
    state and the target window.

    Computed torque: the target's acceleration LOOK ticks ahead (finite differences on the window) is mapped
    through the arm's mass matrix, plus Coriolis, damping and passive-elasticity cancellation, plus a
    Cartesian spring-damper on the remaining error. The split into agonist and antagonist accounts for each
    muscle's force-length and force-velocity gain at the estimated state.
    """
    Hn = targets.shape[1]
    k = min(LOOK, Hn - 2)
    dt = TICK * arm.dt
    x_des = targets[:, k]
    v_des = (targets[:, k + 1] - targets[:, k - 1]) / (2 * dt) if k > 0 else (targets[:, 1] - targets[:, 0]) / dt
    ka = min(k + LEAD, Hn - 2)
    a_des = (targets[:, ka + 1] - 2 * targets[:, ka] + targets[:, ka - 1]) / (dt * dt) if ka > 0 else torch.zeros_like(x_des)
    tip = arm.tip(q)
    J = arm.tip_jac(q)
    v = torch.einsum("bij,bj->bi", J, qd)
    f = KP_X * (x_des - tip) + KD_X * (v_des - v)
    if sensor is not None and KI_X > 0:
        k_leak = 1 - dt / I_LEAK
        sensor.ierr = sensor.ierr * k_leak + (x_des - tip) * dt
        f = f + KI_X * sensor.ierr
    # desired joint acceleration: least squares on the 2x3 Jacobian, wrist pulled toward its rest angle
    JT = J.transpose(1, 2)
    qdd_w = (KP_W * (arm.qref[2] - q[:, 2]) - KD_W * qd[:, 2]) / 0.01
    qdd_des = torch.linalg.solve(JT @ J + 1e-4 * torch.eye(3, device=q.device),
                                 (JT @ a_des.unsqueeze(-1)).squeeze(-1) + 1e-4 * torch.stack(
                                     [torch.zeros_like(qdd_w), torch.zeros_like(qdd_w), qdd_w], 1))
    tau = (arm.mass_matrix(q) @ qdd_des.unsqueeze(-1)).squeeze(-1) + arm.coriolis(q, qd) + arm.b * qd
    tau = tau + torch.einsum("bij,bi->bj", J, f) + arm.kp * (q - arm.qref)
    tau = tau + torch.stack([torch.zeros_like(q[:, 0]), torch.zeros_like(q[:, 0]),
                             KP_W * (arm.qref[2] - q[:, 2]) - KD_W * qd[:, 2]], 1)
    fl = (1 + A.FL_SLOPE * (-((q - arm.qref) @ arm.R)) / A.L0).clamp(0.3, 1.7)      # (B, 8)
    vm = -(qd @ arm.R)
    fv = torch.where(vm >= 0, (1 - vm / A.VMAX).clamp(min=0.05) / (1 + vm / (0.25 * A.VMAX)),
                     1.5 - 0.5 / (1 - 7.5 * vm / A.VMAX))
    gain = arm.Fmax * fl * fv                                                      # N per unit activation
    return tau, f, J, gain, tp[:, k]


def split(arm, tau, gain):
    """Activations (B, 8) producing joint torque tau through agonist/antagonist pairs over a tone floor."""
    a = torch.full((tau.shape[0], A.NM), TONE, device=tau.device)
    tau = tau - (a * gain) @ arm.R.T          # tone is unbalanced by force-length; fold it into the split
    cols = []
    for j, m in enumerate(AGON):
        cols.append((m, tau[:, j] / (arm.R[j, m] * gain[:, m])))
        cols.append((m + 1, -tau[:, j] / (arm.R[j, m] * gain[:, m + 1])))
    a = a.clone()
    for m, val in cols:
        a[:, m] = a[:, m] + val.clamp(min=0)
    return a.clamp(0.0, 1.0)


def features(q, qd, a, p, tip, targets, tp, f):
    rel = (targets - tip.unsqueeze(1)) * 1000.0
    return torch.cat([(q - Q_MEAN.to(q.device)) * 2.0, qd * 0.3, a, p.unsqueeze(1), rel.flatten(1), tp, f * 0.2], 1)


def act(policy, arm, sensor, tg):
    """One control decision from the sensor's estimate and the target window tg (B, H, 3)."""
    q, qd = sensor.estimate()
    tau, f, J, gain, pbase = reflex(arm, q, qd, tg[:, :, :2], tg[:, :, 2], sensor)
    x = features(q, qd, arm.a, arm.p, arm.tip(q), tg[:, :, :2], tg[:, :, 2], f)
    df, dp = policy(x)
    tau = tau + torch.einsum("bij,bi->bj", J, df)
    return split(arm, tau, gain), (pbase + dp).clamp(0.0, 1.0)


def rollout(policy, arm, target, record=False, trunc=10, sensor=None):
    """Drive `arm` (batch B) along target (B, T, 3) = x, y, pressure at the physics rate (250 Hz).

    Returns loss terms and, if record, per-step traces. With sensor=None the pen starts on the first target
    point at rest; pass the Sensor from a previous call to continue a run.
    """
    B, T, _ = target.shape
    d = target.device
    if sensor is None:
        q0 = A.ik(arm, target[:, 0, :2])
        arm.reset(q0)
        sensor = Sensor(arm)
    err2, effort, perr = 0.0, 0.0, 0.0
    rec = {k: [] for k in ("tip", "p", "F", "joints", "a", "fat", "u")}
    nt = T // TICK
    idx = torch.arange(H, device=d) * TICK
    for k in range(nt):
        t = k * TICK
        ti = (t + idx).clamp(max=T - 1)
        u, pc = act(policy, arm, sensor, target[:, ti])
        for s in range(TICK):
            F = arm.step(u, pc)
            tip = arm.tip()
            e = tip - target[:, t + s, :2]
            err2 = err2 + (e * e).sum(1)
            tpr = target[:, t + s, 2]
            # ink where the pen should be up, or none where it should be firmly down; ramps are free
            perr = perr + ((arm.p * (tpr < 0.05)) ** 2 + ((0.5 - arm.p).clamp(min=0) * (tpr > 0.5)) ** 2).mean()
            if record:
                rec["tip"].append(tip); rec["p"].append(arm.p); rec["F"].append(F); rec["joints"].append(arm.joints())
                rec["a"].append(arm.a); rec["fat"].append(arm.fat); rec["u"].append(u)
        effort = effort + (u * u).sum(1).mean()
        if trunc and (k + 1) % trunc == 0:
            arm.detach(); sensor.detach()                    # truncated backprop through time
        sensor.push(u)
    n = nt * TICK
    out = dict(err2=err2.mean() / n, err2_b=err2 / n, effort=effort / nt, perr=perr / n, sensor=sensor)
    if record:
        out.update({k: torch.stack(v, 1) for k, v in rec.items()})
    return out
