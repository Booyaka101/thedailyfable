"""A planar three-link arm (shoulder, elbow, wrist) seen from above, driven by eight muscles.

Everything is batched over B independent arms and differentiable, so a controller can be trained by
backpropagation through the physics. Units are SI. The shoulder is at the origin, +y points away from
the body across the desk, +x to the writer's right.

Muscle model: first-order activation dynamics (Winters 1995 time constants), Hill force-velocity,
constant moment arms, signal-dependent noise on the excitation (Harris & Wolpert 1998: SD grows with
the command) and a slow fatigue state. The force-length curve is its ascending limb only (force rises with
stretch, so co-contraction stiffens the arm); moment arms are constant.
"""
import math
import torch

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = torch.tensor([0.30, 0.26, 0.10])            # upper arm, forearm, hand (wrist to pen tip)
MASS = torch.tensor([2.0, 1.2, 0.5])
INERTIA = MASS * L ** 2 / 12                    # about the centre of mass, as rods
DAMP = torch.tensor([0.35, 0.25, 0.06])         # joint viscous damping, N m s
QLIM = torch.tensor([[-0.2, 2.2], [0.25, 2.7], [-1.1, 1.1]])
KLIM = 40.0                                     # soft joint-limit spring, N m / rad

# moment arms (m): rows = joints, columns = muscles
#   0 shoulder flexor, 1 shoulder extensor, 2 elbow flexor, 3 elbow extensor,
#   4 biarticular flexor, 5 biarticular extensor, 6 wrist flexor, 7 wrist extensor
R = torch.tensor([
    [0.030, -0.030, 0.0, 0.0, 0.020, -0.020, 0.0, 0.0],
    [0.0, 0.0, 0.025, -0.025, 0.020, -0.020, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.012, -0.012],
])
FMAX = torch.tensor([800.0, 800.0, 600.0, 600.0, 500.0, 500.0, 220.0, 220.0])
VMAX = 1.0                                       # m/s shortening at which active force is zero
TAU_ACT, TAU_DEACT = 0.030, 0.050
NOISE_CV = 0.05                                  # signal-dependent noise, coefficient of variation
NOISE_TAU = 0.015                                # correlation time of the noise process
K_FAT, K_REC = 0.006, 0.0012                     # fatigue build (per unit activation per s) and recovery
L0 = 0.10                                        # muscle optimal length, m: sets the force-length slope
FL_SLOPE = 2.0                                   # ascending limb: force rises 2x per optimal length of stretch
Q_REF = torch.tensor([0.5, 1.6, 0.25])           # posture where every muscle sits at optimal length
K_PASSIVE = torch.tensor([3.0, 3.0, 0.6])        # passive joint elasticity about Q_REF, N m / rad
TAU_PEN = 0.015                                  # pen pressure follows the fingers, quick
PEN_NOISE = 0.06

NM = 8


def _e(th):
    return torch.stack([torch.cos(th), torch.sin(th)], -1)


def _ep(th):
    return torch.stack([-torch.sin(th), torch.cos(th)], -1)


class Arm:
    def __init__(self, B, dt=0.004, noise=True, fatigue=True, device=dev, gen=None):
        self.B, self.dt, self.noise_on, self.fatigue_on, self.dev = B, dt, noise, fatigue, device
        self.L, self.M, self.I, self.b = (t.to(device) for t in (L, MASS, INERTIA, DAMP))
        self.R, self.Fmax, self.qlim = R.to(device), FMAX.to(device), QLIM.to(device)
        self.qref, self.kp = Q_REF.to(device), K_PASSIVE.to(device)
        self.gen = gen
        self.reset(torch.zeros(B, 3, device=device))

    def reset(self, q, p=None):
        B, d = self.B, self.dev
        self.q = q.clone()
        self.qd = torch.zeros(B, 3, device=d)
        self.a = torch.zeros(B, NM, device=d)
        self.fat = torch.zeros(B, NM, device=d)
        self.eta = torch.zeros(B, NM, device=d)
        self.eta_p = torch.zeros(B, device=d)
        self.p = torch.zeros(B, device=d) if p is None else p.clone()

    # kinematics -------------------------------------------------------------------------------
    def angles(self, q):
        th1 = q[:, 0]
        th2 = th1 + q[:, 1]
        th3 = th2 + q[:, 2]
        return th1, th2, th3

    def joints(self, q=None):
        """Shoulder, elbow, wrist, pen tip positions: (B, 4, 2)."""
        q = self.q if q is None else q
        th1, th2, th3 = self.angles(q)
        s = torch.zeros(q.shape[0], 2, device=q.device)
        e = s + self.L[0] * _e(th1)
        w = e + self.L[1] * _e(th2)
        t = w + self.L[2] * _e(th3)
        return torch.stack([s, e, w, t], 1)

    def tip(self, q=None):
        return self.joints(q)[:, 3]

    def tip_jac(self, q):
        th1, th2, th3 = self.angles(q)
        c1, c2, c3 = self.L[0] * _ep(th1), self.L[1] * _ep(th2), self.L[2] * _ep(th3)
        return torch.stack([c1 + c2 + c3, c2 + c3, c3], -1)   # (B, 2, 3)

    def _mass_parts(self):
        if not hasattr(self, "_A"):
            L1, L2, L3 = self.L.tolist()
            m1, m2, m3 = self.M.tolist()
            I1, I2, I3 = self.I.tolist()
            d = self.dev
            A0 = torch.tensor([
                [I1 + I2 + I3 + m1 * L1 ** 2 / 4 + m2 * (L1 ** 2 + L2 ** 2 / 4) + m3 * (L1 ** 2 + L2 ** 2 + L3 ** 2 / 4),
                 I2 + I3 + m2 * L2 ** 2 / 4 + m3 * (L2 ** 2 + L3 ** 2 / 4), I3 + m3 * L3 ** 2 / 4],
                [0, I2 + I3 + m2 * L2 ** 2 / 4 + m3 * (L2 ** 2 + L3 ** 2 / 4), I3 + m3 * L3 ** 2 / 4],
                [0, 0, I3 + m3 * L3 ** 2 / 4]], device=d)
            A2 = torch.tensor([[m2 * L1 * L2 + 2 * m3 * L1 * L2, m2 * L1 * L2 / 2 + m3 * L1 * L2, 0],
                               [0, 0, 0], [0, 0, 0]], device=d)
            A3 = torch.tensor([[m3 * L2 * L3, m3 * L2 * L3, m3 * L2 * L3 / 2],
                               [0, m3 * L2 * L3, m3 * L2 * L3 / 2], [0, 0, 0]], device=d)
            A23 = torch.tensor([[m3 * L1 * L3, m3 * L1 * L3 / 2, m3 * L1 * L3 / 2],
                                [0, 0, 0], [0, 0, 0]], device=d)
            sym = lambda A: A + A.T - torch.diag(torch.diag(A))
            self._A = tuple(sym(A) for A in (A0, A2, A3, A23))
        return self._A

    def mass_matrix(self, q):
        A0, A2, A3, A23 = self._mass_parts()
        c2, c3, c23 = torch.cos(q[:, 1]), torch.cos(q[:, 2]), torch.cos(q[:, 1] + q[:, 2])
        return A0 + A2 * c2[:, None, None] + A3 * c3[:, None, None] + A23 * c23[:, None, None]

    def coriolis(self, q, qd):
        """C(q, qd) qd from the Christoffel symbols of the closed-form mass matrix."""
        A0, A2, A3, A23 = self._mass_parts()
        s2, s3, s23 = torch.sin(q[:, 1]), torch.sin(q[:, 2]), torch.sin(q[:, 1] + q[:, 2])
        dM2 = -(A2 * s2[:, None, None] + A23 * s23[:, None, None])       # dM/dq2
        dM3 = -(A3 * s3[:, None, None] + A23 * s23[:, None, None])       # dM/dq3
        dM = torch.stack([torch.zeros_like(dM2), dM2, dM3], -1)           # (B, i, j, k) = dM_ij/dq_k
        t1 = torch.einsum("bijk,bj,bk->bi", dM, qd, qd)                  # dM_ij/dq_k qd_j qd_k
        t3 = torch.einsum("bjki,bj,bk->bi", dM, qd, qd)                  # dM_jk/dq_i qd_j qd_k
        return t1 - 0.5 * t3

    # muscles ----------------------------------------------------------------------------------
    def muscle_torque(self, a, q, qd, fat):
        v = -(qd @ self.R)                                   # shortening velocity of each muscle, m/s
        fv = torch.where(v >= 0, (1 - v / VMAX).clamp(min=0) / (1 + v / (0.25 * VMAX)),
                         1.5 - 0.5 / (1 - 7.5 * v / VMAX))   # lengthening branch tends to 1.5
        stretch = -((q - self.qref) @ self.R)                # length change from optimal, m
        fl = (1 + FL_SLOPE * stretch / L0).clamp(0.3, 1.7)
        F = self.Fmax * a * fl * fv * (1 - fat)
        return F @ self.R.T, F

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.dev, generator=self.gen)

    def step(self, u, p_cmd):
        """Advance one physics step. u: (B, 8) excitations in [0,1]; p_cmd: (B,) pen pressure command."""
        dt, B = self.dt, self.B
        if self.noise_on:
            k = math.exp(-dt / NOISE_TAU)
            self.eta = k * self.eta + math.sqrt(1 - k * k) * self._randn(B, NM)
            self.eta_p = k * self.eta_p + math.sqrt(1 - k * k) * self._randn(B)
            u = (u * (1 + NOISE_CV * self.eta)).clamp(0, 1)
            p_cmd = (p_cmd + PEN_NOISE * p_cmd * self.eta_p).clamp(0, 1)
        tau_a = torch.where(u > self.a, torch.full_like(u, TAU_ACT), torch.full_like(u, TAU_DEACT))
        self.a = self.a + dt * (u - self.a) / tau_a
        self.p = self.p + dt * (p_cmd - self.p) / TAU_PEN
        if self.fatigue_on:
            self.fat = self.fat + dt * (K_FAT * self.a * (1 - self.fat) - K_REC * self.fat)
        tau_m, F = self.muscle_torque(self.a, self.q, self.qd, self.fat)
        lo, hi = self.qlim[:, 0], self.qlim[:, 1]
        tau_lim = -KLIM * ((self.q - hi).clamp(min=0) + (self.q - lo).clamp(max=0))
        tau = tau_m + tau_lim - self.b * self.qd - self.kp * (self.q - self.qref)
        Mq = self.mass_matrix(self.q)
        rhs = tau - self.coriolis(self.q, self.qd)
        qdd = torch.linalg.solve(Mq, rhs.unsqueeze(-1)).squeeze(-1)
        self.qd = self.qd + dt * qdd
        self.q = self.q + dt * self.qd
        return F

    def detach(self):
        for n in ("q", "qd", "a", "fat", "eta", "eta_p", "p"):
            setattr(self, n, getattr(self, n).detach())


def ik(arm, target, wrist=0.25, iters=60):
    """Joint angles putting the pen tip at target (B,2), holding the wrist near a fixed angle."""
    q = torch.tensor([1.3, 1.4, wrist], device=target.device).repeat(target.shape[0], 1)
    for _ in range(iters):
        err = target - arm.tip(q)
        J = arm.tip_jac(q)[:, :, :2]
        dq = torch.linalg.solve(J.transpose(1, 2) @ J + 1e-6 * torch.eye(2, device=q.device),
                                (J.transpose(1, 2) @ err.unsqueeze(-1))).squeeze(-1)
        q = q.clone()
        q[:, :2] = q[:, :2] + 0.8 * dq
    return q


def coriolis_fd(arm, q, qd, eps=1e-3):
    Mp, Mm = arm.mass_matrix(q + eps * qd), arm.mass_matrix(q - eps * qd)
    Mdot_qd = ((Mp - Mm) / (2 * eps) @ qd.unsqueeze(-1)).squeeze(-1)
    dT = []
    for k in range(3):
        dq = torch.zeros_like(q); dq[:, k] = eps
        Tp = 0.5 * (qd.unsqueeze(1) @ arm.mass_matrix(q + dq) @ qd.unsqueeze(-1)).reshape(-1)
        Tm = 0.5 * (qd.unsqueeze(1) @ arm.mass_matrix(q - dq) @ qd.unsqueeze(-1)).reshape(-1)
        dT.append((Tp - Tm) / (2 * eps))
    return Mdot_qd - torch.stack(dT, -1)


if __name__ == "__main__":
    arm0 = Arm(2)
    q = torch.tensor([[1.0, 1.2, 0.3], [0.3, 2.0, -0.5]], device=dev)
    qd = torch.tensor([[1.0, -2.0, 3.0], [0.5, 1.5, -2.0]], device=dev)
    # Jacobian-built mass matrix vs closed form
    th1, th2, th3 = arm0.angles(q); e1, e2, e3 = _ep(th1), _ep(th2), _ep(th3); z = torch.zeros_like(e1)
    L1, L2, L3 = arm0.L
    J1 = torch.stack([0.5 * L1 * e1, z, z], -1); J2 = torch.stack([L1 * e1 + 0.5 * L2 * e2, 0.5 * L2 * e2, z], -1)
    J3 = torch.stack([L1 * e1 + L2 * e2 + 0.5 * L3 * e3, L2 * e2 + 0.5 * L3 * e3, 0.5 * L3 * e3], -1)
    S = torch.tensor([[1., 0, 0], [1, 1, 0], [1, 1, 1]], device=dev)
    Mj = sum(arm0.M[i] * J.transpose(1, 2) @ J for i, J in enumerate((J1, J2, J3))) + sum(arm0.I[i] * torch.outer(S[i], S[i]) for i in range(3))
    print("M closed vs jacobian max abs diff:", (Mj - arm0.mass_matrix(q)).abs().max().item())
    print("C closed vs finite diff max abs diff:", (coriolis_fd(arm0, q, qd) - arm0.coriolis(q, qd)).abs().max().item())
    # energy check: no muscles, no damping, no noise -> kinetic energy conserved
    arm = Arm(4, dt=0.0005, noise=False, fatigue=False)
    arm.b = torch.zeros(3, device=dev); arm.kp = torch.zeros(3, device=dev)
    q0 = torch.tensor([[1.0, 1.2, 0.3]] * 4, device=dev)
    arm.reset(q0)
    arm.qd = torch.tensor([[1.0, -2.0, 3.0], [0.5, 0.5, 0.5], [-1, 2, -1], [2, 0, 0]], device=dev)
    arm.qlim = torch.tensor([[-9, 9], [-9, 9], [-9, 9.]], device=dev)
    def ke():
        return 0.5 * (arm.qd.unsqueeze(1) @ arm.mass_matrix(arm.q) @ arm.qd.unsqueeze(-1)).reshape(-1)
    e0 = ke()
    for _ in range(2000):
        arm.step(torch.zeros(4, NM, device=dev), torch.zeros(4, device=dev))
    print("KE drift over 1 s free swing:", ((ke() - e0) / e0).tolist())
    # reach check
    arm2 = Arm(1)
    tgt = torch.tensor([[0.15, 0.42]], device=dev)
    q = ik(arm2, tgt)
    print("ik q", q.tolist(), "tip", arm2.tip(q).tolist())
