"""The 2022 paper's inextensible cloth solve (Coltraro et al., Applied Mathematical Modelling 101, 2022, secs. 5 and
7.1) as a drop-in subclass of clothilde-sim's Cloth, used for the reproduction check.

Constraints, all solved jointly (one constraint vector C, one Jacobian):
  * interior nodes k: E_k, F_k, G_k of the metric equal their rest values (FEM weighted-Galerkin, eq. 11, Alg. 1)
  * corners: F_k only (Remark 5.2)
  * every boundary edge: |p_a - p_b|^2 = l^2 (eq. 7)
  * scripted nodes: p_c = u (positional constraints, sec. 6 item 3)
Time step: implicit Euler unconstrained step (clothilde's ImplicitEuler), then the fast projection (sec. 7.1):
  repeat  dphi = -M^-1 dC^T (dC M^-1 dC^T)^-1 C(phi)  until every constraint is within `tol` (relative),
followed by v = (phi_new - phi_old) / dt. No self-collisions (the free-hanging model has none); clothilde's floor
projection at z = 0 is kept (the recordings never reach the floor).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sksparse.cholmod import cholesky_AAt

from implementation.Cloth import Cloth

_G = 1.0 / np.sqrt(3.0)
_GAUSS = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]      # 2x2 Gauss points, weights 1


def _shape(xi, eta):
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    Nx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    Ne = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, Nx, Ne


def metric_tensors(X, T):
    """COO (k, i, j, value) of T_E, T_F, T_G (Algorithm 1) and the lumped masses m_k, from rest positions X."""
    n = len(X)
    m = np.zeros(n)
    parts = {"E": [], "F": [], "G": []}
    for q in T:
        P = X[q]
        tE = np.zeros((4, 4, 4)); tF = np.zeros((4, 4, 4)); tG = np.zeros((4, 4, 4))
        for xi, eta in _GAUSS:
            N, Nx, Ne = _shape(xi, eta)
            px, pe = Nx @ P, Ne @ P
            E, F, G = px @ px, px @ pe, pe @ pe
            dA = np.sqrt(abs(E * G - F * F))
            m[q] += N * dA                     # row sums of the element mass matrix (sum_b N_a N_b = N_a)
            tE += np.einsum("k,i,j->kij", N, Nx, Nx) * dA
            tF += 0.5 * (np.einsum("k,i,j->kij", N, Nx, Ne) + np.einsum("k,i,j->kij", N, Ne, Nx)) * dA
            tG += np.einsum("k,i,j->kij", N, Ne, Ne) * dA
        kk, ii, jj = np.meshgrid(np.arange(4), np.arange(4), np.arange(4), indexing="ij")
        for name, t in (("E", tE), ("F", tF), ("G", tG)):
            parts[name].append((q[kk.ravel()], q[ii.ravel()], q[jj.ravel()], t.ravel()))
    out = {}
    for name, lst in parts.items():
        k = np.concatenate([a[0] for a in lst]); i = np.concatenate([a[1] for a in lst])
        j = np.concatenate([a[2] for a in lst]); v = np.concatenate([a[3] for a in lst])
        out[name] = (k, i, j, v / m[k])
    return out, m


class InextensibleCloth(Cloth):
    """Cloth with the 2022 paper's jointly-solved FEM inextensibility (see module docstring)."""

    def __init__(self, verts, faces, **kw):
        super().__init__(verts, faces, **kw)
        X = np.asarray(self.positions, dtype=float)
        n = self.n_verts
        tens, self.m_paper = metric_tensors(X, self.faces)
        interior = np.setdiff1d(np.arange(n), self.nodes_bnd)
        corners = np.asarray(self.corners, dtype=int)
        rows = []                                   # (node set, tensor) blocks in constraint order
        for name, nodes in (("E", interior), ("F", np.union1d(interior, corners)), ("G", interior)):
            k, i, j, v = tens[name]
            keep = np.isin(k, nodes)
            remap = -np.ones(n, dtype=int)
            remap[nodes] = np.arange(len(nodes))
            rows.append((name, nodes, remap[k[keep]], i[keep], j[keep], v[keep]))
        off = 0
        K, I, J, V = [], [], [], []
        self.metric_node = []
        for name, nodes, kk, ii, jj, vv in rows:
            K.append(kk + off); I.append(ii); J.append(jj); V.append(vv)
            self.metric_node.append(nodes)
            off += len(nodes)
        self.TK, self.TI, self.TJ, self.TV = (np.concatenate(K), np.concatenate(I), np.concatenate(J),
                                              np.concatenate(V))
        self.n_metric = off
        self.metric_node = np.concatenate(self.metric_node)
        self.bnd_edges = np.asarray(self.edges_bnd, dtype=int)
        e = X[self.bnd_edges[:, 0]] - X[self.bnd_edges[:, 1]]
        self.l2 = np.einsum("ij,ij->i", e, e)
        self.C0 = self._metric(X)
        # relative-error scale of each metric constraint: (E0 + G0)/2 at its node (F0 ~ 0 on a rectangular grid)
        eg = 0.5 * (self._metric_named(X, "E") + self._metric_named(X, "G"))
        self.scale = np.concatenate([np.maximum(eg[nodes], 1e-12) for nodes in
                                     (interior, np.union1d(interior, corners), interior)])
        self.minv_sqrt = np.tile(1.0 / np.sqrt(self.m_paper), 3)    # M^-1/2 for (x | y | z) ordering
        self.proj_iters = []

    def _metric(self, P):
        dots = np.einsum("ij,ij->i", P[self.TI], P[self.TJ])
        return np.bincount(self.TK, weights=self.TV * dots, minlength=self.n_metric)

    def _metric_named(self, P, name):
        tens, _ = metric_tensors(np.asarray(self.history_pos[0]), self.faces)
        k, i, j, v = tens[name]
        return np.bincount(k, weights=v * np.einsum("ij,ij->i", P[i], P[j]), minlength=self.n_verts)

    def constraints(self, P, u, control):
        """C (nc,), Jacobian dC (nc, 3n) in clothilde's (x | y | z) flattening."""
        n = self.n_verts
        cm = self._metric(P) - self.C0
        e = P[self.bnd_edges[:, 0]] - P[self.bnd_edges[:, 1]]
        ce = np.einsum("ij,ij->i", e, e) - self.l2
        cu = (P[control] - u).ravel(order="F") if len(control) else np.zeros(0)
        blocks = []
        for d in range(3):                                         # metric: dC_k/dp_i = 2 sum_j T_kij p_j
            blocks.append(sp.csr_matrix((2 * self.TV * P[self.TJ, d], (self.TK, self.TI)), shape=(self.n_metric, n)))
        Gm = sp.hstack(blocks)
        nb = len(self.bnd_edges)
        rr = np.concatenate([np.arange(nb)] * 6)
        a, b = self.bnd_edges[:, 0], self.bnd_edges[:, 1]
        cc = np.concatenate([a, a + n, a + 2 * n, b, b + n, b + 2 * n])
        vv = np.concatenate([2 * e[:, 0], 2 * e[:, 1], 2 * e[:, 2], -2 * e[:, 0], -2 * e[:, 1], -2 * e[:, 2]])
        Ge = sp.csr_matrix((vv, (rr, cc)), shape=(nb, 3 * n))
        nc = len(control)
        if nc:
            ctl = np.asarray(control)
            Gu = sp.csr_matrix((np.ones(3 * nc), (np.arange(3 * nc), np.concatenate([ctl, ctl + n, ctl + 2 * n]))),
                               shape=(3 * nc, 3 * n))
            Gc = sp.vstack([Gm, Ge, Gu]).tocsc()
        else:
            Gc = sp.vstack([Gm, Ge]).tocsc()
        return np.concatenate([cm, ce, cu]), Gc, cm, ce, cu

    def _residual(self, cm, ce, cu):
        return float(np.sqrt(np.sum((cm / self.scale) ** 2) + np.sum((ce / self.l2) ** 2) + np.sum(cu ** 2)))

    def project(self, phi, u, control, max_iter=100, reg=1e-14):
        """Fast projection (sec. 7.1): repeated M-orthogonal linearised projections onto C = 0.
        The paper's step is the undamped one (lam = 0). When the scripted corners are slightly farther apart than the
        rest top edge allows (mocap noise), C = 0 has no solution and undamped steps thrash; a step that does not lower
        the scaled residual is then retried with Levenberg damping (dC M^-1 dC^T + lam I), which converges to the
        least-squares compromise. Feasible frames never trigger the damping."""
        n = self.n_verts
        P = phi.reshape((n, 3), order="F").copy()
        C, G, cm, ce, cu = self.constraints(P, u, control)
        res = self._residual(cm, ce, cu)
        lam, it, damped, converged = 0.0, 0, False, False
        for it in range(max_iter):
            err = max(np.max(np.abs(cm) / self.scale), np.max(np.abs(ce) / self.l2),
                      np.max(np.abs(cu)) if len(cu) else 0.0)
            if err <= self.tol and (not len(cu) or np.max(np.abs(cu)) < 1e-9):
                converged = True
                break
            A = (G @ sp.diags(self.minv_sqrt)).tocsc()                  # dC M^-1/2
            diag = max(1.0, float(A.multiply(A).sum()) / A.shape[0])
            accepted = False
            while True:
                factor = cholesky_AAt(A, beta=max(reg * diag, lam))
                Pn = P + (-(self.minv_sqrt * (A.T @ factor(C)))).reshape((n, 3), order="F")
                Cn, Gn, cmn, cen, cun = self.constraints(Pn, u, control)
                rn = self._residual(cmn, cen, cun)
                if it == 0 or rn < res * (1 - 1e-9):   # first step = the paper's undamped projection, always taken
                    accepted = True
                    break
                if lam > 1e6 * diag:
                    break
                lam = 1e-6 * diag if lam == 0.0 else 10.0 * lam
                damped = True
            if not accepted:                           # least-squares stationary point: keep the last accepted state
                break
            P, C, G, cm, ce, cu, res = Pn, Cn, Gn, cmn, cen, cun, rn
            lam = lam / 10.0 if lam > 1e-6 * diag else 0.0
        if len(control):
            P[list(control)] = u                       # scripted nodes exactly on their paths
        self.proj_iters.append(it)
        self.proj_damped = getattr(self, "proj_damped", 0) + int(damped)
        self.proj_unconverged = getattr(self, "proj_unconverged", 0) + int(not converged)
        return P.reshape((3 * n,), order="F")

    def simulate(self, u, control):
        """One frame: sub_steps x (implicit Euler unconstrained step + joint fast projection + floor)."""
        control = list(control)
        u = np.asarray(u, dtype=float).reshape((len(control), 3)) if len(control) else np.zeros((0, 3))
        pos0 = self.positions[control].copy() if len(control) else np.zeros((0, 3))
        for s in range(self.sub_steps):
            us = pos0 + self.t_int[s] * (u - pos0)
            phi0 = self.positions.reshape((3 * self.n_verts,), order="F")
            phi = self.ImplicitEuler()
            phi = self.project(phi, us, control)
            phi = self.floorCollisions(phi)            # replay.run disables it for free-hanging runs
            self.velocities = ((phi - phi0) / self.dt).reshape((self.n_verts, 3), order="F")
            self.positions = phi.reshape((self.n_verts, 3), order="F")
        self.history_pos.append(self.positions)
        self.history_vel.append(self.velocities)
        self.total_iters += float(np.mean(self.proj_iters[-self.sub_steps:]))
