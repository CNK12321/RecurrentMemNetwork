"""Recurrent memory network with provenance-tag credit assignment.

Each state has three stochastic gates (go / read / write).  Every signal
(activation or stored value) carries the set of gate events that produced it.
When an output is judged, every event on its set shares the reward, so a write
is credited only when its stored value is read on a path that gets rewarded,
and a read is credited through the value it pulled.
"""
import sys
import numpy as np

GO, READ, WRITE = 0, 1, 2
EMPTY = frozenset()


class Net:
    """Layout: inputs -> clusters -> bridge neurons -> outputs (+ halt).

    Each cluster is densely connected inside itself and sparsely (prob `inter`)
    to the other clusters. Inputs feed every cluster neuron. Only the bridge
    neurons read from the clusters, and only the bridges feed the output and
    halt units, so all information to the outputs passes through the bridges.
    """

    def __init__(self, n_clusters=4, cluster_size=10, n_bridge=8, inter=0.1,
                 T=0.4, seed=0, th0=1.5, use_halt=True, slack=1, partial=0.3):
        assert 5 <= n_bridge <= 10
        self.use_halt = use_halt
        self.slack = slack                         # extra output ticks with no fixed target
        self.partial = partial                     # credit for a digit from the sequence firing off-target
        self.rng = np.random.default_rng(seed)
        self.T = T
        self.T_out = None                          # separate temperature for output/halt units
        self.n_in, self.n_out = 11, 10            # inputs: digits 0-9 + "done"
        self.inputs = list(range(self.n_in))
        k = self.n_in
        self.clusters = []
        for _ in range(n_clusters):
            self.clusters.append(list(range(k, k + cluster_size)))
            k += cluster_size
        self.bridges = list(range(k, k + n_bridge)); k += n_bridge
        self.outputs = list(range(k, k + self.n_out)); k += self.n_out
        self.halt = k; self.N = N = k + 1
        cl_all = [s for c in self.clusters for s in c]
        self.groups = [cl_all, self.bridges, self.outputs + [self.halt]]   # run order
        self.proc = [s for g in self.groups for s in g]

        # mask[s, i] = 1 if source i may connect into state s
        m = np.zeros((N, N))
        for c in self.clusters:
            for s in c:
                m[s, self.inputs] = 1
                m[s, c] = 1
                others = [i for i in cl_all if i not in c]
                m[s, others] = self.rng.random(len(others)) < inter
        for s in self.bridges:
            m[s, cl_all] = 1
        for s in self.outputs + [self.halt]:
            m[s, self.bridges] = 1
        np.fill_diagonal(m, 0)
        self.mask = m
        self.W = self.rng.uniform(-1, 1, (3, N, N)) * m
        self.th = th0 + self.rng.uniform(-0.3, 0.3, (3, N))
        self._update_callorder()

    def _update_callorder(self):
        # Groups run in order (clusters, bridges, outputs); inside a group, the
        # state wired most strongly from earlier states / least to later ones goes first.
        self.order = []
        for grp in self.groups:
            eff = [(self.W[GO, s, self.inputs].sum() - self.W[GO, s, self.outputs].sum()) / self.th[GO, s]
                   for s in grp]
            self.order += [grp[i] for i in np.argsort(eff)[::-1]]

    def T_of(self, s):
        is_unit = s >= self.outputs[0]
        return self.T_out if (is_unit and self.T_out is not None) else self.T

    def _p(self, total, s):
        return 1 / (1 + np.exp(-(total - self.th[:, s]) / self.T_of(s)))

    def run(self, seq, greedy=False, grads=None):
        """Run one episode. Returns list of per-output-tick exact-correctness.
        If grads=(dW, dth) is given, accumulate policy-gradient credit into it."""
        N, T = self.N, self.T
        written = np.zeros(N)
        wtags = [EMPTY] * N
        events = {}                                # id -> (gate, state, snapshot, p, fired)
        n = len(seq)
        schedule = [(d, False, None) for d in seq]
        schedule += [(None, True, seq[j]) for j in range(n)]
        if self.use_halt:
            schedule.append((None, True, "halt"))
        schedule += [(None, True, "slack")] * self.slack
        correct = []
        self.disc = []

        def credit(ids, r):
            if grads is None or not ids:
                return
            share = r / len(ids)
            for eid in ids:
                g, s, snap, p, fired = events[eid]
                coef = share * (fired - p) / self.T_of(s)
                grads[0][g, s] += coef * snap
                grads[1][g, s] -= coef

        for digit, done, target in schedule:
            act = np.zeros(N)
            tags = [EMPTY] * N
            if digit is not None:
                act[digit] = 1
            if done:
                act[10] = 1
            snap, dirty = act.copy(), False
            unit_ev = {}
            for s in self.order:
                ps = self._p(self.W[:, s, :] @ act, s)
                fired = (ps > 0.5) if greedy else (self.rng.random(3) < ps)
                is_unit = s in self.outputs or s == self.halt
                if dirty and (fired.any() or is_unit):
                    snap, dirty = act.copy(), False
                pre_act = snap
                for g in (GO, READ, WRITE):
                    if not fired[g] and not (is_unit and g != WRITE):
                        continue
                    eid = len(events)
                    events[eid] = (g, s, pre_act, ps[g], float(fired[g]))
                    if is_unit:
                        unit_ev[(g, s)] = eid
                    if not fired[g]:
                        continue
                    if g == GO:
                        src = [i for i in range(N) if pre_act[i] > 0 and self.W[GO, s, i] > 0]
                        t = set().union(*(tags[i] for i in src)) if src else set()
                        act[s] = 1
                        tags[s] = frozenset(t | {eid})
                    elif g == READ:
                        act[s] = written[s]
                        tags[s] = (wtags[s] | {eid}) if written[s] > 0 else EMPTY
                    else:
                        written[s] = act[s]
                        wtags[s] = (tags[s] | {eid}) if act[s] > 0 else EMPTY
                    dirty = True
            # judge outputs this tick
            if target is None or target == "slack":
                want = set()
            elif target == "halt":
                want = {self.halt}
            else:
                want = {self.outputs[target]}
            active = {u for u in self.outputs + ([self.halt] if self.use_halt else []) if act[u] > 0}
            if done and target != "slack":
                correct.append(active == want)
                if isinstance(target, (int, np.integer)):   # target digit fired minus any wrong digit fired
                    others = [u for u in self.outputs if u != self.outputs[target]]
                    self.disc.append(float(self.outputs[target] in active)
                                     - float(any(u in active for u in others)))
            if not done:                           # only judge outputs in the output phase
                continue
            in_seq = {self.outputs[d] for d in seq}
            for u in active:
                if u in want:
                    r = 1.0
                elif u in in_seq:                  # right digit, wrong place: small positive
                    r = self.partial
                else:
                    r = -1.0
                credit(list(tags[u]), r)
            for u in want - active:                # missed: blame the unit's own go/read decisions
                credit([unit_ev[(GO, u)]], -1.0)
                credit([unit_ev[(READ, u)]], -1.0)
        return correct


def evaluate(net, length, alpha, n=200, rng=None):
    rng = rng or np.random.default_rng(123)
    ok = 0
    for _ in range(n):
        seq = list(rng.integers(0, alpha, length))
        ok += all(net.run(seq, greedy=True))
    return ok / n


def train(net, stages, lr=0.05, batch=8, max_eps=6000, check=500, target=0.9, log=print):
    rng = np.random.default_rng(1)
    total = 0
    for length, alpha in stages:
        for ep in range(0, max_eps, batch):
            dW, dth = np.zeros_like(net.W), np.zeros_like(net.th)
            for _ in range(batch):
                seq = list(rng.integers(0, alpha, length))
                net.run(seq, grads=(dW, dth))
            net.W = np.clip(net.W + lr * dW / batch, -2, 2) * net.mask
            net.th = np.clip(net.th + lr * dth / batch, 0.2, 4.0)
            total += batch
            if (ep + batch) % check == 0:
                net._update_callorder()
                acc = evaluate(net, length, alpha)
                log(f"len={length} alpha={alpha} eps={total} greedy exact-match={acc:.2f}")
                if acc >= target:
                    break
        else:
            net._update_callorder()
    return net


if __name__ == "__main__":
    net = Net(seed=0)
    stages = [(1, 2), (1, 4), (1, 10), (2, 10), (3, 10)]
    train(net, stages)


def train_adaptive(net, length, alpha, T_out=None, T_hi=2.0, T_lo=0.3, lr_hi=0.3, lr_lo=0.01,
                   s_ref=0.2, ema=0.02, batch=8, max_eps=20000, check=2000, log=print, seed=1):
    """High chaos + high learning rate until something starts returning the right digit;
    then both fall with the moving success rate s (saturating at s_ref)."""
    rng = np.random.default_rng(seed)
    s = 0.0
    for ep in range(0, max_eps, batch):
        sat = min(1.0, s / s_ref)
        net.T = T_hi + (T_lo - T_hi) * sat
        net.T_out = T_out
        lr = lr_hi + (lr_lo - lr_hi) * sat
        dW, dth = np.zeros_like(net.W), np.zeros_like(net.th)
        for _ in range(batch):
            seq = list(rng.integers(0, alpha, length))
            net.run(seq, grads=(dW, dth))
            s += ema * (max(0.0, float(np.mean(net.disc))) - s)
        net.W = np.clip(net.W + lr * dW / batch, -2, 2) * net.mask
        net.th = np.clip(net.th + lr * dth / batch, 0.2, 4.0)
        if (ep + batch) % check == 0:
            net._update_callorder()
            T_save, net.T = net.T, T_lo
            acc = evaluate(net, length, alpha)
            net.T = T_save
            log(f"eps={ep + batch} s={s:.2f} T={net.T:.2f} lr={lr:.3f} greedy={acc:.2f}")
    return net
