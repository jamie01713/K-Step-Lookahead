import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import itertools
import numpy as np
import scipy 
import utils
from policies.agent import *

MAX_EVI_ITER = 100
DIAMETER = 1
BIAS_SPAN = 1
BIAS = None
DELTA = 1e-5 # supposed to be T (horizon)
MITIGATION_STRENGTH = 1.0 # 1.0 : normal value
BIAS_ERROR_CUTOFF = 1.0 # 1.0 : normal value

def DIRAC(A, a): return [float(b == a) for b in A]
def BIRAC(A, a, b): return [float(c == a) - float(c == b) for c in A]
def UNIFORM(A, B): return [float(a in B)/len(B) for a in A]
def COMPOSITION(funs, v): 
    for f in funs:
        v = f(v)
    return v

def set_DIAMETER(value):
    global DIAMETER
    DIAMETER = value
def set_BIAS_SPAN(value):
    global BIAS_SPAN
    BIAS_SPAN = value
def set_BIAS(vec):
    global BIAS
    BIAS = [v_i for v_i in vec]
    set_BIAS_SPAN(max(BIAS) - min(BIAS))

class BellmanOperator:

    KERNEL_OPTIONS = {
        "weissman": "_kernel_weissman",
        "likelihood": "_kernel_likelihood",
        "bernstein": "_kernel_bernstein",
        "empirical": "_kernel_empirical",
        "trivial": "_kernel_trivial",
        "betaAzuma": "_kernel_beta_azuma",
        "betaBernstein": "_kernel_beta_bernstein",
        "betaOracle": "_kernel_beta_oracle",
    }

    def weissman_lproba(u, hat_p, xi):
        """ Solve max < q|u > for ||hat_p - q||_1 <= xi """
        n = len(u)
        u_sorted = sorted(list((u_i, i) for i, u_i in enumerate(u)), reverse=True)
        i, j = 0, n-1
        p = [hat_p_x for hat_p_x in hat_p]
        while i != j:
            _, x_i = u_sorted[i]
            _, x_j = u_sorted[j]
            dp, case = min((xi/2,0), (1.0-p[x_i],1), (p[x_j],2))
            p[x_i] += dp
            p[x_j] -= dp
            xi -= 2 * dp
            if case == 0: break
            if case == 1: i += 1
            if case == 2: j -= 1
        return sum(pi * ui for pi, ui in zip(p, u))

    def _likelihood_Newton(f, target_value, jac_f, u_0, precision=1e-6, v_max=0.0):
        """ Newton algorithm used in 'likelihood_lproba' """
        u = u_0
        i = 0
        while u - v_max > precision and abs(f(u) - target_value) > precision:
            u_new = u + (target_value - f(u))/jac_f(u)
            u = v_max + 0.5 * (u - v_max) if u_new <= v_max else u_new
            i += 1
            if i >= 100: 
                print("[E] Newton miserably failed. Killing it.")
                break
        return u
    
    def likelihood_lproba(v, hat_p, xi, eps=0.001):
        """ 
        Solve max < q|v > for KL(hat_p||q) <= xi 

        This code is impossible to understand as is.
        Go read the paper of Filippi et. al. on KLUCRL.
        This code takes the same notations as their appendix A,B.
        A few hacks are added for numerical stability.
        """

        # Short notations + HACK (making p epsilon-uniform)
        p = hat_p
        for i in range(len(p)): p[i] += 1e-3
        s = sum(p)
        for i in range(len(p)): p[i] /= s

        # Avoid null gradient issues
        if max(v) == min(v): return v[0]

        # Compute suports
        n     = len(p)
        infty = 1e15
        z     = [i for (i, p_i) in enumerate(p) if p_i == 0]
        bar_z = [i for (i, p_i) in enumerate(p) if p_i >  0]
        v_max = max(v)
        v_thr = max(v_i for p_i, v_i in zip(p, v) if p_i > 0.0)
        i_opt = [i for i in z if v[i] == v_max]

        # Objective function, its jacobian
        f     = lambda u: \
            sum(p[i] * np.log (u - v[i]) for i in bar_z) + \
            np.log(sum(p[i] / (u - v[i]) for i in bar_z))
        jac_f = lambda u: \
            sum(p[i] / (u - v[i])    for i in bar_z) + \
            sum(-p[i]/ (u - v[i])**2 for i in bar_z) / \
            sum(p[i] / (u - v[i])    for i in bar_z)
        
        # Optimisze f, see Filippi's paper, appendix A.
        f_set = [i for i in i_opt if v_thr < v_max and f(v[i]) < eps]
        if f_set:
            u = v[f_set[0]]
            r = 1 - np.exp(f(u) - eps)
            q = [0 for _ in range(n)]
            for i in i_opt: q[i] = r / len(i_opt)
        else:
            r = 0
            q = [0 for _ in range(n)]
            m1 = sum(p_i * v_i for p_i, v_i in zip(p, v))
            u_0= sum(p_i * (v_i - m1)**2 for p_i, v_i in zip(p, v))
            # u_0  = sum(p_i * v_i * v_i for p_i, v_i in zip(p, v))
            # u_0 -= sum(p_i * v_i for p_i, v_i in zip(p, v))**2
            u_0  = max(1e-12, u_0)
            u_min = max(v) + 1e-10
            u_0 = max(u_min, v_thr + np.sqrt(u_0/(2*eps)))    
            # u_0  = v_thr + np.sqrt(u_0 / (2*eps)) 
            u    = BellmanOperator._likelihood_Newton(f, eps, jac_f, u_0, 1e-6, v_thr)
            if (not np.isfinite(u)) or (u <= u_min):
               u = u_min
            if u == v_thr: u += 1e-6
        
        # Normalize q
        t_q = [0 for _ in range(n)]
        for i in bar_z: t_q[i] = p[i] / (u - v[i])
        sum_t_q = sum(t_q)
        for i in bar_z: q[i] = (1 - r) * t_q[i] / sum_t_q
        
        # Return result <q|v>
        return sum(qi * vi for qi, vi in zip(q, v))

    def bernstein_lproba(u, hat_p, xi):
        """ 
        Solve max < q|u > for q satisfying Bernstein like inequalities wrt p
        (see UCRL2-B's paper for more information). 
    
        - xi: array of error at coordinate xi
        """
        p = hat_p # for convenience
        S        = len(u)
        v_sorted = sorted(list((u_i, i) for i, u_i in enumerate(u)), reverse=True)
        a        = [max(0.0, p[i] - xi[i]) for _, i in v_sorted]
        b        = [min(1.0, p[i] + xi[i]) for _, i in v_sorted]
        i        = 0
        q        = [max(0.0, p[i] - xi[i]) for i in range(S)]
        d        = 1.0 - sum(q)
        while d > 0.0 and i < S:
            v_i, j_i = v_sorted[i]    
            delta    = min(d, b[i] - a[i])
            d        = d - delta
            i       += 1
            q[j_i]  += delta
        assert (0.95 < sum(q) < 1.05)
        return sum(qi * ui for qi, ui in zip(q, u))

    def __init__(self, agent, config, trunc):
        self.config = config
        self._parse_r_config(agent, config, trunc)
        self._parse_p_config(agent, config, trunc)

    def _parse_r_config(self, agent, config, trunc):
        options = config["r-type"].split("+")
        rew_list = []
        for option in options:
            if option == "azuma":
                rew_list.append(self._init_r_azuma(config))
            elif option == "alphaAzuma":
                rew_list.append(self._init_r_azuma_alpha(config))
            else:
                raise Exception(f"Unknown reward config {config['r-type']}")
        self.tilde_r = min(rew_list)

    def _parse_p_config(self, agent, config, trunc):
        options = config["p-type"].split("+")
        max_pv_list = []
        for option in options:
            name = BellmanOperator.KERNEL_OPTIONS[option]
            f = self.__getattribute__(name)
            max_pv_list.append(f(agent, config, trunc))
        self.max_pv = lambda v: min(mpv(v) for mpv in max_pv_list)

    def _init_r_azuma(self, config):
        n, hat_r, delta = config["n"], config["hat_r"], config["delta"]
        if n == 0:
            return 1.0
        else:
            log = max(0.0, np.log(np.sqrt(1+n)/delta))
            xi = np.sqrt(0.5*(n+1)*log)/n
            return min(1.0, hat_r + xi)

    def _init_r_azuma_alpha(self, config):
        n, hat_r, delta = config["n"], config["hat_r"], config["delta"]
        if n == 0:
            return 1.0
        else:
            log = max(0.0, np.log(np.sqrt(1+n)/delta))
            xi = np.sqrt(0.5*(n+1)*log)/n
            bonus = BIAS_SPAN * xi
            return min(1.0, hat_r + xi + bonus)

    def _kernel_empirical(self, agent, config, trunc):
        hat_p = config["hat_p"]
        self.max_pv = lambda v: sum(p_i * v_i for p_i, v_i in zip(hat_p, v))

    def _kernel_weissman(self, agent, config, trunc):
        S = config["S"]
        n, hat_p, delta = config["n"], config["hat_p"], config["delta"]
        if n == 0:
            return lambda v: max(v)
        else:
            log = max(0.0, np.log(2*np.sqrt(1+n)/delta))
            xi = np.sqrt(0.5*(n+1)*S*log)/n
            return lambda v: BellmanOperator.weissman_lproba(v, hat_p, xi)

    def _kernel_likelihood(self, agent, config, trunc):
        S = config["S"]
        n, hat_p, delta = config["n"], config["hat_p"], config["delta"]
        if n == 0:
            return lambda v: max(v)
        else:
            log = np.log(1/delta) + (S-1)*(1+np.log(1+n/(S-1)))
            xi = log / n
            return lambda v: BellmanOperator.likelihood_lproba(v, hat_p, xi)

    def _kernel_bernstein(self, agent, config, trunc):
        S = config["S"]
        n, hat_p, delta = config["n"], config["hat_p"], config["delta"]
        if n == 0:
            return lambda v: max(v)
        else:
            log = 2 * np.log((1+n)/delta)
            xi  = [ np.sqrt(2*pi*(1-pi)*log/n) + 3.0 * log/n for pi in hat_p ]
            return lambda v: BellmanOperator.bernstein_lproba(v, hat_p, xi)

    def _kernel_trivial(self, agent, config, trunc):
        return lambda v: max(v)

    def _kernel_beta_azuma(self, agent, config, trunc):
        n, hat_p, delta = config["n"], config["hat_p"], config["delta"]
        if n == 0:
            return lambda v: max(v)
        else:
            log = max(0.0, np.log(np.sqrt(1+n)/delta))
            xi = np.sqrt(0.5*(n+1)*log)/n
            bonus = BIAS_SPAN * xi
            return lambda v: min(
                max(v),
                bonus + sum(p_i * v_i for p_i, v_i in zip(hat_p, v)),
            )

    def _kernel_beta_bernstein(self, agent, config, trunc):
        n, hat_p, delta = config["n"], config["hat_p"], config["delta"]
        bias_diff = config["bias differences"]
        bias_erro = agent.bias_error()
        bias_prox = [bias_diff[0, x] for x in agent.S]
        bias_prox = trunc(bias_prox)
        co_h = max(bias_erro.values())
        sp_h = max(bias_diff.values()) + co_h
        p_h  = sum(p_i * bias_i for p_i, bias_i in zip(hat_p, bias_prox))
        var  = sum(p_i * (bias_i - p_h)**2 for p_i, bias_i in zip(hat_p, bias_prox))
        wvar = var + 8 / MITIGATION_STRENGTH * sp_h * \
                min(sum(hat_p[y] * bias_erro[y, x] for y in agent.S) for x in agent.S)
        # print(bias_prox, co_h, sp_h, p_h, var, wvar)
        if n == 0:
            return lambda v: max(v)
        else:
            log = max(0.0, np.log(np.sqrt(1+n)/delta))
            bonus = np.sqrt(1+1/n) * (np.sqrt(2 * wvar * log/n) + 3 * sp_h * log / n)
            return lambda v: min(
                max(v),
                bonus + sum(p_i * v_i for p_i, v_i in zip(hat_p, v)),
            )

    def _kernel_beta_oracle(self, agent, config, trunc):
        n, hat_p, delta = config["n"], config["hat_p"], config["delta"]
        h = BIAS
        sp_h = max(h) - min(h)
        epsilon = 0.1 * sp_h
        p_h  = sum(p_i * h_i for p_i, h_i in zip(hat_p, h))
        var  = sum(p_i * (h_i - p_h)**2 for p_i, h_i in zip(hat_p, h))
        wvar = var # + 8 * sp_h * epsilon
        # print(bias_prox, co_h, sp_h, p_h, var, wvar)
        if n == 0:
            return lambda v: max(v)
        else:
            log = max(0.0, np.log(np.sqrt(1+n)/delta))
            bonus = np.sqrt(1+1/n) * (np.sqrt(2 * wvar * log/n) + 3 * sp_h * log / n)
            return lambda v: min(
                max(v),
                bonus + sum(p_i * v_i for p_i, v_i in zip(hat_p, v)),
            )

    def iterate(self, v, x=0):
        return self.tilde_r + self.max_pv(v)
        # return self.tilde_r + 0.5*(v[x] + self.max_pv(v)) ### PATCH

class EVI:

    def span_truncation(v, c):
        return [min(v_x, min(v) + c) for v_x in v]

    def poly_truncation(v, poly):
        v0 = [v_x - v[0] for v_x in v]
        u  = [0 for _ in v]
        u0 = [0 for _ in v]
        S  = list(range(len(v)))
        A_ub, b_ub = poly
        bounds = [(None, v0_x) for v0_x in v0]
        for x in S:
            c = - np.array(DIRAC(S, x))
            res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds) # , x0=u0)
            u0 = res.x
            if res.success == False:
                print("[E] Truncation failed with empty polyhedron")
                return v
            u[x] = u0[x]
        c = v[x] - v0[x] 
        u = [u_x + c for u_x in u]
        du = [v_i - u_i for v_i, u_i in zip(u, v)]
        if max(du) - min(du) > 0.05:
            pass
        return u

    def __init__(self, agent, config):
        self.Z = agent.Z.copy()
        self.S = agent.S.copy()
        self.A = { x: agent.A[x].copy() for x in self.S }
        self._parse_truncation_config(agent, config)
        self._parse_bo_config(agent, config, trunc=self.truncate)
        self._parse_pi_selector(config)

    def _parse_bo_config(self, agent, config, trunc=lambda v: v):
        r, p = agent.empirical_structure()
        self.bo = dict()
        b_diffe = agent.bias_differences_estimator()
        for x, a in self.Z:
            config_xa = dict()
            config_xa["bias differences"] = b_diffe
            config_xa["r-type"] = config["r-type"]
            config_xa["p-type"] = config["p-type"]
            config_xa["n"] = agent.N[x, a]
            config_xa["t"] = agent.t
            config_xa["delta"] = config["delta"]
            config_xa["hat_r"] = r[x, a]
            config_xa["hat_p"] = p[x, a]
            config_xa["S"] = len(self.S)
            self.bo[x,a] = BellmanOperator(agent, config_xa, trunc)

    def _parse_truncation_config(self, agent, config):
        options = config["truncation"].split("+")
        funs = []
        for option in options:
            if option == "identity":
                continue
            elif option == "span":
                c = BIAS_SPAN
                funs.append(lambda v: EVI.span_truncation(v, c))
            elif option == "polyhedron":
                h = agent.bias_differences_estimator()
                e = agent.bias_error()
                A_ub, b_ub = [], []
                for x, y in itertools.product(self.S, self.S):
                    if x <= y: continue
                    A_ub.append(BIRAC(agent.S, x, y))
                    A_ub.append(BIRAC(agent.S, y, x))
                    b_ub.append(h[y,x] + e[x,y]/MITIGATION_STRENGTH)
                    b_ub.append(h[x,y] + e[x,y]/MITIGATION_STRENGTH)
                for x, y, c_xy in agent.bias_prior:
                    A_ub.append(BIRAC(agent.S, x, y))
                    b_ub.append(c_xy)
                poly = (A_ub, b_ub)
                funs.append(lambda v: EVI.poly_truncation(v, poly))
            elif option == "oracle":
                h = BIAS
                epsilon = 0.1 * (max(h) - min(h))
                A_ub, b_ub = [], []
                for x, y in itertools.product(self.S, self.S):
                    if x <= y: continue
                    A_ub.append(BIRAC(agent.S, x, y))
                    A_ub.append(BIRAC(agent.S, y, x))
                    b_ub.append(h[x] - h[y] + epsilon)
                    b_ub.append(h[y] - h[x] + epsilon)
                poly = (A_ub, b_ub)
                funs.append(lambda v: EVI.poly_truncation(v, poly))
            else:
                print("[E] Unknown truncation operator", option)
        self.truncate = lambda v: COMPOSITION(funs, v)

    def _parse_pi_selector(self, config):
        if config["pi-type"] == "greedy-sd":
            self.pi_selector = self.greedy_sd
        elif config["pi-type"] == "greedy-sr":
            self.pi_selector = self.greedy_sr
        else:
            raise Exception(f"[F] Unknown policy selector {config['pi-type']}")

    def fixpoint(self, epsilon, u_init=None, truncate=True):
        """ Compute an epsilon span-fixpoint """

        # EVI's main loop
        u = [0 for _ in self.S] if not u_init else u_init
        iteration = 0
        sp = 0
        while True:
            iteration += 1
            if iteration >= MAX_EVI_ITER: 
                # print(f"[W] Killed infinite loop in EVI()")
                break
            
            # Iterate Bellman operator
            v = [-np.inf for _ in u]
            for x, a in self.Z:
                # if not act_allowed[x,a]: continue
                q_xa = self.bo[x,a].iterate(u, x)
                v[x] = max(v[x], q_xa)

            # Take truncation
            # v = [2.0*v_x for v_x in v] ### PATCH
            v = self.truncate(v) if truncate else v
            # v = [0.5*v_x for v_x in v] ### PATCH

            # Stop if span of difference is small enough
            du = [v_x - u_x for u_x, v_x in zip(u, v)]
            sp = max(du) - min(du)
            if sp < epsilon: break

            # For scalability
            u = [v_x - v[0] for v_x in v]

        # Finallzing things, storing the result 
        g = 0.5 * (max(du) + min(du))
        return v, g

    def greedy_sd(self, u, epsilon=0.01):
        """ Return a greedy policy with respect to u .
        epsilon is used for numerical stability """
        pi = [-1 for _ in u]
        v = [-np.inf for _ in u]
        q = dict()
        for x, a in self.Z:
            q[x,a] = self.bo[x,a].iterate(u)
            v[x] = max(v[x], q[x,a])
        v = self.truncate(v)
        for x, a in self.Z:
            if pi[x] == -1 and q[x,a] >= v[x] - epsilon:
                pi[x] = DIRAC(self.A[x], a)
        return pi

    def greedy_sr(self, u, epsilon=0.01):
        """ Return a randomized greedy policy with respect to u.
        epsilon is used for numerical stability """
        v = [-np.inf for _ in u]
        q = dict()
        for x, a in self.Z:
            q[x,a] = self.bo[x,a].iterate(u)
            v[x] = max(v[x], q[x,a])
        v = self.truncate(v)
        pi = [-1 for _ in u]
        for x in self.S:
            B_x = {b for b in self.A[x] if q[x,b] >= v[x] - epsilon}
            if len(B_x) == 0: 
                print(f"[W] Panicking with empty choice @ {self}")
                B_x = set(self.A[x])
            pi[x] = UNIFORM(self.A[x], B_x)
        return pi

class EVI_based(Agent):

    prefabs = {
        "UCRL2": {
            "name": "UCRL2", 
            "r-type": "azuma", 
            "p-type": "weissman", 
            "pi-type": "greedy-sd",
            "truncation": "identity",
            "episodes": "VT",
        },
        "UCRL2B": {
            "name": "UCRL2B",
            "r-type": "azuma",
            "p-type": "bernstein+weissman",
            "pi-type": "greedy-sd",
            "truncation": "identity",
            "episodes": "VT",
        },
        "KLUCRL": {
            "name": "KLUCRL",
            "r-type": "azuma",
            "p-type": "likelihood",
            "pi-type": "greedy-sd",
            "truncation": "identity",
            "episodes": "VT",
        },
        "REGAL": {
            "name": "REGAL", 
            "r-type": "azuma", 
            "p-type": "weissman", 
            "pi-type": "greedy-sd",
            "truncation": "span",
            "episodes": "VT",
        },
        "UCBVI-": {
            "name": "UCBVI-", 
            "r-type": "alphaAzuma", 
            "p-type": "empirical", 
            "pi-type": "greedy-sd",
            "truncation": "identity",
            "episodes": "DT",
        },
        "UCBVI": {
            "name": "UCBVI", 
            "r-type": "alphaAzuma", 
            "p-type": "empirical", 
            "pi-type": "greedy-sd",
            "truncation": "span",
            "episodes": "DT",
        },
        "PMEVI-": {
            "name": "PMEVI-", 
            "r-type": "azuma", 
            "p-type": "trivial+betaBernstein", 
            "pi-type": "greedy-sd",
            "truncation": "polyhedron",
            "episodes": "VT",
        },
        "PMEVI": {
            "name": "PMEVI", 
            "r-type": "azuma", 
            "p-type": "weissman+betaBernstein", 
            "pi-type": "greedy-sd",
            "truncation": "polyhedron",
            "episodes": "VT",
        },
        "PMEVI-UCRL2": {
            "name": "PMEVI-UCRL2", 
            "r-type": "azuma", 
            "p-type": "weissman+betaBernstein", 
            "pi-type": "greedy-sd",
            "truncation": "polyhedron",
            "episodes": "VT",
        },
        "PMEVI-UCRL2B": {
            "name": "PMEVI-UCRL2B", 
            "r-type": "azuma", 
            "p-type": "bernstein+betaBernstein", 
            "pi-type": "greedy-sd",
            "truncation": "polyhedron",
            "episodes": "VT",
        },
        "PMEVI*": {
            "name": "PMEVI*", 
            "r-type": "azuma", 
            "p-type": "weissman+likelihood+betaBernstein", 
            "pi-type": "greedy-sd",
            "truncation": "polyhedron",
            "episodes": "VT",
        },
        "PMEVI-KLUCRL": {
            "name": "PMEVI-KLUCRL", 
            "r-type": "azuma", 
            "p-type": "likelihood+betaBernstein", 
            "pi-type": "greedy-sd",
            "truncation": "polyhedron",
            "episodes": "VT",
        },
        "PMEVI-oracle": {
            "name": "PMEVI-oracle", 
            "r-type": "azuma", 
            "p-type": "weissman+likelihood+betaBernstein", 
            "pi-type": "greedy-sd",
            "truncation": "oracle+span",
            "episodes": "VT",
        },
    }
    
    def __init__(self, model, config,model_type):
        super().__init__(model)
        self.config = config
        self.set_bias_prior([])
        self.reset(model)
        self.set_name(config["name"])
    
    def reset(self, model):
        self.t = 1

        # Visit count, observations etc.
        self.N = {
            **{ z :        0 for z in self.Z },
            **{ (x, a, y): 0 for ((x, a), y) in itertools.product(self.Z, self.S) },
        }
        self.alpha = { z : 1.0 for z in self.Z }
        self.beta  = { z : 1.0 for z in self.Z }
        self.rewards = { z : 0.0 for z in self.Z }
        self.kernels = { z : 0.0 for z in self.Z }
        if utils.KNOWN_REWARDS: self.rewards = model.rewards().copy()
        if utils.KNOWN_KERNELS: self.kernels = model.kernels().copy()

        # Bias estimation stuff
        # - sign: sign[x, y] == +1 means "I am coming from x, waiting for y"
        #         sign[x, y] ==  0 means "I may have seen y, but didn't seen x since"
        # - commute: Number of observed commutations x <-> y
        # - balance: balance[x, y] = timewise sum of sign[x, y]
        # - ssum: ssum[x, y] = timewise sum of sign[x, y] * reward
        # - rsum: total reward (sum of rewards over time)
        self.sign    = { (x, y): 0.0 for x, y in itertools.product(self.S, self.S) } 
        self.travels = { (x, y): 0.0 for x, y in itertools.product(self.S, self.S) }
        self.signsum = { (x, y): 0.0 for x, y in itertools.product(self.S, self.S) }
        self.signedr = { (x, y): 0.0 for x, y in itertools.product(self.S, self.S) }
        self.rsum    = 0.0 # reward sum

        # Episoode stuff
        self.k  = 1   # label of current episodes
        self.Nk = { z : 0.0 for z in self.Z } # number of visits within episode
        self.tk = 1   # starting times of episodes
        self.Tk = 1   # duration of episodes
        self.gs = [1] # optimistic gains of episodes
        self.saved_u = [0 for _ in self.S] # saved value for code acceleration

        # EVI stuff
        self.pi = [UNIFORM(self.A[x], self.A[x]) for x in self.S] # current pol
        self.u_vi = [0 for _ in range(self.n_states)] # optimistic bias vector

    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    
    def set_bias_prior(self, prior):
        """
        Set the bias prior.
        Add prior knowledge of the form of a list of constraints:

        h(s) - h(s') <= c(s,s')

        each of them is modelized by tuple (s, s', c(s,s')).
        """
        self.bias_prior = prior

    def observe(self, x, a, r, y,done,truncated):
        """ Update inner data according to observations """
        # Usual observation
        self.t        += 1
        self.N[x,a]   += 1
        self.N[x,a,y] += 1
        self.rsum     += r
        if not utils.KNOWN_REWARDS:
            self.alpha[x,a] += float(r > 0.9)
            self.beta [x,a] += float(r < 0.1)
            self.rewards[x,a]  = (self.N[x,a]-1)*self.rewards[x,a] + r
            self.rewards[x,a] /=  self.N[x,a]
        # Update bias estimation data
        for x0, y0 in itertools.product(self.S, self.S):
            self.signsum[x0,y0] += self.sign[x0,y0]
            self.signedr[x0,y0] += self.sign[x0,y0] * r
        # Changing signs
        for x0 in self.S:
            if x0 == y: continue
            if self.sign[x0, y] == 1:
                self.travels[x0, y] += 1
                self.sign[x0, y] = 0
            self.sign[y, x0] = 1

    def bias_differences_estimator(self):
        """ Estimate local bias differences """
        h = { (x, y): 0.0 for x, y in itertools.product(self.S, self.S) }
        g = self.rsum/(self.t-1)
        # print("\n=> Achieved gain:", g)
        for x, y in itertools.product(self.S, self.S):
            if x == y: continue
            t_xy, t_yx = self.travels[x,y], self.travels[y,x]
            d_xy, d_yx = self.signsum[x,y], self.signsum[y,x]
            s_xy, s_yx = self.signedr[x,y], self.signedr[y,x]
            if t_xy + t_yx < 1: continue
            h[x, y]  = g * (d_xy - d_yx)
            h[x, y] -= s_xy - s_yx
            h[x, y] /= (t_xy + t_yx)
        return h

    def bias_error(self, epsilon=0.1, g=None):
        """ Estimate the plausible error of the bias differences estimator """
        
        ## Optimistic gain for inner regret estimation

        # # EVI-style (with UCRL2 parametrization)
        # config = EVI_based.prefabs["UCRL2"]
        # config["delta"] = 1.0/self.t
        # _, g = EVI(self, config).fixpoint(1e-3)

        # Old-optimistic-gain-style
        g = min(self.gs)

        # Estimation of inner regret
        regret = (self.t-1)*g - self.rsum

        # Computation of bias errro
        log = max(0.0, np.log(2/DELTA))
        error = dict()
        for x, y in itertools.product(self.S, self.S):
            if x == y:
                error[x, y] = 0.0
                continue
            n_xy = max(1.0, self.travels[x,y] + self.travels[y,x])
            error[x, y]  = pow(1/DELTA, 1/6) * np.sqrt(8 * self.t * log)
            error[x, y] += 2 * regret + 3 * pow(1/DELTA, 1/6)
            error[x, y] /= n_xy
            error[x, y] *= BIAS_ERROR_CUTOFF
        return error

    def empirical_structure(self):
        """ Construct the empirical structure (reward and kernels), 
        return in a paired of arrays. """
        if utils.KNOWN_KERNELS: # If kernels are known they are simply returned
            return self.rewards, self.kernels
        kernels = self.kernels
        for x, a in self.Z:
            N_xa = self.N[x, a]
            if N_xa: 
                kernels[x,a] = [ self.N[x,a,y] / N_xa for y in self.S ]
            else:
                kernels[x,a] = [1.0/self.n_states for _ in self.S]
        return self.rewards, self.kernels

    def change_episode(self):
        """ Update the episode: change policy, change counters """
        
        # Run EVI
        # self.config["delta"] = 1.0/self.t
        self.config["delta"] = DELTA
        evi  = EVI(self, self.config)

        # Run without truncation first
        u, g = evi.fixpoint(1e-3, u_init=self.saved_u, truncate=False)
        u_tr = evi.truncate(u)
        dist = sum(abs(u_i - u_tr_i) for u_i, u_tr_i in zip(u, u_tr))
        if dist > 1e-6: # Then truncation required, reboot
            # print("Starting over with truncation...")
            u, g = evi.fixpoint(1e-3, u_init=self.saved_u, truncate=True)

        # Update policy, reset counters
        self.gs.append(g)
        self.pi = evi.pi_selector(u)
        self.tk = self.t
        self.k += 1
        self.Nk = { z: self.N[z] for z in self.Z }
        self.saved_u = u

    def act(self, x):
        """ Pick an action according to current policy. Play action. """
        t = self.t
        if np.isscalar(self.A[x]):
            a=self.A[x]
        else:
            a = np.random.choice(self.A[x], p=self.pi[x])
        DT = self.N[x, a] >= max(2.0 * self.Nk[x, a], 1.0)
        VT = self.N[x, a] >= max((1.0 + np.sqrt(np.sqrt(t)/t)) * self.Nk[x, a], 1.0)
        if (DT and self.config["episodes"] == "DT") or (VT and self.config["episodes"] == "VT"): 
            self.change_episode()
            if np.isscalar(self.A[x]):
                a=self.A[x]
            else:
                a = np.random.choice(self.A[x], p=self.pi[x])
        return a
