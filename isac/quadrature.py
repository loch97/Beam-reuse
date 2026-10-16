"""Exact array-response and UE-loss caches, plus frozen-decision scoring."""

import numpy as np
from functools import lru_cache
from collections import OrderedDict
from . import model as V


@lru_cache(maxsize=512)
def array_response(shape, values):
    u = np.frombuffer(values, dtype=np.float64).reshape(shape)
    return np.exp(1j * np.pi * np.multiply.outer(u, V.N_IDX))


def cached_gain(W, u):
    """Original gain expression; cache its deterministic array-response matrix only."""
    W = np.atleast_2d(W)
    u = np.asarray(u, float)
    A = array_response(u.shape, u.tobytes())
    return np.abs(np.tensordot(W, A, axes=([1], [-1]))) ** 2 / V.M


class PointTruth:

    def __init__(self):
        self.ctx = {}

    def __call__(self, f, W, n_re, bg, clut):
        c = self.ctx

        def pd(bias, scale):
            return V.predictive_pd(
                W,
                c["u_hat"] + bias * c["sigma_u"],
                c["r_hat"],
                scale * c["sigma_u"],
                c["prior"],
                c["th"],
                n_re,
                bg,
                clut,
                s_r=scale * c["s_r"],
                c_ur=scale**2 * c["c_ur"],
            )[0]

        return {
            k: pd(b, s)
            for k, b, s in [("s1", 0, 1), ("s2", 0, 2), ("b+2", 2, 1), ("b-2", -2, 1)]
        }


class UELossCache:
    """Cache scalar quantiles for byte-identical beams and UE beam cells only."""

    def __init__(self):
        self.rows = OrderedDict()
        self.base = {}

    def __call__(self, W, g0, uq):
        scope = (float(g0), float(V.P["eps_ue"]), len(uq))
        keys = [(scope, w.tobytes()) for w in W]
        result = np.empty(len(W))
        missing = []
        for i, k in enumerate(keys):
            if k in self.rows:
                result[i] = self.rows[k]
                self.rows.move_to_end(k)
            else:
                missing.append(i)
        if missing:
            if scope not in self.base:
                self.base[scope] = V.db(V.gain(V.beam(g0), uq))[0]
            q = self.base[scope][None, :] - V.db(V.gain(W[missing], uq))
            values = np.quantile(q, 1 - V.P["eps_ue"], axis=1, method="inverted_cdf")
            for i, value in zip(missing, values):
                result[i] = value
                self.rows[keys[i]] = value
            while len(self.rows) > 16384:
                self.rows.popitem(last=False)
        return result
