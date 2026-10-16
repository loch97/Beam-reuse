"""Reference focus and matched ULA receive response."""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from . import model as V
from .detection import delay_power


def array_power(delta, size=32):
    t = np.asarray(delta) / 2
    t = t - np.rint(t)
    return np.clip((np.sinc(size * t) / np.sinc(t)) ** 2, 0.0, 1.0)


def reference_focus(ut, ur, r, p, bandwidth):
    """Finite reference search: strongest 128 paths then two local refinements.

    This reference is generated before beam selection. Held-out RT scoring never
    adapts it to a tested beam or validation material.
    """
    ix = np.argsort(-p, kind="stable")[:128]

    def score(u0, r0):
        u0, r0 = (np.asarray(u0), np.asarray(r0))
        weights = array_power(u0[:, None] - ut) * array_power(u0[:, None] - ur, V.P["M_rx"])
        weights *= delay_power(r0[:, None] - r, bandwidth)
        return weights @ p

    u0, r0 = (ur[ix], r[ix])
    values = score(u0, r0)
    best = int(np.argmax(values))
    u, radius, value = (float(u0[best]), float(r0[best]), float(values[best]))
    for scale in [1.0, 0.25]:
        du = np.linspace(-1 / V.P["M_rx"], 1 / V.P["M_rx"], 9) * scale
        dr = np.linspace(-V.C / (4 * bandwidth), V.C / (4 * bandwidth), 9) * scale
        U, R = np.meshgrid(u + du, radius + dr, indexing="ij")
        keep = (abs(U.ravel()) < 1) & (R.ravel() > 0)
        U, R = (U.ravel()[keep], R.ravel()[keep])
        values = score(U, R)
        best = int(np.argmax(values))
        if values[best] > value:
            u, radius, value = (float(U[best]), float(R[best]), float(values[best]))
    sigma = value * (4 * np.pi) ** 3 * radius**4 / V.LAM**2
    return dict(u=u, r=radius, reference_power=value, sigma=10 * np.log10(sigma))
