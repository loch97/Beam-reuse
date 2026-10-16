"""Ray-path target/clutter power and grant-averaged communication power."""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from . import model as V
from .detection import delay_power
from .calibration import array_power

FAMILIES = {"fine": 2, "sector": 11, "embedded": 13}


def config():
    V.config()
    V.P.update(
        deploy="corridor",
        sector_centres=9,
        unc_model="track",
        psi_deg=90.0,
        gap_k=19,
        ue_model="U1",
        sector_rule="union",
        eps_ue=0.01,
        inr=10.0,
        l_proc=3.0,
    )


def echo_power(data, W, nre, bg, focus):
    """Power relative to receiver noise plus the declared background INR."""
    mask = np.asarray(data["target"], bool) | np.asarray(data["served"], bool)
    a = np.asarray(data["a"], dtype=np.complex128)[mask]
    values = {
        k: np.asarray(data[k], dtype=np.float64)[mask]
        for k in ["ut", "ur", "r", "phi_t", "phi_r", "theta_t", "theta_r"]
    }
    ut, ur = (values["ut"], values["ur"])
    elem = 10 ** (
        (
            V.elem_dbi(values["phi_t"], values["theta_t"])
            + V.elem_dbi(values["phi_r"], values["theta_r"])
        )
        / 10
    )
    response = array_power(focus["u"] - ur, V.P["M_rx"]) * delay_power(
        focus["r"] - values["r"], bg
    )
    common = 10 ** (
        (V.P["p_ref_dbm"] - V.NOISE_RX + 10 * np.log10(V.M * V.P["M_rx"] * nre) - V.P["l_proc"])
        / 10
    ) / (1 + 10 ** (V.P["inr"] / 10))
    tx = W @ np.exp(1j * np.pi * np.outer(ut, V.N_IDX)).T / math.sqrt(V.M)
    per_path = common * abs(tx) ** 2 * (abs(a) ** 2 * elem * response)
    target = np.asarray(data["target"], bool)[mask]
    served = np.asarray(data["served"], bool)[mask] & ~target
    return (per_path[:, target].sum(axis=1), per_path[:, served].sum(axis=1))


def communication_power(data, W, bg, bore):
    """One-way channel, coherent across paths and averaged across grant tones.

    A path amplitude already contains one-way propagation loss. Carrier phase is
    applied exactly once. Array and element gain are applied exactly once.
    """
    a = np.asarray(data["comm_a"], complex)
    tau = np.asarray(data["comm_tau"], float)
    phi = np.asarray(data["comm_phi_t"], float)
    theta = np.asarray(data["comm_theta_t"], float)
    rel = (phi - bore + np.pi) % (2 * np.pi) - np.pi
    u = np.sin(theta) * np.sin(rel)
    elem = np.sqrt(10 ** (V.elem_dbi(rel, np.degrees(theta)) / 10))
    tones = (
        np.arange(int(round(bg / V.P["df"]))) - (int(round(bg / V.P["df"])) - 1) / 2
    ) * V.P["df"]
    fields = np.zeros((len(W), len(tones)), complex)
    for start in range(0, len(a), 2048):
        sl = slice(start, start + 2048)
        tx = W @ np.exp(1j * np.pi * np.outer(u[sl], V.N_IDX)).T / math.sqrt(V.M)
        phase = np.exp(-2j * np.pi * (V.P["fc"] + tones[None, :]) * tau[sl, None])
        fields += tx * (a[sl] * elem[sl])[None, :] @ phase
    return (abs(fields) ** 2).mean(axis=1)
