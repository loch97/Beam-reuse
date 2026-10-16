"""Numerical regression against outcomes from the original paper implementation."""

from pathlib import Path
import json
import shutil
import tempfile
import unittest
import numpy as np
from isac import detection, model, scheduling, simulate, rt

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "outputs").mkdir(exist_ok=True)

    def test_finite_subcarrier_response(self):
        df = model.P["df"]
        for k in [12, 48, 384]:
            dr = np.array([0.0, 0.7, 10.0, model.C / (2 * df)])
            tones = (np.arange(k) - (k - 1) / 2) * df
            exact = np.exp(-4j * np.pi * tones[:, None] * dr / model.C).mean(axis=0)
            np.testing.assert_allclose(
                detection.delay_power(dr, k * df), abs(exact) ** 2, atol=1e-12
            )

    def test_uniform_grant_rank_probabilities(self):
        from math import comb

        for n, grants, compatible in [(115, 5, 17), (115, 10, 110), (12, 1, 3)]:
            void = (
                comb(n - compatible, grants) / comb(n, grants)
                if n - compatible >= grants
                else 0.0
            )
            self.assertAlmostEqual(
                scheduling.rank_weights(n, grants, compatible).sum() + void, 1.0
            )

    def test_simulation_against_original(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            original_out = simulate.OUT
            simulate.OUT = Path(tmp)
            for condition in ["broad_U1", "narrow_U0"]:
                result = simulate.run(condition, 2026100600, frames=1)
                expected = json.loads(
                    (ROOT / "data" / f"expected_{condition}.json").read_text()
                )
                self.assertEqual(result["metrics"], expected["metrics"])
                if "stress" in expected:
                    self.assertEqual(result["stress"], expected["stress"])
                    self.assertEqual(result["cascade_S10_T"], expected["cascade_S10_T"])
            simulate.OUT = original_out

    def test_rt_replay_against_original(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as tmp:
            original_out = rt.OUT
            rt.OUT = Path(tmp)
            (rt.OUT / "paths").mkdir()
            for path in (ROOT / "data/example_rt").glob("t00_f0*"):
                shutil.copy2(path, rt.OUT / "paths" / path.name)
            case = json.loads((ROOT / "data/rt/protocol.json").read_text())["cases"][0]
            rt.score_case((case, 16000000, 32))
            result = json.loads((rt.OUT / "scores/t00_f0_n16000000_s32.json").read_text())[
                "records"
            ]
            expected = json.loads((ROOT / "data/example_rt/expected_records.json").read_text())
            self.assertEqual(len(result), len(expected))
            for row, ref in zip(result, expected):
                self.assertEqual(
                    (row["selector"], row["family"], row["candidate"]),
                    (ref["selector"], ref["family"], ref["candidate"]),
                )
                for key in [
                    "prediction",
                    "point_ue_pd",
                    "entity_ue_pd",
                    "target_power",
                    "point_ue_power",
                    "entity_ue_power",
                    "rt_comm_sinr_db",
                ]:
                    np.testing.assert_allclose(row[key], ref[key], rtol=1e-12, atol=1e-12)
            rt.OUT = original_out


if __name__ == "__main__":
    unittest.main()
