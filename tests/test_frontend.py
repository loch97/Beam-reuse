"""Checks of the multi-scatterer target, the tracker interface and the beam front end."""

import json
import math
import unittest
import numpy as np
from isac import model as V, simulation_engine as LS, closed_loop as CL
from isac import target as TG, tracker as TR
from isac.detection import predictive_pd
from isac.frontend import BeamFrontEnd
from isac.interfaces import GrantReport, SensingReport, RcsReport

ROOT = TG.ROOT


def single_node(u, r, th, sig):
    return TG.PoseNodes(
        np.array([[u]]), np.array([[r]]), np.array([[th]]), np.ones(1), np.array([sig]), 0, ["d"]
    )


class FrontEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CL.configure()
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        cls.row = cls.meta["rows"][3]
        cls.W = np.array([V.beam(g) for g in np.linspace(-0.6, 0.0, 25)])

    def test_box_scatterers_match_vehicle_rcs(self):
        for row in self.meta["rows"]:
            ref = V.vehicle_rcs(row, self.rsu, 3.0)
            L, W = row["truth_extent_LWH_m"][:2]
            keys, xy, sig, dom, body = TG.box_scatterers(
                row["target_xy_m"], row["heading_rad"], L, W, self.rsu, 3.0
            )
            self.assertEqual(keys, ref[0])
            self.assertEqual(dom, ref[3])
            np.testing.assert_allclose(xy, ref[1], atol=1e-12)
            np.testing.assert_allclose(sig, ref[2], atol=1e-12)
            back = row["target_xy_m"] + body @ TG.rot(row["heading_rad"]).T
            np.testing.assert_allclose(back, xy, atol=1e-12)

    def test_single_scatterer_reduces_to_point_detector(self):
        u, r, th = 0.21, 40.0, 98.0
        nodes = single_node(u, r, th, -2.0)
        n_re, bw = 4 * 12 * 12, 4 * 12 * V.P["df"]
        point = predictive_pd(self.W, u, r, 0.0, -2.0, th, n_re, s_r=0.0)[0]
        ext = TG.extended_pd(self.W, nodes, n_re, bw)[0]
        np.testing.assert_allclose(ext, point, atol=1e-12)
        x, w = np.polynomial.legendre.leggauss(9)
        clut = (0.18 + x / 64, 35.0, 0.0, 99.0, w / 2)
        point = predictive_pd(self.W, u, r, 0.0, -2.0, th, n_re, bw, clut, s_r=0.0)[0]
        ext = TG.extended_pd(self.W, nodes, n_re, bw, clut)[0]
        np.testing.assert_allclose(ext, point, atol=1e-12)

    def test_extra_scatterers_only_add_detection(self):
        x, P = self._track_state(pos_std=1.5)
        nodes = TG.pose_nodes(x[:2], P, math.pi / 2, 0.02, (4.73, 1.88), self.rsu, self.bore, -1.0)
        alone = TG.PoseNodes(
            nodes.u, nodes.r, nodes.theta, nodes.w,
            np.where(np.arange(len(nodes.keys)) == nodes.dom, nodes.sigma_dbsm, -300.0),
            nodes.dom, nodes.keys,
        )
        bw = 8 * 12 * V.P["df"]
        full = TG.extended_pd(self.W, nodes, 8 * 144, bw)[0]
        dom = TG.extended_pd(self.W, alone, 8 * 144, bw, focus="dominant")[0]
        self.assertTrue(np.all(full >= dom - 1e-12))
        self.assertGreater(np.max(full - dom), 1e-3)

    def test_pose_quadrature_converged(self):
        x, P = self._track_state(pos_std=2.5)
        args = (x[:2], P, math.pi / 2, 0.05, (11.995, 2.642), self.rsu, self.bore, -1.0)
        coarse = TG.extended_pd(self.W, TG.pose_nodes(*args), 1152, 96 * 12 * V.P["df"])[0]
        saved = dict(TG.FP)
        try:
            TG.FP.update(ext_cells_per_sigma=8, ext_u_cell=TG.FP["ext_u_cell"] / 4,
                         ext_max_cells=400, ext_heading_nodes=7, ext_z_max=5.0)
            fine = TG.extended_pd(self.W, TG.pose_nodes(*args), 1152, 96 * 12 * V.P["df"])[0]
        finally:
            TG.FP.clear()
            TG.FP.update(saved)
        self.assertLess(np.max(np.abs(coarse - fine)), 5e-3)

    def test_steady_state_matches_paper_track_covariance(self):
        saved = dict(V.P)
        try:
            V.P.update(psi_deg=None, gap_k=9)
            xy = np.array(self.row["target_xy_m"]) + np.array([-1.0, -2.0])
            ref = V.track_covariance(xy, self.rsu, self.bore, self.row["heading_rad"])
            got = TR.steady_state_polar(xy, self.rsu, self.bore, self.row["heading_rad"],
                                        V.P["snr_prev_db"], V.P["prb_prev"], V.P["gap_k"])
        finally:
            V.P.clear()
            V.P.update(saved)
        np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-12)

    def test_tracker_update_and_prediction(self):
        trk = TR.CVTracker(0, 0.0, (5.0, -60.0), (0.0, 8.0), self.rsu, self.bore, (4.73, 1.88))
        P0 = np.trace(trk.P[:2, :2])
        x_pred, P_pred = trk.predicted(0.5)
        self.assertGreater(np.trace(P_pred[:2, :2]), P0)
        np.testing.assert_allclose(x_pred[:2], [5.0, -56.0])
        offs, dom = trk.scatterer_offsets()
        z = TR.polar(x_pred[:2] + offs[dom], self.rsu, self.bore)
        R = TR.crb_polar(20.0, 32 * 12 * V.P["df"])
        self.assertTrue(trk.update(SensingReport(0.5, True, z, R, 20.0, 0.0)))
        self.assertLess(np.trace(trk.P[:2, :2]), np.trace(P_pred[:2, :2]))
        self.assertFalse(trk.update(SensingReport(0.9, False)))
        self.assertEqual(trk.misses, 1)

    def test_rcs_estimator_lower_bound(self):
        rng = np.random.default_rng(3)
        est = TR.RcsEstimator(-1.0)
        self.assertEqual(est.report().source, "calibration")
        for p in rng.exponential(10 ** 0.4, 400):
            est.update(SensingReport(0.0, True, snr_db=10 * math.log10(p) + 20, unit_snr_db=20.0))
        rep = est.report()
        self.assertEqual(rep.source, "tracker")
        self.assertLess(rep.design_dbsm, rep.mean_dbsm)
        self.assertLess(abs(rep.mean_dbsm - 4.0), 2.5)

    def test_plan_respects_link_and_detection_constraints(self):
        rng = np.random.default_rng(5)
        sched = CL.Scheduler(self.rsu, self.bore, rng, n_grants=4)
        grants = sched.grants(0.4)
        trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), self.rsu, self.bore, (4.73, 1.88),
                           P0=np.diag([0.3, 0.3, 0.2, 0.2]))
        rcs = RcsReport(3.0)
        for model in ["extended", "point"]:
            fe = BeamFrontEnd(self.rsu, self.bore, model)
            d = fe.plan(trk.report(), grants, rcs)
            if d.mode == "reuse":
                g = grants[d.grant]
                self.assertGreaterEqual(d.pd_pred, fe.tau)
                self.assertGreaterEqual(d.mcs, max(g.mcs - g.max_mcs_drop, V.P["mcs_floor"]))
                thr = next(m for m in V.MCS if m[0] == d.mcs)[1]
                self.assertLessEqual(d.ue_loss_db, g.sinr_db - thr + 1e-9)
                self.assertEqual(d.n_re, g.n_prb * 12 * g.n_sym)
            else:
                self.assertEqual(d.mode, "dedicated")
        self.assertEqual(fe.plan(trk.report(), [], rcs, t=0.4).mode, "dedicated")

    def _track_state(self, pos_std):
        x = np.array([*self.row["target_xy_m"], 0.0, 8.0])
        P = np.diag([0.3 * pos_std, pos_std]) ** 2
        return x, P


if __name__ == "__main__":
    unittest.main()
