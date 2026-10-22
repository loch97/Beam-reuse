"""Checks of the multi-scatterer target, the tracker interface and the beam front end."""

import json
import math
import unittest
import numpy as np
from isac import model as V, simulation_engine as LS, closed_loop as CL
from isac import target as TG, tracker as TR
from isac.detection import predictive_pd
from isac.frontend import BeamFrontEnd, curve_sigma
from isac.interfaces import GrantReport, SensingReport, RcsReport, TrackReport
from isac.rcs import RcsPopulation, RcsEstimator, RcsLook

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
        self.assertTrue(trk.update(SensingReport(0.5, True, z, R, snr_db=20.0)))
        self.assertLess(np.trace(trk.P[:2, :2]), np.trace(P_pred[:2, :2]))
        self.assertFalse(trk.update(SensingReport(0.9, False)))
        self.assertEqual(trk.misses, 1)

    def test_plan_respects_link_and_detection_constraints(self):
        rng = np.random.default_rng(5)
        sched = CL.Scheduler(self.rsu, self.bore, rng, n_grants=4)
        grants = sched.grants(0.4)
        trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), self.rsu, self.bore, (4.73, 1.88),
                           P0=np.diag([0.3, 0.3, 0.2, 0.2]))
        w = np.array([1.0, 2.0, 1.0])
        reports = [RcsReport(3.0), RcsReport(-1.0, 0.5, 0, "x", np.array([-3.0, 0.0, 3.0]), w)]
        for rcs in reports:
            for model in ["extended", "point"]:
                fe = BeamFrontEnd(self.rsu, self.bore, model)
                d = fe.plan(trk.report(), grants, rcs)
                self.assertEqual(d.mode, "reuse")
                g = grants[d.grant]
                self.assertGreaterEqual(d.pd_pred, fe.tau)
                self.assertGreaterEqual(d.mcs, max(g.mcs - g.max_mcs_drop, V.P["mcs_floor"]))
                thr = next(m for m in V.MCS if m[0] == d.mcs)[1]
                self.assertLessEqual(d.ue_loss_db, g.sinr_db - thr + 1e-9)
                self.assertEqual(d.n_re, g.n_prb * 12 * g.n_sym)
                if rcs.nodes_dbsm is not None:
                    np.testing.assert_allclose(d.info["rcs_weights"], w / w.sum())
        self.assertEqual(fe.plan(trk.report(), [], reports[0], t=0.4).mode, "dedicated")
        unweighted = RcsReport(0.0, nodes_dbsm=np.array([-1.0, 1.0]))
        np.testing.assert_allclose(fe.rcs_nodes(unweighted)[1], [0.5, 0.5])
        with self.assertRaises(ValueError):
            fe.rcs_nodes(RcsReport(0.0, nodes_dbsm=np.array([-1.0, 1.0]), weights=np.zeros(2)))

    def _track_state(self, pos_std):
        x = np.array([*self.row["target_xy_m"], 0.0, 8.0])
        P = np.diag([0.3 * pos_std, pos_std]) ** 2
        return x, P



ETA = -math.log(1e-3)


def static_population(mu=0.0, sd=3.0):
    return RcsPopulation({"car": mu}, {"car": sd}, {"car": 1.0}, sd, 0.0, 0.0, 1.2)


def single_cell_look(rng, t, sigma_true, unit_db, grid):
    """Exact single-cell Swerling-1 look: P_D curve and truncated statistic."""
    m_true = 1 + 10 ** ((sigma_true + unit_db) / 10)
    stat = rng.exponential(m_true)
    curve = np.exp(-ETA / (1 + 10 ** ((grid + unit_db) / 10)))
    if stat > ETA:
        return RcsLook(t, True, grid, curve, stat, ETA, unit_db), stat
    return RcsLook(t, False, grid, curve), None


def posterior_quantiles(est, qs):
    s, p = est.predictive(est.t)
    o = np.argsort(s)
    cdf = np.cumsum(p[o])
    return [s[o][min(np.searchsorted(cdf, q), len(s) - 1)] for q in qs]


class RcsEstimatorTests(unittest.TestCase):
    grid = np.arange(-40.0, 50.01, 1.5)

    def test_population_fit(self):
        recs = json.loads((ROOT / "data/reference/main_calibration.json").read_text())["records"]
        pop = RcsPopulation.fit(recs)
        train = [r for r in recs if r["site"] != V.EVAL_SITE]
        for c in ("car", "bus"):
            self.assertAlmostEqual(pop.mu[c], np.mean([r["sigma"] for r in train if r["cls"] == c]))
            self.assertGreater(pop.prior_sd[c], pop.tau)
        self.assertGreater(pop.omega, 0.5)
        self.assertTrue(0 <= pop.rho_frame < 1)
        self.assertAlmostEqual(pop.frame_dt, 1.2)
        self.assertLess(pop.tau, TG.FP["rcs_tau0_db"])  # little between-target spread: shrunk
        self.assertEqual(RcsEstimator(pop, "truck").cls, None)  # unknown class -> mixture

    def test_truncated_likelihood_removes_detection_bias(self):
        sigma_true, unit_db = 2.0, 4.0  # mean SNR 6 dB: P_D ~ 0.25
        for mode in ("event+truncated", "truncated"):
            rng = np.random.default_rng(11)
            est = RcsEstimator(static_population(0.0, 5.0), "car", likelihood=mode)
            naive = []
            for i in range(400):
                look, stat = single_cell_look(rng, 0.1 * i, sigma_true, unit_db, self.grid)
                est.update(look)
                if stat is not None:
                    naive.append(stat - 1)
            rep = est.report()
            self.assertLess(abs(rep.mean_dbsm - sigma_true), 0.6, mode)
        naive_db = 10 * math.log10(np.mean(naive)) - unit_db
        self.assertGreater(naive_db - sigma_true, 3.0)

    def test_credible_interval_coverage(self):
        rng = np.random.default_rng(7)
        hits, n = 0, 120
        for _ in range(n):
            sigma_true = 3.0 * rng.normal()
            est = RcsEstimator(static_population(0.0, 3.0), "car")
            for i in range(25):
                est.update(single_cell_look(rng, 0.1 * i, sigma_true, 6.0, self.grid)[0])
            lo, hi = posterior_quantiles(est, (0.1, 0.9))
            hits += lo <= sigma_true <= hi
        self.assertTrue(0.7 <= hits / n <= 0.92, hits / n)

    def test_ar_transition_and_report_nodes(self):
        pop = RcsPopulation({"car": 0.0}, {"car": 3.0}, {"car": 1.0}, 3.0, 2.0, 0.5, 1.2)
        est = RcsEstimator(pop, "car")
        T = est._transition(0.4)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-12)
        marg0 = est.p.sum(axis=0)
        np.testing.assert_allclose(marg0 @ T, marg0, atol=2e-3)
        rep = est.report(0.4)
        self.assertEqual(len(rep.nodes_dbsm), TG.FP["rcs_nodes"])
        self.assertAlmostEqual(rep.weights.sum(), 1.0)
        self.assertTrue(np.all(np.diff(rep.nodes_dbsm) > 0))
        self.assertAlmostEqual(rep.weights @ rep.nodes_dbsm, rep.mean_dbsm, places=9)
        self.assertAlmostEqual(rep.sd_dbsm, math.sqrt(9 + 4), delta=0.05)
        self.assertLess(rep.design_dbsm, rep.mean_dbsm)


class PredictiveFrontEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CL.configure()
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        rng = np.random.default_rng(5)
        cls.grants = CL.Scheduler(cls.rsu, cls.bore, rng, n_grants=4).grants(0.4)
        cls.trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), cls.rsu, cls.bore,
                               (4.73, 1.88), P0=np.diag([0.3, 0.3, 0.2, 0.2]))

    def test_predictive_pd_is_rcs_average(self):
        fe = BeamFrontEnd(self.rsu, self.bore, "extended")
        g = self.grants[0]
        pred = fe.prediction(self.trk.report(), g.t)
        W = np.array([V.beam(pred["u_hat"]), V.beam(V.GRID[g.k0])])
        n_re, bw, clut = g.n_prb * 12 * g.n_sym, g.n_prb * 12 * V.P["df"], fe.clutter(g)
        sig, w = np.array([-4.0, 0.0, 5.0]), np.array([0.2, 0.5, 0.3])
        single = np.stack([fe.detect(W, pred, n_re, bw, clut, np.array([s]), np.ones(1))
                           for s in sig])
        np.testing.assert_allclose(fe.detect(W, pred, n_re, bw, clut, sig, w), w @ single,
                                   atol=1e-12)
        self.assertTrue(np.all(np.diff(single, axis=0) >= -1e-12))  # P_D rises with RCS
        rep = RcsReport(-2.0, 1.0, 3, "x", sig, w)
        self.assertEqual(fe.rcs_nodes(rep)[0].tolist(), sig.tolist())
        fd = BeamFrontEnd(self.rsu, self.bore, "extended", rcs_mode="design")
        self.assertEqual(fd.rcs_nodes(rep)[0].tolist(), [-2.0])

    def test_lazy_plan_matches_exhaustive(self):
        grid = curve_sigma()
        nodes = grid[np.searchsorted(grid, [-4.0, -1.0, 2.0])]
        rcs = RcsReport(-1.0, 0.5, 0, "x", nodes, np.ones(3) / 3)
        for model in ("extended", "point"):
            lazy = BeamFrontEnd(self.rsu, self.bore, model).plan(self.trk.report(), self.grants, rcs)
            full = BeamFrontEnd(self.rsu, self.bore, model, lazy=False).plan(
                self.trk.report(), self.grants, rcs)
            self.assertEqual((lazy.mode, lazy.grant, lazy.tag), (full.mode, full.grant, full.tag))
            self.assertAlmostEqual(lazy.pd_pred, full.pd_pred, places=12)
            np.testing.assert_array_equal(lazy.weights, full.weights)
            self.assertLessEqual(lazy.info["n_evaluated"], full.info["n_evaluated"])
            self.assertEqual(len(lazy.pd_curve), len(grid))
            self.assertTrue(np.all(np.diff(lazy.pd_curve) >= -1e-12))

    def test_lazy_plan_matches_exhaustive_deep(self):
        """Weak target: the search must pass the cheapest level and enter the second stage."""
        rng = np.random.default_rng(101)
        grants = CL.Scheduler(self.rsu, self.bore, rng, n_grants=5).grants(0.4)
        y = -90 + 60 * rng.random()
        ps = 0.3 + 2 * rng.random()
        trk = TR.CVTracker(0, 0.0, (5.25, y), (0.0, 8.0), self.rsu, self.bore, (4.73, 1.88),
                           P0=np.diag([ps, ps, 0.5, 0.5]))
        mu = -14 + 14 * rng.random()
        grid = curve_sigma()
        nodes = grid[np.searchsorted(grid, mu + np.array([-4.0, -1.5, 0.0, 1.5, 4.0]))]
        rcs = RcsReport(mu - 2, mu, 0, "x", nodes, np.array([0.15, 0.2, 0.3, 0.2, 0.15]))
        fe = BeamFrontEnd(self.rsu, self.bore, "point")
        lazy = fe.plan(trk.report(), grants, rcs)
        full = BeamFrontEnd(self.rsu, self.bore, "point", lazy=False).plan(trk.report(), grants, rcs)
        self.assertEqual(lazy.mode, "reuse")
        self.assertGreater(lazy.cost_bits, 0.0)
        self.assertEqual(lazy.info["stage"], fe.stages[1])
        self.assertLess(lazy.info["n_evaluated"], full.info["n_evaluated"])
        self.assertEqual((lazy.mode, lazy.grant, lazy.tag, lazy.cost_bits),
                         (full.mode, full.grant, full.tag, full.cost_bits))
        self.assertAlmostEqual(lazy.pd_pred, full.pd_pred, places=12)
        np.testing.assert_array_equal(lazy.weights, full.weights)

    def test_screen_keeps_every_feasible_beam(self):
        fe = BeamFrontEnd(self.rsu, self.bore, "extended")
        g = self.grants[0]
        pred = fe.prediction(self.trk.report(), g.t)
        e = fe.evaluate_grant(g, pred)
        sig, w = np.array([-20.0, 0.0, 5.0]), np.array([0.1, 0.5, 0.4])
        full = fe.detect(e["W"], pred, e["n_re"], e["bw"], e["clutter"], sig, w)
        scr = fe.detect(e["W"], pred, e["n_re"], e["bw"], e["clutter"], sig, w, threshold=fe.tau)
        up = fe.pd_shifts(e["W"], pred, e["n_re"], e["bw"], None, sig)
        keep = full >= fe.tau
        self.assertTrue((scr < 0).any())
        self.assertTrue((keep & (up[0] < fe.tau)).any())  # the RCS spread matters
        np.testing.assert_allclose(scr[keep], full[keep], atol=1e-12, rtol=0)
        self.assertTrue(np.all(w @ up[:, scr < 0] < fe.tau))

    def test_rcs_look_designated_cell(self):
        fe = BeamFrontEnd(self.rsu, self.bore, "extended")
        tr = self.trk.report()
        d = fe.dedicated(tr, [], RcsReport(0.0), t=0.0)
        point = TrackReport(tr.track_id, tr.t, tr.x, np.zeros((4, 4)), tr.extent, tr.cls)
        h = TR.heading_of(tr.x, tr.P)[0]
        nodes = TG.pose_nodes(tr.x[:2], np.zeros((2, 2)), h, 0.0, tr.extent, self.rsu, self.bore, 0.0)
        snr = TG.cell_snr(d.weights[None], nodes, d.n_re, d.bandwidth)[0, 0, nodes.dom]
        rep = SensingReport(0.0, True, rcs_detected=True, rcs_stat=20.0, rcs_floor_db=3.0)
        look = fe.rcs_look(d, rep, point)
        np.testing.assert_allclose(look.unit_db, [10 * math.log10(snr) - 3.0], atol=1e-9)
        self.assertEqual((look.stat, look.detected), (20.0, True))
        self.assertAlmostEqual(look.eta, ETA)
        spread = fe.rcs_look(d, rep, tr)  # posterior pose spread -> mixture over pose nodes
        self.assertGreater(len(spread.unit_db), 1)
        self.assertAlmostEqual(spread.unit_w.sum(), 1.0)
        miss = fe.rcs_look(d, SensingReport(0.0, True, rcs_detected=False, rcs_stat=1.0,
                                            rcs_floor_db=3.0), tr)
        self.assertFalse(miss.detected)
        self.assertIsNone(miss.stat)
        np.testing.assert_allclose(miss.unit_db, spread.unit_db)  # misses carry unit nodes too
        self.assertIsNone(fe.rcs_look(d, SensingReport(0.0, False), tr))
        # point target: Gauss-Hermite in direction around the posterior position
        tr0 = TrackReport(tr.track_id, tr.t, tr.x, np.zeros((4, 4)), None, tr.cls)
        fp = BeamFrontEnd(self.rsu, self.bore, "point")
        d0 = fp.dedicated(tr0, [], RcsReport(0.0), t=0.0)
        pred = fp.prediction(tr0, 0.0)
        ref = V.rc_snr_db(TG.array_gain(d0.weights[None], [pred["u_hat"]])[0, 0], pred["u_hat"],
                          pred["r_hat"], 0.0, pred["theta"], d0.n_re) - 3.0
        look0 = fp.rcs_look(d0, rep, tr0)
        np.testing.assert_allclose(look0.unit_db, ref, atol=1e-9)

    def test_backend_looks_give_unbiased_rcs(self):
        """Looks generated by the closed-loop back end at a known pose, near P_d = 0.5."""
        rng = np.random.default_rng(23)
        sched = CL.Scheduler(self.rsu, self.bore, rng, n_grants=4)
        grants = sched.grants(0.0)
        trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), self.rsu, self.bore, (4.73, 1.88),
                           P0=np.diag([1e-4, 1e-4, 1e-4, 1e-4]))
        tr = trk.report()
        fe = BeamFrontEnd(self.rsu, self.bore, "extended")
        reuse = fe.plan(tr, grants, RcsReport(-1.0), t=0.0)
        self.assertEqual(reuse.mode, "reuse")
        cases = [(fe.dedicated(tr, [], RcsReport(0.0), t=0.0), None), (reuse, grants[reuse.grant])]
        back = CL.SensingBackend(self.rsu, self.bore, rng, 0.0)
        h = TR.heading_of(tr.x, tr.P)[0]
        for d, g in cases:
            sig_true = float(np.interp(0.5, d.pd_curve, d.pd_curve_sigma))
            truth = dict(xy=tr.x[:2], heading=h, extent=tr.extent, rcs=sig_true)
            for mode in ("event+truncated", "truncated"):
                est = RcsEstimator(static_population(sig_true + 3.0, 6.0), "car", likelihood=mode)
                hits = 0
                for _ in range(500):
                    rep = back.execute(d, truth, sched, g)
                    hits += rep.rcs_detected
                    est.update(fe.rcs_look(d, rep, tr))
                rep_ = est.report()
                self.assertLess(abs(rep_.mean_dbsm - sig_true), max(0.5, 2 * rep_.sd_dbsm),
                                (d.mode, mode, rep_.mean_dbsm, sig_true))
                self.assertAlmostEqual(hits / 500, 0.5, delta=0.08)


class RtExtendedTests(unittest.TestCase):
    def test_vehicle_extent_from_meshes(self):
        from isac.rt_extended import vehicle_extent

        meta, _, _ = LS.geometry()
        for row in meta["rows"]:
            np.testing.assert_allclose(vehicle_extent(row["category"]),
                                       row["truth_extent_LWH_m"][:2], atol=1e-5)

    def test_box_fits_rt_direct_paths_better_than_point(self):
        from types import SimpleNamespace
        from isac import rt_extended as RX, rt_scoring as S

        S.config()
        case = json.loads((ROOT / "data/rt/protocol.json").read_text())["cases"][0]
        data, meta = RX.load_case(ROOT / "data/example_rt", case, 16000000, 32)
        args = SimpleNamespace(prbs=[1, 32], mono_tol=0.01, fit_db=10.0, window_db=20.0)
        r = RX.validate_case(case, data, meta, args)
        for prb in ("1", "32"):
            box, point = r["prb"][prb]["box_direct"], r["prb"][prb]["point_direct"]
            self.assertLess(box["rms_db"], 1.0)
            self.assertLess(box["rms_db"], 0.5 * point["rms_db"])
        CL.configure()

    def test_extended_detector_matches_point_at_reference_focus(self):
        """With no pose spread, a beam on the dominant scatterer and the calibration bandwidth,
        the box gives the paper's effective RCS at the reference cell, hence the same P_D."""
        CL.configure()
        meta, rsu, bore = LS.geometry()
        n_re = 4 * 144
        for row in (meta["rows"][0], meta["rows"][8]):
            det = TG.ExtendedDetector()
            det.set_frame(row, rsu, bore)
            self.assertGreater(det.frame["delta_db"], 0.0)
            keys, xy, _, dom = V.vehicle_rcs(row, rsu, 0.0)
            u, r, th, _ = V.to_polar(xy, rsu, bore, V.P["rsu_h"], V.P["rc_h"])
            i = keys.index(dom)
            W = V.beam(u[i])[None]
            ext = det(W, u[i], r[i], 1e-9, -2.0, th[i], n_re, V.P["B"], s_r=1e-9, c_ur=0.0)[0]
            point = predictive_pd(W, u[i], r[i], 0.0, -2.0, th[i], n_re, s_r=0.0)[0]
            np.testing.assert_allclose(ext, point, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
