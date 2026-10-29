"""Checks of the multi-scatterer target, the tracker interface and the beam front end."""

import json
import math
import unittest
import numpy as np
from isac import model as V, simulation_engine as LS, closed_loop as CL
from isac import target as TG, tracker as TR
from isac.detection import predictive_pd
from isac.frontend import BeamFrontEnd, curve_sigma
from isac.interfaces import GrantReport, SensingReport, RcsReport, TrackReport, BeamDecision
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



class SiteInterfaceTests(unittest.TestCase):
    """Injected site/model configuration, quality requirements and robustness (P1-P6)."""

    @classmethod
    def setUpClass(cls):
        from isac.site import SiteConfig

        CL.configure()
        cls.SiteConfig = SiteConfig
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        rng = np.random.default_rng(5)
        cls.grants = CL.Scheduler(cls.rsu, cls.bore, rng, n_grants=4).grants(0.4)
        cls.trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), cls.rsu, cls.bore,
                               (4.73, 1.88), P0=np.diag([0.3, 0.3, 0.2, 0.2]))
        cls.rcs = RcsReport(-1.0, 0.5, 0, "x", np.array([-3.0, 0.0, 3.0]), np.ones(3) / 3)

    def _same(self, a, b):
        self.assertEqual((a.mode, a.grant, a.tag), (b.mode, b.grant, b.tag))
        self.assertAlmostEqual(a.pd_pred, b.pd_pred, places=12)

    def test_default_and_paper_valued_site_reproduce_decision(self):
        ref = BeamFrontEnd(self.rsu, self.bore).plan(self.trk.report(), self.grants, self.rcs)
        d0 = BeamFrontEnd(site=self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore)).plan(
            self.trk.report(), self.grants, self.rcs)
        self._same(ref, d0)
        P0, noise = dict(V.P), V.NOISE_RX
        paper = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore, rsu_h=V.P["rsu_h"],
                                rc_h=V.P["rc_h"], M=32, M_rx=32, fc=V.P["fc"],
                                p_ref_dbm=V.P["p_ref_dbm"], nf_rx=V.P["nf_rx"],
                                noise_bw=V.P["B"], l_proc=V.P["l_proc"], inr=V.P["inr"],
                                pfa=V.P["pfa"])
        self.assertFalse(paper.is_default())
        d1 = BeamFrontEnd(site=paper).plan(self.trk.report(), self.grants, self.rcs)
        self._same(ref, d1)
        self.assertEqual(V.P, P0)  # globals restored
        self.assertEqual(V.NOISE_RX, noise)
        self.assertEqual(V.to_polar.__module__, "isac.model")

    def test_radio_overrides_change_snr_and_restore(self):
        quiet = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore, nf_rx=V.P["nf_rx"] + 10)
        tr = self.trk.report()
        base = BeamFrontEnd(self.rsu, self.bore).dedicated(tr, [], RcsReport(-1.0), t=0.0)
        noisy = BeamFrontEnd(site=quiet).dedicated(tr, [], RcsReport(-1.0), t=0.0)
        self.assertAlmostEqual(base.pred_cell_snr_db - noisy.pred_cell_snr_db, 10.0, places=6)
        iso = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore, element_pattern=False)
        d = BeamFrontEnd(site=iso).dedicated(tr, [], RcsReport(-1.0), t=0.0)
        self.assertLess(d.pred_cell_snr_db, base.pred_cell_snr_db)  # loses 2 x e_max - pattern
        self.assertGreater(float(np.asarray(V.elem_dbi(0.0, 90.0))), 0.0)

    def test_cone_angle(self):
        from isac.site import cone_to_polar

        xy = np.array([[5.0, -40.0], [12.0, -20.0]])
        u, r, th, _ = cone_to_polar(xy, self.rsu, self.bore, 6.0, 1.5)
        us = V.to_polar(xy, self.rsu, self.bore, 6.0, 1.5)[0]
        d2 = np.hypot(*(xy - self.rsu).T)
        np.testing.assert_allclose(u, us * d2 / r)
        # a horizontal ULA's phase follows the cone angle: beam(u_cone) reaches the peak gain
        phase = np.exp(1j * np.pi * np.outer(u, V.N_IDX))
        for k in range(2):
            self.assertAlmostEqual(abs(phase[k] @ V.beam(u[k])) ** 2 / V.M, 1.0, places=12)
            self.assertLess(abs(phase[k] @ V.beam(us[k])) ** 2 / V.M, 1.0)
        site = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore, cone_angle=True)
        with site.active():
            J = TR.polar_jacobian(xy[0], self.rsu, self.bore)
            h = 1e-3
            num = (TR.polar(xy[0] + [h, 0], self.rsu, self.bore)
                   - TR.polar(xy[0] - [h, 0], self.rsu, self.bore)) / (2 * h)
            np.testing.assert_allclose(J[:, 0], num, rtol=1e-5)
        d = BeamFrontEnd(site=site).plan(self.trk.report(), self.grants, self.rcs)
        self.assertGreaterEqual(d.pd_pred, 0.9)

    def test_snr_calibration(self):
        tr = self.trk.report()
        off = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore, snr_offset_db=6.0)
        a = BeamFrontEnd(site=off).dedicated(tr, [], RcsReport(-1.0), t=0.0)
        b = BeamFrontEnd(self.rsu, self.bore).dedicated(tr, [], RcsReport(-7.0), t=0.0)
        self.assertAlmostEqual(a.pd_pred, b.pd_pred, places=12)
        spread = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore, snr_offset_db=6.0,
                                 snr_sigma_db=4.0)
        fe = BeamFrontEnd(site=spread)
        sig, w = fe.rcs_nodes(self.rcs)
        self.assertEqual(len(sig), 3 * spread.sigma_hermite_nodes)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertAlmostEqual(w @ sig, 0.0 - 6.0, places=9)
        self.assertAlmostEqual(math.sqrt(w @ (sig + 6.0) ** 2), math.sqrt(6.0 + 16.0), places=6)
        # evidence for the RCS estimator stays in calibrated RCS units
        tr0 = TrackReport(tr.track_id, tr.t, tr.x, np.zeros((4, 4)), tr.extent, tr.cls)
        rep = SensingReport(0.0, True, rcs_detected=True, rcs_stat=20.0, rcs_floor_db=0.0)
        u_off = BeamFrontEnd(site=off).rcs_look(a, rep, tr0).unit_db
        u_ref = BeamFrontEnd(self.rsu, self.bore).rcs_look(a, rep, tr0).unit_db
        np.testing.assert_allclose(u_off, u_ref - 6.0)

    def test_quality_requirement_and_table(self):
        from isac.interfaces import SensingRequirement

        fe = BeamFrontEnd(self.rsu, self.bore)
        tr = self.trk.report()
        free = fe.plan(tr, self.grants, self.rcs, return_table=True)
        tab = free.candidates
        self.assertEqual(len(tab["pd"]), free.n_candidates)
        self.assertTrue(np.isfinite(free.pred_cell_snr_db))
        hi = float(np.nanmax(tab["pred_cell_snr_db"][tab["pd"] >= fe.tau]))
        need = 0.5 * (free.pred_cell_snr_db + hi) if hi > free.pred_cell_snr_db + 1e-6 else hi
        req = SensingRequirement(min_cell_snr_db=need)
        d = fe.plan(tr, self.grants, self.rcs, requirement=req)
        if d.mode == "reuse":
            self.assertTrue(d.requirement_met)
            self.assertGreaterEqual(d.pred_cell_snr_db, need - 1e-9)
            self.assertGreaterEqual(d.cost_bits, free.cost_bits)
            ok = (tab["pd"] >= fe.tau) & (tab["pred_cell_snr_db"] >= need)
            self.assertAlmostEqual(d.cost_bits, float(tab["cost_bits"][ok].min()))
        impossible = SensingRequirement(min_cell_snr_db=200.0)
        d2 = fe.plan(TrackReport(tr.track_id, tr.t, tr.x, tr.P, tr.extent, tr.cls,
                                 requirement=impossible), self.grants, self.rcs)
        self.assertEqual(d2.mode, "dedicated")
        self.assertFalse(d2.requirement_met)
        cells = SensingRequirement(quality_snr_db=10.0, min_cells=1.5)
        d3 = fe.plan(tr, self.grants, self.rcs, requirement=cells)
        if d3.mode == "reuse":
            self.assertGreaterEqual(d3.exp_quality_cells, 1.5)

    def test_rsu_inside_box_falls_back_to_point(self):
        fe = BeamFrontEnd(self.rsu, self.bore)
        ghost = TrackReport(1, 0.4, np.array([*self.rsu, 0.0, 0.5]), np.eye(4) * 0.5,
                            (4.73, 1.88), "car")
        keys, xy, sig, dom, body = TG.box_scatterers(self.rsu, 0.3, 4.73, 1.88, self.rsu, 0.0)
        self.assertIsNone(dom)
        self.assertEqual(len(keys), 0)
        pred = fe.prediction(ghost, 0.4)
        self.assertTrue(pred["degenerate_box"])
        self.assertNotIn("nodes", pred)
        d = fe.dedicated(ghost, [], RcsReport(0.0), t=0.4)
        self.assertTrue(d.info["degenerate_box"])
        rep = SensingReport(0.4, True, rcs_detected=True, rcs_stat=30.0, rcs_floor_db=0.0)
        self.assertTrue(np.all(np.isfinite(fe.rcs_look(d, rep, ghost).unit_db)))
        self.assertTrue(np.isfinite(d.pd_pred) and np.isfinite(d.sigma_u))
        trk = TR.CVTracker(0, 0.0, self.rsu, (0.0, 0.5), self.rsu, self.bore, (4.73, 1.88))
        self.assertEqual(trk.scatterer_offsets()[0].shape, (1, 2))

    def test_class_priors_and_arbitrary_road(self):
        from isac.site import road_points, ue_link_site

        pop = RcsPopulation.from_priors({"car": (-10.0, 3.0), "bus": (2.0, 3.0)}, omega=1.5)
        est = RcsEstimator(pop, "car")
        self.assertAlmostEqual(est.report().mean_dbsm, -10.0, delta=0.1)
        self.assertAlmostEqual(RcsEstimator(pop, None).report().mean_dbsm, -4.0, delta=0.1)
        xy, kind = road_points((20.0, 10.0), math.radians(35), [-3.5, 0.0, 6.0], (-50, 50), 10.0,
                               ["vehicle", "vehicle", "sidewalk"])
        self.assertEqual(len(xy), 33)
        along = (xy - [20.0, 10.0]) @ [math.cos(math.radians(35)), math.sin(math.radians(35))]
        self.assertAlmostEqual(along.min(), -50.0)
        site = self.SiteConfig(rsu_xy=(0.0, 0.0), boresight_rad=math.radians(30), rsu_h=6.0,
                               cone_angle=True)
        with site.active():
            rng = np.random.default_rng(1)
            sinr, k0, u, r, th = ue_link_site(xy, kind, np.zeros(2), math.radians(30), rng)
            sinr_i = ue_link_site(xy, kind, np.zeros(2), math.radians(30),
                                  np.random.default_rng(1), interference_dbm=-60.0)[0]
            G = V.gain(np.array([V.beam(g) for g in V.GRID]), u)
            np.testing.assert_array_equal(k0, np.argmax(G, axis=0))
        self.assertTrue(np.all(sinr_i < sinr))
        self.assertTrue(np.all(np.isfinite(sinr)))



class RoundTwoTests(unittest.TestCase):
    """Resolution bandwidth, quality calibration, fallbacks, screening and fixed-cell evidence."""

    @classmethod
    def setUpClass(cls):
        from isac.site import SiteConfig

        CL.configure()
        cls.SiteConfig = SiteConfig
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        rng = np.random.default_rng(5)
        cls.grants = CL.Scheduler(cls.rsu, cls.bore, rng, n_grants=4).grants(0.4)
        cls.trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), cls.rsu, cls.bore,
                               (4.73, 1.88), P0=np.diag([0.3, 0.3, 0.2, 0.2]))
        cls.rcs = RcsReport(-1.0, 0.5, 0, "x", np.array([-3.0, 0.0, 3.0]), np.ones(3) / 3)

    def test_resolution_bandwidth_decoupled_from_res(self):
        tr = self.trk.report()
        base = BeamFrontEnd(self.rsu, self.bore).plan(tr, self.grants, self.rcs)
        self.assertEqual(base.resolution_bw, base.bandwidth)
        wide = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore,
                               sensing_bandwidth_hz=190.08e6)
        fe = BeamFrontEnd(site=wide)
        d = fe.plan(tr, self.grants, self.rcs)
        self.assertEqual(d.resolution_bw, 190.08e6)
        self.assertEqual(d.bandwidth, self.grants[d.grant].n_prb * 12 * V.P["df"])
        self.assertEqual(d.n_re, self.grants[d.grant].n_prb * 12 * self.grants[d.grant].n_sym)
        # same beam, same REs: resolving the scatterers in range lowers the strongest cell
        g = self.grants[base.grant]
        pred = fe.prediction(tr, g.t)
        e = fe.evaluate_grant(g, pred)
        self.assertEqual(e["bw"], 190.08e6)
        narrow = BeamFrontEnd(self.rsu, self.bore)
        snr_w = fe.quality(base.weights, pred, e["n_re"], 190.08e6, None, np.zeros(1), np.ones(1))[0]
        snr_n = narrow.quality(base.weights, pred, e["n_re"], e["grant_bw"], None, np.zeros(1),
                               np.ones(1))[0]
        self.assertLess(snr_w[0], snr_n[0])
        ded = fe.dedicated(tr, [], RcsReport(-1.0), t=0.0)
        self.assertEqual(ded.resolution_bw, 190.08e6)

    def test_logistic_quality_requirement(self):
        from isac.interfaces import SensingRequirement

        req = SensingRequirement(quality_logistic=(-8.0, 0.25), min_quality_prob=0.5)
        self.assertTrue(req.active())
        self.assertAlmostEqual(float(req.quality_prob(32.0)), 0.5)
        fe = BeamFrontEnd(self.rsu, self.bore, on_unmet="reuse_pd")
        d = fe.plan(self.trk.report(), self.grants, self.rcs, requirement=req)
        self.assertAlmostEqual(d.quality_prob, float(req.quality_prob(d.pred_cell_snr_db)))
        if d.requirement_met:
            self.assertGreaterEqual(d.quality_prob, 0.5)

    def test_on_unmet_fallbacks(self):
        from isac.interfaces import SensingRequirement

        tr = self.trk.report()
        impossible = SensingRequirement(min_cell_snr_db=200.0)
        fe = BeamFrontEnd(self.rsu, self.bore)
        free = fe.plan(tr, self.grants, self.rcs, return_table=True)
        d0 = fe.plan(tr, self.grants, self.rcs, requirement=impossible)
        self.assertEqual((d0.mode, d0.info["fallback"], d0.requirement_met),
                         ("dedicated", "dedicated", False))
        d1 = fe.plan(tr, self.grants, self.rcs, requirement=impossible, on_unmet="reuse_pd")
        self.assertEqual((d1.mode, d1.grant, d1.tag), (free.mode, free.grant, free.tag))
        self.assertEqual((d1.info["fallback"], d1.requirement_met), ("reuse_pd", False))
        d2 = fe.plan(tr, self.grants, self.rcs, requirement=impossible, on_unmet="best_quality")
        self.assertEqual((d2.mode, d2.info["fallback"], d2.requirement_met),
                         ("reuse", "best_quality", False))
        tab = free.candidates
        ok = np.round(tab["pd"], 12) >= fe.tau
        self.assertAlmostEqual(d2.pred_cell_snr_db, float(tab["pred_cell_snr_db"][ok].max()),
                               places=9)
        self.assertGreaterEqual(d2.pd_pred, fe.tau)
        self.assertEqual(free.info["fallback"], "none")

    def test_upper_bound_screen_is_safe(self):
        fe = BeamFrontEnd(self.rsu, self.bore)
        tr = self.trk.report()
        weak = RcsReport(-14.0, -13.0, 0, "x", np.array([-17.0, -13.0, -9.0]), np.ones(3) / 3)
        d = fe.plan(tr, self.grants, weak, return_table=True)
        tab = d.candidates
        screened = tab["pd"] < 0
        self.assertTrue(screened.any())
        for i, g in enumerate(self.grants):
            sel = screened & (tab["grant"] == i)
            if not sel.any():
                continue
            pred = fe.prediction(tr, g.t)
            e = fe.evaluate_grant(g, pred)
            sig, w = fe.rcs_nodes(weak)
            exact = fe.detect(tab["weights"][sel], pred, e["n_re"], e["bw"], e["clutter"], sig, w)
            self.assertTrue(np.all(exact < fe.tau))
            bound = TG.pd_upper_bound(tab["weights"][sel], pred["nodes"], e["n_re"], sig, w)
            clean = w @ TG.extended_pd_shifts(tab["weights"][sel], pred["nodes"], e["n_re"],
                                              e["bw"], None, shifts_db=sig)[0]
            self.assertTrue(np.all(bound >= clean - 1e-12))
        # the bound must actually bite for a weak target (not a vacuous 1.0)
        g = self.grants[0]
        pred = fe.prediction(tr, g.t)
        e = fe.evaluate_grant(g, pred)
        W = e["W"][e["ok"]]
        bound = TG.pd_upper_bound(W, pred["nodes"], e["n_re"], np.array([-38.0]), np.ones(1))
        self.assertGreater(np.mean(bound < fe.tau), 0.5)

    def test_predicted_evidence_cell(self):
        wide = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore,
                               sensing_bandwidth_hz=190.08e6)
        fs = BeamFrontEnd(site=wide)
        fp = BeamFrontEnd(site=wide, evidence_cell="predicted")
        tr = self.trk.report()
        exact = TrackReport(tr.track_id, tr.t, tr.x, np.zeros((4, 4)), tr.extent, tr.cls)
        d = fp.dedicated(exact, [], RcsReport(0.0), t=0.0)
        rep = SensingReport(0.0, True, rcs_detected=True, rcs_stat=50.0, rcs_floor_db=0.0)
        np.testing.assert_allclose(fp.rcs_look(d, rep, exact).unit_db,
                                   fs.rcs_look(d, rep, exact).unit_db, atol=1e-9)
        moved = TrackReport(tr.track_id, tr.t, tr.x + np.array([0.0, 1.0, 0, 0]),
                            np.diag([0.05, 0.05, 0.1, 0.1]), tr.extent, tr.cls)
        def mean_unit(look):
            return 10 * math.log10(look.unit_w @ 10 ** (look.unit_db / 10))

        self.assertLess(mean_unit(fp.rcs_look(d, rep, moved)),
                        mean_unit(fs.rcs_look(d, rep, moved)) - 1.0)
        dp = fp.dedicated(tr, [], RcsReport(0.0), t=0.0)
        ds = fs.dedicated(tr, [], RcsReport(0.0), t=0.0)
        self.assertTrue(np.all(dp.pd_curve <= ds.pd_curve + 1e-9))  # fixed cell loses SNR
        self.assertTrue(np.all(np.diff(dp.pd_curve) >= -1e-12))



class MultiGrantTests(unittest.TestCase):
    """R5: one sensing look integrated over several grants of a frame."""

    @classmethod
    def setUpClass(cls):
        from isac.site import SiteConfig

        CL.configure()
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        rng = np.random.default_rng(5)
        cls.grants = CL.Scheduler(cls.rsu, cls.bore, rng, n_grants=4).grants(0.4)
        cls.trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), cls.rsu, cls.bore,
                               (4.73, 1.88), P0=np.diag([0.3, 0.3, 0.2, 0.2]))
        cls.rcs = RcsReport(-1.0, 0.5, 0, "x", np.array([-3.0, 0.0, 3.0]), np.ones(3) / 3)
        cls.site = SiteConfig(rsu_xy=cls.rsu, boresight_rad=cls.bore, sensing_bandwidth_hz=46.08e6)

    def test_single_grant_cases_equal_plan(self):
        from isac.interfaces import SensingRequirement

        fe = BeamFrontEnd(site=self.site, on_unmet="reuse_pd")
        tr = self.trk.report()
        p = fe.plan(tr, self.grants, self.rcs)
        m = fe.plan_multi(tr, self.grants, self.rcs, max_grants=4)
        self.assertEqual((m.mode, m.grant, m.tag), (p.mode, p.grant, p.tag))
        self.assertEqual(len(m.parts), 1)
        self.assertFalse(m.info["multi"])
        req = SensingRequirement(min_cell_snr_db=200.0)
        one = fe.plan_multi(tr, self.grants, self.rcs, requirement=req, max_grants=1)
        ref = fe.plan(tr, self.grants, self.rcs, requirement=req)
        self.assertEqual((one.mode, one.grant, one.tag, one.info["fallback"]),
                         (ref.mode, ref.grant, ref.tag, "reuse_pd"))
        ded = fe.plan_multi(tr, self.grants, self.rcs, requirement=req, max_grants=4,
                            on_unmet="dedicated")
        self.assertEqual((ded.mode, ded.requirement_met), ("dedicated", False))

    def test_combined_grants_meet_requirement(self):
        from isac.interfaces import SensingRequirement

        fe = BeamFrontEnd(site=self.site, on_unmet="reuse_pd")
        tr = self.trk.report()
        tab = fe.plan(tr, self.grants, self.rcs, return_table=True).candidates
        best = float(np.nanmax(tab["pred_cell_snr_db"][np.round(tab["pd"], 12) >= fe.tau]))
        need = best + 0.05
        d = fe.plan_multi(tr, self.grants, self.rcs,
                          requirement=SensingRequirement(min_cell_snr_db=need), max_grants=4)
        self.assertEqual(d.mode, "reuse_multi")
        self.assertTrue(d.requirement_met)
        self.assertEqual(d.info["fallback"], "none")
        self.assertGreaterEqual(d.pred_cell_snr_db, need)
        self.assertGreaterEqual(d.pd_pred, fe.tau)
        grants = [p["grant"] for p in d.parts]
        self.assertGreaterEqual(len(grants), 2)
        self.assertLessEqual(len(grants), 4)
        self.assertEqual(len(set(grants)), len(grants))
        self.assertIsNone(d.weights)
        self.assertAlmostEqual(d.cost_bits, sum(p["cost_bits"] for p in d.parts))
        self.assertEqual(d.n_re, sum(p["n_re"] for p in d.parts))
        self.assertEqual(d.resolution_bw, 46.08e6)
        for prt in d.parts:
            g = self.grants[prt["grant"]]
            self.assertGreaterEqual(prt["mcs"], max(g.mcs - g.max_mcs_drop, V.P["mcs_floor"]))
            self.assertEqual(prt["n_re"], g.n_prb * 12 * g.n_sym)
        # RCS evidence of a multi-grant look sums the parts' unit SNRs
        rep = SensingReport(0.4, True, rcs_detected=True, rcs_stat=50.0, rcs_floor_db=0.0)
        exact = TrackReport(tr.track_id, tr.t, tr.x, np.zeros((4, 4)), tr.extent, tr.cls)
        lm = fe.rcs_look(d, rep, exact)
        for prt in d.parts:
            dp = BeamDecision("reuse", d.t, weights=prt["weights"], n_re=prt["n_re"],
                              bandwidth=prt["bandwidth"], resolution_bw=d.resolution_bw,
                              pd_curve_sigma=d.pd_curve_sigma, pd_curve=d.pd_curve)
            self.assertGreater(lm.unit_db.max(), fe.rcs_look(dp, rep, exact).unit_db.max())
        self.assertTrue(np.all(np.diff(d.pd_curve) >= -1e-12))
        self.assertTrue(np.all(np.diff(d.info["steps_snr_db"]) > 0))

    def test_never_worse_than_best_single_grant(self):
        """R5-fix: additions must not lower the combined SNR below the best single grant."""
        from isac.interfaces import SensingRequirement

        fe = BeamFrontEnd(site=self.site, on_unmet="best_quality")
        tr = self.trk.report()
        tab = fe.plan(tr, self.grants, self.rcs, return_table=True).candidates
        best = float(np.nanmax(tab["pred_cell_snr_db"][np.round(tab["pd"], 12) >= fe.tau]))
        for k in (2, 3, 4):
            d = fe.plan_multi(tr, self.grants, self.rcs,
                              requirement=SensingRequirement(min_cell_snr_db=200.0), max_grants=k)
            self.assertFalse(d.requirement_met)
            self.assertEqual(d.info["fallback"], "best_quality")
            self.assertGreaterEqual(d.pred_cell_snr_db, best - 1e-9)
            self.assertLessEqual(len(d.parts), k)
            self.assertTrue(np.all(np.diff(d.info["steps_snr_db"]) > 0))



class RoundThreeTests(unittest.TestCase):
    """Q1 decision-event P_D versus SNR offset; Q2 frame-level grant assignment."""

    @classmethod
    def setUpClass(cls):
        CL.configure()
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        rng = np.random.default_rng(5)
        cls.grants = CL.Scheduler(cls.rsu, cls.bore, rng, n_grants=4).grants(0.4)
        cls.trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), cls.rsu, cls.bore,
                               (4.73, 1.88), P0=np.diag([0.3, 0.3, 0.2, 0.2]))
        cls.weak = RcsReport(-27.0, -27.0, 0, "x", np.array([-29.0, -27.0, -25.0]), np.ones(3) / 3)

    def test_offset_curve_is_decision_event(self):
        tr = self.trk.report()
        for model in ("extended", "point"):
            fe = BeamFrontEnd(self.rsu, self.bore, model)
            for d in (fe.plan(tr, self.grants, self.weak), fe.dedicated(tr, [], self.weak, t=0.0)):
                off, pd = d.pd_offset_curve
                self.assertAlmostEqual(float(pd[np.argmin(abs(off))]), d.pd_pred, places=9)
                self.assertTrue(np.all(np.diff(pd) >= -1e-12))
                self.assertLess(pd[0], 0.5)  # spans the transition for a weak target
                self.assertGreater(pd[-1], 0.99)

    def test_frame_assignment_never_double_books(self):
        from isac.interfaces import SensingRequirement

        a = self.trk.report()
        near = TR.CVTracker(1, 0.0, (5.25, -60.5), (0.0, 8.0), self.rsu, self.bore, (4.73, 1.88),
                            P0=np.diag([0.3, 0.3, 0.2, 0.2])).report()
        fe = BeamFrontEnd(self.rsu, self.bore, on_unmet="reuse_pd")
        d0, d1 = fe.plan_frame([a, near], self.grants)
        self.assertEqual(d1.info["shared_with"], 0)
        self.assertEqual((d1.grant, d1.cost_bits), (d0.grant, 0.0))
        np.testing.assert_array_equal(d1.weights, d0.weights)
        self.assertGreaterEqual(d1.pd_pred, fe.tau)
        # a requirement the shared beam cannot meet forces another grant, never grant 0 re-steered
        hard = SensingRequirement(min_cell_snr_db=200.0)
        e0, e1 = fe.plan_frame([a, near], self.grants, requirements=[None, hard])
        self.assertIsNone(e1.info["shared_with"])
        if e1.mode == "reuse":
            self.assertNotEqual(e1.grant, e0.grant)
        for ds in ((d0, d1), (e0, e1)):
            books = {}
            for d in ds:
                if d.mode == "reuse":
                    if d.grant in books:
                        np.testing.assert_array_equal(books[d.grant], d.weights)
                    books[d.grant] = d.weights



class CoverageObjectiveTests(unittest.TestCase):
    """Round 4: coverage diagnostics, selection objectives, covering dedicated width, extent prior."""

    @classmethod
    def setUpClass(cls):
        from isac.site import SiteConfig

        CL.configure()
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        rng = np.random.default_rng(5)
        cls.grants = CL.Scheduler(cls.rsu, cls.bore, rng, n_grants=4).grants(0.4)
        cls.trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), cls.rsu, cls.bore,
                               (4.73, 1.88), P0=np.diag([0.8, 0.8, 0.2, 0.2]))
        cls.rcs = RcsReport(-1.0, 0.5, 0, "x", np.array([-3.0, 0.0, 3.0]), np.ones(3) / 3)
        cls.site = SiteConfig(rsu_xy=cls.rsu, boresight_rad=cls.bore, sensing_bandwidth_hz=46.08e6)

    def _check_diag(self, d):
        self.assertTrue(np.isfinite(d.coverage_db) and d.coverage_db <= 1e-9)
        self.assertTrue(np.isfinite(d.coverage_gain_dbi))
        self.assertTrue(0.0 <= d.exp_detected_cells <= 6.0)

    def test_diagnostics_on_every_decision_kind(self):
        tr = self.trk.report()
        fe = BeamFrontEnd(site=self.site, on_unmet="reuse_pd")
        d = fe.plan(tr, self.grants, self.rcs)
        self._check_diag(d)
        self._check_diag(fe.dedicated(tr, [], self.rcs, t=0.4))
        near = TR.CVTracker(1, 0.0, (5.25, -60.5), (0.0, 8.0), self.rsu, self.bore, (4.73, 1.88),
                            P0=np.diag([0.3, 0.3, 0.2, 0.2])).report()
        for x in fe.plan_frame([tr, near], self.grants):
            self._check_diag(x)
        m = fe.plan_multi(tr, self.grants, self.rcs, max_grants=3, objective="coverage")
        self._check_diag(m)
        # a larger inflation can only lower the worst-case coverage of the same beam
        wide = BeamFrontEnd(site=self.site, coverage_k=3.0)
        pred = fe.prediction(tr, 0.4)
        c1 = fe.coverage_beams(d.weights, fe.coverage_points(pred))[0][0]
        c3 = wide.coverage_beams(d.weights, wide.coverage_points(pred))[0][0]
        self.assertAlmostEqual(c1, d.coverage_db, places=9)
        self.assertLessEqual(c3, c1 + 1e-12)

    def test_coverage_objective_maximises_coverage(self):
        tr = self.trk.report()
        fe = BeamFrontEnd(site=self.site)
        tab = fe.plan(tr, self.grants, self.rcs, return_table=True).candidates
        d = fe.plan(tr, self.grants, self.rcs, objective="coverage")
        self.assertEqual(d.info["objective"], "coverage")
        ok = np.round(tab["pd"], 12) >= fe.tau
        pred = fe.prediction(tr, 0.4)
        cov = fe.coverage_beams(tab["weights"][ok], fe.coverage_points(pred))[0]
        self.assertAlmostEqual(d.coverage_db, float(cov.max()), places=9)
        self.assertGreaterEqual(d.pd_pred, fe.tau)
        base = fe.plan(tr, self.grants, self.rcs)
        self.assertGreaterEqual(d.coverage_db, base.coverage_db - 1e-9)

    def test_quality_objective_with_min_coverage(self):
        from isac.interfaces import SensingRequirement

        tr = self.trk.report()
        fe = BeamFrontEnd(site=self.site, on_unmet="reuse_pd")
        cov_best = fe.plan(tr, self.grants, self.rcs, objective="coverage").coverage_db
        req = SensingRequirement(min_coverage_db=cov_best - 1.0)
        d = fe.plan(tr, self.grants, self.rcs, requirement=req, objective="quality")
        self.assertEqual(d.mode, "reuse")
        self.assertTrue(d.requirement_met)
        self.assertGreaterEqual(d.coverage_db, req.min_coverage_db - 1e-9)
        self.assertIn("min_coverage_db", d.info["binding"])
        free = fe.plan(tr, self.grants, self.rcs, objective="quality")
        self.assertGreaterEqual(free.exp_detected_cells, d.exp_detected_cells - 1e-9)
        # the default objective also honours min_coverage_db as a requirement
        dp = fe.plan(tr, self.grants, self.rcs, requirement=req)
        if dp.requirement_met:
            self.assertGreaterEqual(dp.coverage_db, req.min_coverage_db - 1e-9)

    def test_multi_grant_coverage_growth(self):
        tr = self.trk.report()
        fe = BeamFrontEnd(site=self.site)
        single = fe.plan(tr, self.grants, self.rcs, objective="coverage")
        m = fe.plan_multi(tr, self.grants, self.rcs, max_grants=3, objective="coverage")
        self.assertTrue(np.all(np.diff(m.info["steps_score"]) > 0))
        self.assertGreaterEqual(m.coverage_db, single.coverage_db - 1e-6)
        if m.mode == "reuse_multi":
            self.assertIn("coverage_maxpool_db", m.info)
            self.assertGreaterEqual(m.info["coverage_maxpool_db"], m.coverage_db - 1e-9)

    def test_covering_dedicated_width_and_extent_prior(self):
        from isac.site import young_track_extent

        tr = self.trk.report()
        plain = BeamFrontEnd(site=self.site).dedicated(tr, [], self.rcs, t=0.4)
        cov = BeamFrontEnd(site=self.site, dedicated_covering=True, coverage_k=2.0)
        d = cov.dedicated(tr, [], self.rcs, t=0.4)
        self.assertIn(d.info["dedicated_width"], (32, 16, 8, 4))
        self.assertEqual(d.tag, -100 - d.info["dedicated_width"])
        pred = cov.prediction(tr, 0.4)
        full = np.exp(-1j * np.pi * V.N_IDX * V.to_polar(pred["x"][None, :2], self.rsu, self.bore,
                                                          V.P["rsu_h"], V.P["rc_h"])[0][0])
        full /= math.sqrt(V.M)
        self.assertGreaterEqual(d.coverage_gain_dbi,
                                cov.coverage_beams(full, cov.coverage_points(pred))[1][0] - 1e-9)
        self.assertTrue(np.all(np.isfinite([plain.coverage_db, d.coverage_db])))
        young = TrackReport(0, 0.0, tr.x, tr.P, (3.0, 1.5), "car", hits=1)
        fe = BeamFrontEnd(site=self.site, extent_prior=young_track_extent())
        self.assertEqual(fe.prediction(young, 0.4)["extent"], (4.5, 1.9))
        old = TrackReport(0, 0.0, tr.x, tr.P, (3.0, 1.5), "car", hits=7)
        self.assertEqual(fe.prediction(old, 0.4)["extent"], (3.0, 1.5))



class LinkBudgetSpeckleTests(unittest.TestCase):
    """Round 5: explicit link-budget parameters and Gamma(m) speckle in the extended detector."""

    @classmethod
    def setUpClass(cls):
        from isac.site import SiteConfig

        CL.configure()
        cls.SiteConfig = SiteConfig
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        rng = np.random.default_rng(5)
        cls.grants = CL.Scheduler(cls.rsu, cls.bore, rng, n_grants=4).grants(0.4)
        cls.trk = TR.CVTracker(0, 0.0, (5.25, -62.0), (0.0, 8.0), cls.rsu, cls.bore,
                               (4.73, 1.88), P0=np.diag([0.3, 0.3, 0.2, 0.2]))
        cls.rcs = RcsReport(-15.0, -15.0, 0, "x", np.array([-18.0, -15.0, -12.0]), np.ones(3) / 3)

    def test_explicit_link_budget(self):
        S = self.SiteConfig
        a = S(rsu_xy=self.rsu, boresight_rad=self.bore, tx_power_dbm=38.0, noise_figure_db=7.0,
              noise_bandwidth_hz=190.08e6, element_gain_dbi=5.0)
        self.assertEqual((a.p_ref_dbm, a.nf_rx, a.noise_bw, a.e_max), (38.0, 7.0, 190.08e6, 5.0))
        with self.assertRaises(ValueError):
            S(tx_power_dbm=38.0, p_ref_dbm=30.0)
        pre = S.fr2_38dbm(rsu_xy=self.rsu, boresight_rad=self.bore)
        self.assertEqual((pre.p_ref_dbm, pre.nf_rx, pre.noise_bw, pre.e_max, pre.sensing_bandwidth_hz),
                         (38.0, 7.0, 190.08e6, 5.0, 190.08e6))
        tr = self.trk.report()
        ref = BeamFrontEnd(self.rsu, self.bore).dedicated(tr, [], self.rcs, t=0.0)
        new = BeamFrontEnd(site=a).dedicated(tr, [], self.rcs, t=0.0)
        expect = ((38.0 - V.P["p_ref_dbm"]) - 10 * math.log10(190.08e6 / V.P["B"])
                  - (7.0 - V.P["nf_rx"]) + 2 * (5.0 - V.P["e_max"]))
        np.testing.assert_array_equal(new.weights, ref.weights)
        self.assertAlmostEqual(new.pred_cell_snr_db - ref.pred_cell_snr_db, expect, places=6)
        self.assertEqual(V.P["p_ref_dbm"], CL.V.P["p_ref_dbm"])  # globals restored

    def test_cell_pd_series_exact(self):
        from scipy.stats import ncx2

        eta = -math.log(V.P["pfa"])
        rho = np.array([0.1, 1.0, 5.0, 20.0, 100.0, 1000.0, 1e4])
        leak = np.array([0.0, 0.0, 0.5, 2.0, 0.0, 3.0, 0.0])
        ray = TG.cell_pd(rho, leak)
        saved = dict(V.P)
        try:
            V.P["speckle_m"] = 1.0 + 1e-9
            np.testing.assert_allclose(TG.cell_pd(rho, leak), ray, atol=1e-8)
            V.P["speckle_m"] = float("inf")
            np.testing.assert_allclose(TG.cell_pd(rho, leak),
                                       ncx2.sf(2 * eta, 2, 2 * rho / (1 + leak)), atol=1e-10)
            rng = np.random.default_rng(1)
            V.P["speckle_m"] = 3.0
            S = rng.gamma(3.0, 1 / 3.0, size=(200000, 1)) * (rho / (1 + leak))[None, :]
            n = (rng.normal(size=S.shape) + 1j * rng.normal(size=S.shape)) / math.sqrt(2)
            mc = (np.abs(np.sqrt(S) + n) ** 2 > eta).mean(axis=0)
            np.testing.assert_allclose(TG.cell_pd(rho, leak), mc, atol=0.004)
            pd = [V.P.__setitem__("speckle_m", m) or float(TG.cell_pd(np.array([40.0]))[0])
                  for m in (1.0 + 1e-9, 2.0, 8.0, float("inf"))]
            self.assertTrue(np.all(np.diff(pd) > 0))  # less fading, higher P_D at high SNR
        finally:
            V.P.clear()
            V.P.update(saved)

    def test_speckle_order_in_planning(self):
        tr = self.trk.report()
        ray = BeamFrontEnd(self.rsu, self.bore)
        site = self.SiteConfig(rsu_xy=self.rsu, boresight_rad=self.bore, speckle_m=4.0)
        fe = BeamFrontEnd(site=site)
        lazy = fe.plan(tr, self.grants, self.rcs)
        full = BeamFrontEnd(site=site, lazy=False).plan(tr, self.grants, self.rcs)
        self.assertEqual((lazy.mode, lazy.grant, lazy.tag), (full.mode, full.grant, full.tag))
        self.assertAlmostEqual(lazy.pd_pred, full.pd_pred, places=12)
        off, pd = lazy.pd_offset_curve
        self.assertAlmostEqual(float(pd[np.argmin(abs(off))]), lazy.pd_pred, places=9)
        self.assertTrue(np.all(np.diff(pd) >= -1e-12))
        self.assertNotIn("speckle_m", V.P)
        d_ray = ray.dedicated(tr, [], self.rcs, t=0.0)
        d_m = fe.dedicated(tr, [], self.rcs, t=0.0)
        np.testing.assert_array_equal(d_ray.weights, d_m.weights)
        if d_ray.pd_pred > 0.6:
            self.assertGreater(d_m.pd_pred, d_ray.pd_pred)



class CalibrationPresetTests(unittest.TestCase):
    """Round 6: mimo_tracking v6 calibration preset and per-class design RCS."""

    @classmethod
    def setUpClass(cls):
        from isac.site import SiteConfig

        CL.configure()
        cls.SiteConfig = SiteConfig
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        cls.grants = CL.Scheduler(cls.rsu, cls.bore, np.random.default_rng(5), n_grants=4).grants(0.4)

    def test_preset_values(self):
        s = self.SiteConfig.mimo_v6(rsu_xy=self.rsu, boresight_rad=self.bore)
        self.assertEqual((s.p_ref_dbm, s.nf_rx, s.noise_bw, s.e_max, s.sensing_bandwidth_hz),
                         (38.0, 7.0, 190.08e6, 5.0, 190.08e6))
        self.assertAlmostEqual(s.pfa, math.exp(-10 ** 1.3))
        self.assertEqual((s.snr_offset_db, s.snr_sigma_db, s.rcs_prior_dbsm), (19.4, 4.3, -1.1))
        self.assertFalse(s.element_pattern)  # the RT and the fit used isotropic elements
        self.assertTrue(self.SiteConfig.fr2_38dbm().element_pattern)
        np.testing.assert_allclose([s.rcs_prior_by_class[c] for c in ("car", "bus", "truck")],
                                   [-5.9, 0.1, 0.8], atol=1e-9)
        self.assertEqual(self.SiteConfig.mimo_v6(snr_offset_db=18.9, snr_sigma_db=0.0).snr_offset_db,
                         18.9)

    def test_per_class_design_rcs(self):
        # fixed-size dedicated look, so the P_D comparisons isolate the calibration
        site = self.SiteConfig.mimo_v6(rsu_xy=self.rsu, boresight_rad=self.bore,
                                       dedicated_sizing=False)
        fe = BeamFrontEnd(site=site, on_unmet="reuse_pd")
        x, P = np.array([5.25, -62.0, 0.0, 8.0]), np.diag([0.3, 0.3, 0.2, 0.2])
        car = TrackReport(0, 0.0, x, P, (4.73, 1.88), "car")
        bus = TrackReport(0, 0.0, x, P, (4.73, 1.88), "bus")
        ped = TrackReport(0, 0.0, x, P, (4.73, 1.88), "pedestrian")
        self.assertEqual(fe.default_rcs(car).design_dbsm, site.rcs_prior_by_class["car"])
        self.assertEqual(fe.default_rcs(ped).design_dbsm, -1.1)
        self.assertEqual(BeamFrontEnd(self.rsu, self.bore).default_rcs(car).design_dbsm,
                         -1.10319987650635)
        dc, db_ = fe.dedicated(car, [], t=0.4), fe.dedicated(bus, [], t=0.4)
        self.assertEqual(dc.info["rcs_source"], "site-class:car")
        self.assertAlmostEqual(db_.rcs_design_dbsm - dc.rcs_design_dbsm, 6.0, places=9)
        self.assertGreaterEqual(db_.pd_pred, dc.pd_pred)
        d = fe.plan(car, self.grants)
        self.assertEqual(d.rcs_design_dbsm, site.rcs_prior_by_class["car"])
        # calibration lowers the predicted P_D relative to the uncalibrated link budget
        raw = BeamFrontEnd(site=self.SiteConfig.fr2_38dbm(rsu_xy=self.rsu, boresight_rad=self.bore))
        self.assertLess(dc.pd_pred, raw.dedicated(car, [], t=0.4).pd_pred)


class DedicatedSizingTests(unittest.TestCase):
    """Round 7: RE-sized dedicated fallback and the mimo_v6 RCS population."""

    @classmethod
    def setUpClass(cls):
        from isac.site import SiteConfig

        CL.configure()
        cls.SiteConfig = SiteConfig
        cls.meta, cls.rsu, cls.bore = LS.geometry()
        P = np.diag([0.3, 0.3, 0.2, 0.2])
        cls.near = TrackReport(0, 0.0, np.array([5.25, -30.0, 0.0, 8.0]), P, (4.73, 1.88), "car")
        cls.far = TrackReport(0, 0.0, np.array([5.25, -62.0, 0.0, 8.0]), P, (4.73, 1.88), "car")

    def site(self, **kw):
        return self.SiteConfig.mimo_v6(rsu_xy=self.rsu, boresight_rad=self.bore, **kw)

    def test_smallest_step_meeting_tau(self):
        fe = BeamFrontEnd(site=self.site())
        d = fe.dedicated(self.near, [], t=0.4)
        base = V.P["ded_prb"] * 12 * V.P["ded_sym"]
        self.assertEqual(d.info["dedicated_ladder"][0], base)
        self.assertEqual(d.info["dedicated_ladder"][-1], 32768)
        self.assertEqual(d.n_re, d.info["dedicated_n_re"])
        self.assertEqual(d.n_re, d.info["dedicated_ladder"][d.info["dedicated_step"]])
        self.assertFalse(d.info["dedicated_cap_hit"])
        self.assertGreaterEqual(d.pd_pred, fe.tau)
        self.assertGreater(d.info["dedicated_step"], 0)
        k = d.n_re // base  # slow-time multiple; bandwidth unchanged
        self.assertEqual(d.bandwidth, V.P["ded_prb"] * 12 * V.P["df"])
        # the RCS-shift shortcut equals a fixed look of the same size; one step less misses tau
        fixed = BeamFrontEnd(site=self.site(dedicated_sizing=False,
                                            extra={"ded_sym": V.P["ded_sym"] * k}))
        self.assertAlmostEqual(fixed.dedicated(self.near, [], t=0.4).pd_pred, d.pd_pred, places=9)
        less = BeamFrontEnd(site=self.site(dedicated_sizing=False,
                                           extra={"ded_sym": V.P["ded_sym"] * k // 2}))
        self.assertLess(less.dedicated(self.near, [], t=0.4).pd_pred, fe.tau)
        grants = CL.Scheduler(self.rsu, self.bore, np.random.default_rng(5), n_grants=4).grants(0.4)
        se = np.mean([next(m for m in V.MCS if m[0] == g.mcs)[2] for g in grants])
        self.assertAlmostEqual(fe.dedicated(self.near, grants, t=0.4).cost_bits, d.n_re * se)

    def test_cap_flagged(self):
        fe = BeamFrontEnd(site=self.site())
        d = fe.dedicated(self.far, [], t=0.4)
        self.assertTrue(d.info["dedicated_cap_hit"])
        self.assertEqual(d.n_re, 32768)
        self.assertLess(d.pd_pred, fe.tau)
        small = BeamFrontEnd(site=self.site(dedicated_cap_re=1728)).dedicated(self.far, [], t=0.4)
        self.assertEqual((small.n_re, small.info["dedicated_ladder"]), (1728, [432, 864, 1728]))

    def test_default_unchanged(self):
        for fe in (BeamFrontEnd(self.rsu, self.bore),
                   BeamFrontEnd(site=self.site(dedicated_sizing=False))):
            d = fe.dedicated(self.near, [], t=0.4)
            self.assertEqual(d.n_re, V.P["ded_prb"] * 12 * V.P["ded_sym"])
            self.assertNotIn("dedicated_step", d.info)
        self.assertFalse(self.SiteConfig().dedicated_sizing)
        self.assertTrue(self.SiteConfig().is_default())

    def test_population_preset(self):
        pop = RcsPopulation.mimo_v6()
        np.testing.assert_allclose([pop.mu[c] for c in ("car", "bus", "truck")], [-5.9, 0.1, 0.8],
                                   atol=1e-9)
        self.assertAlmostEqual(sum(pop.weight.values()), 1.0)
        self.assertAlmostEqual(pop.weight["bus"] / pop.weight["car"], 370 / 163)
        self.assertNotIn("pedestrian", pop.weight)
        self.assertAlmostEqual(pop.rho_frame, 0.333 ** (0.1 / 1.2))
        self.assertEqual((pop.omega, pop.frame_dt), (1.7, 0.1))
        rep = RcsEstimator(pop).report(0.0)  # unknown class: the bus/truck/car mixture
        self.assertTrue(-5.9 < rep.mean_dbsm < 0.8)
        self.assertLess(rep.design_dbsm, rep.mean_dbsm)


if __name__ == "__main__":
    unittest.main()
