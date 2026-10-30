"""Full-body M2 v2 mjlab task, pure parts (M2V2_PLAN_v3 M9; gpu/fullbody_task.py, gpu/fb_population.py; A9): the population draws, the
originals + L1 motion, the T0 / T1 mux and the mirror mask, design 4.2's mirrored-strike swap (inert on T0 rows), the opponent DR rows,
the spec edits (shell names, hulls, bits, floor), the critic blocks, the B-from-A load, the WATCH6 key contract, the replica's L1 /
fb_mirror wiring (no bout simulated), and mjlab 1.2.0's real scene compiled with warp stubbed against the replica (subprocess:
scripts/fb_m9_check.py --only model, skipped without the mjlab source)."""
import sys, os, copy, subprocess
from pathlib import Path
import numpy as np, pytest, torch, mujoco
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "gpu", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import fb_population as POP, fb_opponent as FO, fb_rewards as RW, fb_mirror as FM, fullbody_task as FT, fb_watch as WT

@pytest.fixture(scope="module")
def M(): return POP.FbMotion()

def test_motion_is_originals_then_l1(M):
    with np.load(FO.MOTION) as O, np.load(POP.FB / "fb_l1.npz") as L:
        assert M.clips[:3] == FO.TRAIN and M.clips[3:] == tuple(L["names"].tolist()) and M.n_orig == 3 and M.family[:3] == ("original",) * 3
        for i, c in enumerate(M.clips):
            A, j = (O, i) if i < 3 else (L, i - 3); r0, T = int(A["row0"][j]), int(A["frames"][j]); s = slice(int(M.row0[i]), int(M.row0[i]) + T)
            assert int(M.frames[i]) == T and np.array_equal(M.q[s].numpy(), A["q"][r0:r0 + T]) and np.array_equal(M.ref_q[s].numpy(), A["ref_joint_pos"][r0:r0 + T])
        assert np.array_equal(M.conj.numpy(), np.r_[[False] * 3, L["conj"].astype(bool)]) and np.array_equal(M.clean[3:].numpy(), L["clean"])
        assert len(M.strikes["clip"]) == len(O["strike_clip"]) + len(L["strike_row"]) and np.array_equal(M.strikes["onset"][len(O["strike_clip"]):], L["strike_onset"])
    Ml = POP.FbMotion(loco=True); assert POP.LOCO_DROP not in Ml.host and Ml.n_orig == 2 and set(Ml.host) == {"kfab_200", "kfab_201"}

def test_population_draws(M):
    assert POP.ROWS == RW.ROWS
    p = POP.Population(M); d = p.draw(20000, torch.Generator().manual_seed(0)); fam = np.array(M.family)
    for i, s in enumerate(POP.SHARES): assert abs(float((d["row"] == i).double().mean()) - s) < 0.012
    t0, t1 = d["row"] < 2, d["row"] >= 2
    assert (d["clip"][t0] < 3).all() and (d["clip"][t1] >= 3).all() and set(d["distance"][t0].tolist()) <= set(POP.T0_DIST)
    assert float(d["distance"][t1].min()) >= 0.35 - 1e-9 and float(d["distance"][t1].max()) <= 0.65 + 1e-9 and len(set(d["distance"][t1].tolist())) == 31
    assert M.clean[d["clip"], d["dist_ix"], torch.round(d["t0"] / M.start_step).long()].all()
    assert torch.equal(d["conj"], M.conj[d["clip"]]) and set(fam[d["clip"][d["t1"]].numpy()]) == {"reangle"} and not (d["t1"] & t0).any()
    assert torch.equal(d["aimed"], d["row"] != 1) and torch.equal(d["jitter"], t1) and torch.equal(d["push"], t1) and torch.equal(d["opp_dr"], d["row"] == 3)
    assert (p.draw(500, torch.Generator().manual_seed(1), t0_only=True)["row"] == 0).all()
    d2 = POP.Population(M, t1_ok=False).draw(5000, torch.Generator().manual_seed(2)); assert not d2["t1"].any() and "reangle" not in set(fam[d2["clip"].numpy()])

def test_mux_and_mirror_mask(M):
    drv, mux, pol = POP.driver(4, M, torch.float64); t1 = FO.TrackerMLP(dtype=torch.float64); mux.t1 = lambda o: t1(o) + 1.0
    mux.sel[:] = torch.tensor([False, True, False, True]); pol.set([0, 1, 2, 3], torch.tensor([False, False, True, True])); o = torch.randn(4, 132, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    a, base = drv.mlp(o), t1(o); m = FM.mirror_act(t1(FM.mirror_obs(o)))
    assert torch.equal(a[0], base[0]) and torch.equal(a[1], base[1] + 1) and torch.equal(a[2], m[2]) and torch.equal(a[3], FM.mirror_act(t1(FM.mirror_obs(o))[3:] + 1)[0])

def _drive(M, sw, ref, clip, jit, steps):
    n = len(clip); dt = torch.float64; drv = FO.TrackerDriver(n, dt, motion=M); drv.clip[:] = clip; sw.reset(None, clip, 0.45, True, jit); ref.reset(None, clip, 0.45, True, jit)
    ox = torch.tensor([[0.45, 0.0, 1.0]], dtype=dt).expand(n, 3); oR = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=dt)).expand(n, 3, 3); dx = torch.tensor([[0.0, 0.0, 1.0]], dtype=dt).expand(n, 3); dR = torch.eye(3, dtype=dt).expand(n, 3, 3)
    out = []
    for k in range(steps):
        t = torch.full((n,), k * 0.02 + 0.02, dtype=dt); fi = drv.clock(t); rq, rdq, _, _ = drv.reference(fi); z = torch.zeros(n, dtype=torch.bool)
        out.append((sw.step(fi, t, rq, rdq, ox, oR, dx, dR, z), ref.step(fi, t, rq, rdq, ox, oR, dx, dR, z), rq, sw.weight(t).clone()))
    return out

def test_swap_is_inert_on_t0_rows_and_mirrors_the_upper_body_on_t1(M):
    """Design 4.2 / A9.4: jitter False rows bit-equal to the AimOp alone; p = 1 on jitter rows: every new intent swapped (limb flipped),
    legs untouched, the upper body blended toward its mirror only while a swapped intent's weight is on."""
    dt = torch.float64; mk = lambda: FO.AimOp(2, dt, motion=M, slots=8); a, r = mk(), mk(); sw = POP.SwapAim(a, p=1.0, gen=torch.Generator().manual_seed(1))
    out = _drive(M, sw, r, torch.tensor([0, M.index("l1_reangle_kfab_200_o0")]), torch.tensor([False, False]), 150)
    assert all(all(torch.equal(u, v) for u, v in zip(x, y)) for x, y, _, _ in out) and int(sw.swaps.sum()) == 0
    a, r = mk(), mk(); sw = POP.SwapAim(a, p=1.0, gen=torch.Generator().manual_seed(1)); out = _drive(M, sw, r, torch.tensor([0, 0]), torch.tensor([False, True]), 150)
    assert int(sw.swaps[0]) == 0 and int(sw.swaps[1]) > 0 and torch.equal(a.limb[0], r.limb[0]) and not torch.equal(a.limb[1, :8], r.limb[1, :8])
    for (x, _, rq, w) in out:
        assert torch.equal(x[0][:, :12], rq[:, :12]) and float(w[0]) == 0.0
    on = [k for k, (_, _, _, w) in enumerate(out) if float(w[1]) > 0]; off = [k for k, (_, _, _, w) in enumerate(out) if float(w[1]) == 0 and k > 0 and float(out[k - 1][3][1]) == 0]
    assert on and off
    sw0 = POP.SwapAim(FO.AimOp(2, dt, motion=M, slots=8), p=0.0); r0 = FO.AimOp(2, dt, motion=M, slots=8); out0 = _drive(M, sw0, r0, torch.tensor([0, 0]), torch.tensor([True, True]), 120)
    assert all(all(torch.equal(u, v) for u, v in zip(x, y)) for x, y, _, _ in out0) and int(sw0.swaps.sum()) == 0   # p 0: the jittered AimOp exactly

def test_opp_dr_touches_the_opponent_only():
    import fb_dr as DR, fb_replica as RP, export_hybrid_fb as EX
    from r1gym.model import CTRL_JOINTS
    _, _, cfg = EX.env_scale(); m, _ = RP.build_model({**cfg, "opponent_mode": "policy", "opponent_aim": EX.AIM}, RP.conf()); ox = POP.OppDR(m, CTRL_JOINTS)
    F, nom = DR.model_fields(m); F = {k: v.clone() for k, v in F.items()}; ox.write(F, [0], ox.sample(1, generator=torch.Generator().manual_seed(0)))
    opp = lambda b: m.body(int(b)).name.startswith("opp_")
    assert all(opp(b) for b in np.flatnonzero((F["body_mass"][0] != nom["body_mass"]).numpy())) and int((F["body_mass"][0] != nom["body_mass"]).sum()) == 25
    assert all(opp(m.geom_bodyid[g]) for g in np.flatnonzero((F["geom_friction"][0] != nom["geom_friction"]).any(1).numpy())) and ox.na == 24 and len(ox.feet) == 14
    assert all(m.actuator(int(a)).name.startswith("opp_") for a in np.flatnonzero((F["actuator_gainprm"][0] != nom["actuator_gainprm"]).any(1).numpy()))
    for k in ("body_ipos", "actuator_forcerange", "dof_armature"): assert torch.equal(F[k][0], nom[k])

def test_spec_edits():
    base = lambda: mujoco.MjSpec.from_file(str(ROOT / "r1gym" / "assets" / "r1" / "r1.xml"))
    s = FT.entity_spec(base, "full"); sh = [g for g in s.geoms if g.name.endswith("_shell")]; assert len(sh) == 32 and tuple(g.meshname for g in sh) == FT._shell_meshes()
    s = FT.entity_spec(base, "mix"); sh = [g for g in s.geoms if g.name.endswith("_shell")]
    with np.load(FT.HULLS_NPZ) as z: n = dict(zip(z["meshes"].tolist(), z["n"].tolist()))
    assert all(len(s.mesh(g.meshname).uservert) == 3 * n[g.meshname] and s.mesh("fborig_" + g.meshname) is not None for g in sh) and tuple(g.meshname for g in sh) == FT._shell_meshes()   # A10
    sc = mujoco.MjSpec(); sc.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01]); b = sc.worldbody.add_body(name="robot/x"); o = sc.worldbody.add_body(name="opponent/x")
    g1 = b.add_geom(name="robot/a_shell"); g2 = o.add_geom(name="opponent/a_shell"); g3 = o.add_geom(name="opponent/a_collision"); g3.contype, g3.conaffinity = 2, 2; g4 = b.add_geom(name="robot/b_collision")
    FT.scene_edit(sc); fl = sc.geoms[0]
    assert (fl.contype, fl.conaffinity) == FT.FLOOR_BITS and (g1.contype, g1.conaffinity, g1.condim) == (0, 4, 1) and (g2.contype, g2.conaffinity) == (4, 0)
    assert list(g1.solref) == list(FT.SHELL_SOLREF) and list(g3.solimp) == list(FT.OPP_SOLIMP) and list(g4.solref) == [0.02, 1.0]
    assert FT.collision_kwargs(FT.ENT)["priority"] == 1 and FT.collision_kwargs(FT.OPP)["priority"] == {FT.FOOT: 1}

def test_critic_blocks_and_intent_features(M):
    n = 3; aim = FO.AimOp(n, torch.float64, motion=M); aim.reset(None, torch.tensor([0, 1, 2]), 0.45); now = torch.full((n,), 0.5, dtype=torch.float64)
    f = FT.intent_features(aim, torch.zeros(n, aim.K + 1, dtype=torch.bool), now); assert f.shape == (n, FT.INTENT_DIM) and not f.any()
    aim.valid[0, 0] = True; aim.limb[0, 0] = 1; aim.t_on[0, 0], aim.t_apex[0, 0], aim.t_end[0, 0] = 0.4, 0.6, 0.7; sw = torch.zeros(n, aim.K + 1, dtype=torch.bool); sw[0, 0] = True
    f = FT.intent_features(aim, sw, now); assert f[0, 5] == 1 and f[0, 6] > 0 and abs(float(f[0, 7]) - 0.1) < 1e-6 and f[0, 9] == 1 and not f[0, :5].any() and not f[1:].any()
    import fb_dr as DR
    dr = {k: torch.ones(n, *s) for k, s in (("mass", (25,)), ("torso_kg", ()), ("com_m", (2, 3)), ("foot_mu", ()), ("kp", (24,)), ("kd", (24,)), ("cap", (24,)), ("armature", (30,)))}
    p = FT.priv_block(torch.zeros(n, 2), torch.zeros(n), torch.tensor([[0.5, 0.0, 0.7]] * n), torch.full((n,), np.pi), torch.zeros(n, 3), torch.tensor([True, False, False]),
                      torch.zeros(n), torch.tensor([True, False, True]), torch.tensor([0, 1, 2]), dr, torch.zeros(n), torch.zeros(n))
    assert p.shape == (n, len(FT.PRIV)) and abs(float(p[0, 0]) - 0.5) < 1e-6 and float(p[0, 4]) == -1.0 and p[0, 8] == 1 and p[1, 12] == 1 and p[2, 13] == 1
    assert torch.allclose(FT.ring_home(torch.tensor([[1.0, 0.0]]), torch.zeros(1, 2), torch.tensor([np.pi / 2])), torch.tensor([[0.0, -1.0]]), atol=1e-6)

def test_b_loads_an_a_checkpoint_as_to_stage_b():
    """The B restart (FbActorModel B + load_into_b) = to_stage_b on the loaded A checkpoint, bit for bit; a B checkpoint loads as is."""
    import hybrid_actor as HA, fb_m7_check as M7
    sc, clip, _ = FT.eval_contract(); A = M7.perturbed(HA.HybridActor.from_checkpoints(ROOT, sc, clip), 3); sd = A.state_dict()
    B = HA.HybridActor.from_checkpoints(ROOT, sc, clip).to_stage_b(seed=7); FT.load_into_b(B, B.load_state_dict, sd); ref = copy.deepcopy(A).to_stage_b(seed=7)
    assert all(torch.equal(v, ref.state_dict()[k]) for k, v in B.state_dict().items())
    with torch.no_grad(): ref.residual[0].weight.add_(0.01)
    FT.load_into_b(B, B.load_state_dict, ref.state_dict()); assert torch.equal(B.residual[0].weight, ref.residual[0].weight)

def test_a_checkpoint_loads_only_into_its_own_twin():
    """A9.2: FbActorModel's load refuses a checkpoint whose stored blind flag differs from the run's (a sighted A into a blind B, ...)."""
    import hybrid_actor as HA
    sc, clip, _ = FT.eval_contract(); a, b = (HA.HybridActor.from_checkpoints(ROOT, sc, clip, x) for x in (False, True))
    FT.load_same_blind(a, a.load_state_dict, a.state_dict()); FT.load_same_blind(b, b.load_state_dict, b.state_dict())
    with pytest.raises(AssertionError): FT.load_same_blind(a, a.load_state_dict, b.state_dict())
    with pytest.raises(AssertionError): FT.load_same_blind(b, b.load_state_dict, a.state_dict())
    assert HA.stored_blind(a.state_dict()) is False and HA.stored_blind(b.state_dict()) is True

def test_t1_samples_the_reangle_family_only():
    """A9.3: R1-Track-Spar-Reangle keeps the exported T1 motion (fed references unchanged) and gives every bin outside the originals and
    the reangle rows probability 0; with every bin allowed it is r1_track_spar_cfg.group_probs exactly."""
    import json, r1_track_spar_t1 as T1, r1_track_spar_cfg as C
    assert T1.T1_FAMILIES == POP.T1_FAMILIES
    for f in ("fb_l1_track_index.json", "fb_l1_track_loco_index.json"):
        ix = json.load(open(POP.FB / f, encoding="utf-8")); a = T1.bin_allowed(ix, ix["frames"]); g = C.bin_groups(ix, ix["frames"]); k = torch.tensor([0.5, 0.3, 0.2])
        assert a[g == 0].all() and a[g == 1].any() and not a[g == 1].all() and len(a) == len(g)
        failed = torch.rand(len(a), generator=torch.Generator().manual_seed(0)); p = T1.masked_probs(failed, 0.1, k, torch.as_tensor(g), torch.as_tensor(a))
        assert abs(float(p.sum()) - 1) < 1e-6 and float(p[torch.as_tensor(~a)].sum()) == 0.0 and abs(float(p[torch.as_tensor(g == 0)].sum()) - C.ORIGINAL_SHARE) < 1e-6
        assert torch.equal(T1.masked_probs(failed, 0.1, k, torch.as_tensor(g), torch.ones(len(a), dtype=torch.bool)), C.group_probs(failed, 0.1, k, torch.as_tensor(g)))
        starts = {r["start_frame"]: r["group"] for r in ix["rows"].values()}; c = (np.arange(len(a)) + 0.5) / len(a) * (ix["frames"] - 1); s = np.array(sorted(starts))
        assert all(a[i] == (starts[s[np.searchsorted(s, c[i], side="right") - 1]] in ("original", "reangle")) for i in range(len(a)))

def test_watch_and_metric_keys_match_the_task():
    assert FT.WATCH_KEYS == WT.COUNTERS and FT.ACTOR_KEYS == WT.ACTOR and FT.STAGES == WT.STAGES

def test_replica_l1_wiring_sets_the_mirror_mask(M):
    """fb_replica conf motion 'l1' (M8b contract): FbMotion, the conjugated driver (mask = the row's npz conj at reset), SwapAim; no bout."""
    import fb_replica as RP
    c = RP.conf(backend="torch64", reset="train", motion="l1"); Mr, drv, aim, pol = RP.opp_modules(2, torch.float64, c)
    assert isinstance(drv.mlp, FM.MirrorPolicy) and isinstance(aim, POP.SwapAim) and Mr.clips == M.clips
    for name in ("l1_warp_kfab_200_m0", "l1_warp_kfab_200_o0", "kfab_201"):
        ci = RP.opp_clip(Mr, pol, [1], dict(clip=M.host[M.index(name)], row=name)); assert ci == M.index(name) and bool(pol.mask[1]) == bool(M.conj[ci])
    with pytest.raises(AssertionError): RP.opp_clip(Mr, pol, [0], dict(clip="kfab_201", row="l1_warp_kfab_200_o0"))
    with pytest.raises(AssertionError): RP.conf(backend="numpy", motion="l1")
    assert RP.opp_modules(1, torch.float64, RP.conf(backend="torch64"))[3] is None   # motion 'fb': M6's modules unchanged

@pytest.mark.skipif(not (Path(os.environ.get("FB_MJLAB_SRC", r"C:\Users\dundy\AppData\Local\Temp\claude\C--Users-dundy-Downloads\96df40a6-a649-4bd5-bef4-75678b326b3c\scratchpad\m1m2\run2b\src")) / "mjlab_x").exists(), reason="mjlab 1.2.0 source not present")
def test_mjlab_scene_equals_the_replica(tmp_path):
    """M9 (1) in a subprocess (it stubs warp / scipy in sys.modules): mjlab's real scene vs the replica, every hull set and solver."""
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "fb_m9_check.py"), "--only", "model", "--out", str(tmp_path / "m.json")], capture_output=True, text=True, cwd=str(ROOT), timeout=600,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    import json; d = json.load(open(tmp_path / "m.json"))["model"]
    assert r.returncode == 0 and d["ok"] and all(x["n_mismatch"] == 0 for x in d["runs"]) and d["floor_without_scene_edit"]["conaffinity"] == 1, r.stdout[-2000:] + r.stderr[-2000:]
