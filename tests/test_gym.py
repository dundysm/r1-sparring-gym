import numpy as np, mujoco, pytest
from r1gym.model import build_spec, R1Robot, load_cfg, torque_caps, CTRL_JOINTS
from r1gym.check_model import stand
from r1gym.perturb import make_hit, HitApplier, random_schedule, LOCATIONS
from r1gym.sensors import DelayedNoisyObs
from r1gym.env import R1HitEnv, BatchEnv, OBS_DIM, ACT_DIM
from r1gym.opponent import FightReplay, list_clips

CFG = load_cfg()

def test_model_compiles_with_24_capped_actuators():
    m = build_spec(cfg=CFG).compile(); assert m.nu == 24
    caps = torque_caps(CFG); assert np.allclose(m.actuator_forcerange[:, 1], caps)
    assert caps.max() <= 60 and abs(m.body_subtreemass[m.body("pelvis").id] - 28.93) < 0.1

def test_stand_pd_stands_and_stock_gains_fall():
    assert stand(CFG, "stand_pd")[0] is True
    assert stand(CFG, "pd")[0] is False      # documented: Unitree deploy gains cannot hold a passive stand

def test_hit_impulse_and_wrench():
    rng = np.random.default_rng(0); h = make_hit("strike", "chest", 10.0, 0.5, 0.0, rng)
    assert abs(h.impulse - 10.0) < 1e-6 and h.force[0] < 0       # frontal strike pushes robot backward (-x)
    m = build_spec(cfg=CFG).compile(); d = mujoco.MjData(m); R1Robot(m, CFG).reset(m, d)
    HitApplier(m).apply(d, [h], 0.55); assert np.linalg.norm(d.xfrc_applied[m.body("torso_link").id, :3]) > 0
    assert all(sum(x.impulse for x in random_schedule(rng, 10, 5)) > 0 for _ in range(3))

def test_delay_buffer():
    s = DelayedNoisyObs(2, np.zeros(1), np.random.default_rng(0)); s.reset(np.array([0.0]))
    outs = [s(np.array([float(i)]))[0] for i in range(1, 5)]; assert outs == [0.0, 0.0, 1.0, 2.0]

def test_env_shapes_and_pd_survives_no_hits():
    e = R1HitEnv(seed=0, train=False); o = e.reset(hits=[]); assert o.shape == (OBS_DIM,)
    for _ in range(150): o, r, d, info = e.step(np.zeros(ACT_DIM)); assert not info["fallen"]

def test_big_strike_knocks_pd_over():
    e = R1HitEnv(seed=0, train=False); e.reset(hits=[make_hit("strike", "chest", 20.0, 0.3, 0.0, np.random.default_rng(0))], delay=0)
    assert any(e.step(np.zeros(ACT_DIM))[3]["fallen"] for _ in range(150))

def test_batch_env_shares_model():
    b = BatchEnv(4, threads=2); o = b.reset(); assert o.shape == (4, OBS_DIM) and all(e.m is b.envs[0].m for e in b.envs)
    o, r, d, i = b.step(np.zeros((4, ACT_DIM))); assert r.shape == (4,)

def test_opponent_replay_moves_and_touches():
    assert len(list_clips()) >= 1
    m = build_spec(opponent=True, cfg=CFG).compile(); op = FightReplay(m, list_clips()[0])
    q0, q1 = op.qpos_at(0.0), op.qpos_at(1.0); assert q0.shape == (31,) and np.abs(q0[7:] - q1[7:]).max() > 0.05
    assert 0.3 < op.q[:, 0].mean() < 1.0                          # in front of the R1


def test_verify_r1_specs(tmp_path):
    import subprocess, sys, json, pathlib
    root = pathlib.Path(__file__).resolve().parents[1]; out = tmp_path / "r1v.json"
    r = subprocess.run([sys.executable, str(root / "scripts" / "verify_r1.py"), "--out", str(out), "--policies"], capture_output=True, text=True)
    res = json.load(open(out))
    failed = [c["check"] for c in res["checks"] if not c["pass"] and "reference available" not in c["check"]]
    assert not failed, failed


def test_r1_opponent_retarget():
    from r1gym.model import build_spec
    from r1gym.opponent import FightReplay, list_clips
    m = build_spec(opponent=True).compile()
    op = FightReplay(m, list_clips()[0]); assert op.kind == "r1" and op.q.shape[1] == 31


def test_opponent_loop_is_seamless_and_retarget_has_no_flips():
    m = build_spec(opponent=True, cfg=CFG).compile()
    for clip in list_clips():
        op = FightReplay(m, clip, distance=0.6); q = np.vstack([op.q, op.q[:1]])   # includes the wrap back to frame 0
        assert np.abs(np.diff(q[:, :2], axis=0)).max() < 0.05, clip              # no root teleport
        assert np.abs(np.diff(q[:, 7:], axis=0)).max() < 0.4, clip               # 0.25 rad/frame cap (+~0.1 inside the loop cross-fade)
        assert abs(np.median(q[:, 0]) - 0.6) < 0.1 and abs(np.median(q[:, 1])) < 0.1, clip   # stays in front, in range


def test_puppet_opponent_is_a_capped_physical_r1():
    cfg = {**CFG, "opponent_mode": "puppet"}
    m = build_spec(opponent=True, cfg=cfg).compile(); assert m.nu == 48
    caps = torque_caps(cfg); opp = [m.actuator("opp_" + j.replace("_joint", "")).id for j in CTRL_JOINTS]
    assert np.allclose(m.actuator_forcerange[opp, 1], caps)
    from r1gym.scenario import demo_env
    from r1gym.contacts import per_body_force
    e, o = demo_env(m, cfg); op = e.opp; peak = 0.0; assert op.puppet and op.tracker is None
    for _ in range(150):   # 3 s, PD stand
        e.step(np.zeros(ACT_DIM)); peak = max(peak, sum(per_body_force(m, e.d, e.r.r1_bodies).values()))
        assert np.linalg.norm(e.d.qpos[op.qadr:op.qadr + 3] - op.qpos_at(e.d.time)[:3]) < 0.25   # tracks its clip
    assert peak < 3000.0


def test_tracked_opponent_stands_on_its_own_motors():
    """Default opponent: an R1 driven by the tracking policy, Unitree's motors, no assist; it performs a clip it handles."""
    m = build_spec(opponent=True, cfg=CFG).compile()
    e = R1HitEnv(seed=0, cfg=CFG, model=m, episode_s=6, max_impulse=0, train=False)
    e.opp = FightReplay(m, "kfab_200", distance=4.0, cfg=CFG); e.reset(hits=[], delay=0); op = e.opp
    assert op.tracker is not None and not op.puppet and m.actuator_gainprm[op.act[0], 0] == 100
    peak = [0.0]; orig = mujoco.mj_step
    def spy(mm, dd, *a):   # inside the substeps: nothing may push on the opponent's pelvis
        peak[0] = max(peak[0], float(np.abs(dd.xfrc_applied[op.pel]).max())); return orig(mm, dd, *a)
    mujoco.mj_step = spy
    try:
        for _ in range(250):
            e.step(np.zeros(ACT_DIM)); assert e.d.xmat[op.pel][8] > 0.5 and e.d.xpos[op.pel][2] > 0.25
    finally: mujoco.mj_step = orig
    assert peak[0] == 0.0


def test_wilson_and_paired_diff():
    from r1gym.stats import wilson, paired_diff
    lo, hi = wilson(50, 100); assert lo < 0.5 < hi and 0.39 < lo < 0.41
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    d, (a, b) = paired_diff([1, 1, 1, 0] * 25, [0, 1, 0, 0] * 25); assert d == 0.5 and a < 0.5 < b



def test_deploy_loop_matches_env():
    """The documented deploy contract (r1gym/deploy.py) reproduces the env's joint targets exactly, step by step."""
    from r1gym.policy import TorchPolicy
    from r1gym.deploy import DeployLoop
    pol = TorchPolicy("results/policy/policy.pt"); loop = DeployLoop(pol)
    e = R1HitEnv(seed=0, train=False); o = e.reset(hits=[make_hit("strike", "chest", 6.0, 0.4, 0.0, np.random.default_rng(0))], delay=0)
    e.sensor.std = np.zeros_like(e.sensor.std); o = e._raw_obs()   # no sensor noise on either side
    r, d = e.r, e.d
    for _ in range(60):
        tgt = loop.step(d.qvel[r.free_v + 3:r.free_v + 6], d.qpos[r.free_q + 3:r.free_q + 7], d.qpos[r.qadr], d.qvel[r.vadr])
        o, *_ = e.step(pol.act(o[None])[0])
        assert np.allclose(tgt, d.ctrl[r.act], atol=1e-5)


def test_mjlab_contract_config():
    """configs/r1_mjlab.yaml reproduces the unitree_rl_mjlab R1 contract (values from an exported ONNX's metadata),
    and the opponent stays identical to the residual config's."""
    from r1gym.model import action_scale
    from r1gym import CONFIGS
    c = load_cfg(CONFIGS / "r1_mjlab.yaml")
    assert c["contract"] == "mjlab" and c["sensors"]["delay_steps"] == CFG["sensors"]["delay_steps"]   # inherited
    onnx_scale = [0.150] * 4 + [0.312] * 2 + [0.150] * 4 + [0.312] * 2 + [0.150] * 2 + [0.375, 0.375, 0.412, 0.412, 0.412] * 2
    assert np.allclose(action_scale(c), onnx_scale, atol=2e-3)
    m, m0 = build_spec(cfg=c, opponent=True).compile(), build_spec(cfg=CFG, opponent=True).compile()
    assert m.actuator_gainprm[0, 0] == 100 and m.actuator_forcerange[0, 1] == 60            # Unitree gains / caps
    assert np.allclose(m.actuator_forcerange[24:], m0.actuator_forcerange[24:]) and np.allclose(m.actuator_gainprm[24:], m0.actuator_gainprm[24:])
    e = R1HitEnv(seed=0, cfg=c, model=m, train=False); o = e.reset(hits=[]); assert o.shape == (83,)
    e.command = np.array([0.5, 0, 0]); assert np.abs(e._raw_obs()[9:11]).sum() > 0   # gait phase active when moving


def test_stepping_base_runs_in_this_gym():
    """Cross-simulator regression: the mjlab-trained stepping base (MuJoCo Warp) stands and walks forward here (CPU MuJoCo)."""
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller("step=results/policy_mjlab/policy.onnx"); cfg = controller_cfg(c); m = build_spec(cfg=cfg).compile()
    for cmd, check in [((0.0, 0.0, 0.0), lambda x: abs(x) < 0.25), ((0.5, 0.0, 0.0), lambda x: x > 0.2)]:
        e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=5, max_impulse=0, train=False); e.reset(hits=[], delay=1); e.command = np.array(cmd)
        o = e._raw_obs(); x0 = e.d.qpos[0]
        for _ in range(200):
            o, r, d, info = e.step(c.act(o[None])[0]); assert not info["fallen"]
        assert check((e.d.qpos[0] - x0) / 4.0), (cmd, e.d.qpos[0] - x0)


def test_guard_variant_holds_its_guard():
    """The guard variant takes its default pose (a guard) from its ONNX metadata, stands, and keeps its fists up."""
    from r1gym.policy import load_controller, controller_cfg
    c = load_controller("guard=results/policy_mjlab_guard/policy.onnx"); cfg = controller_cfg(c)
    home = np.array(cfg["home"]); assert np.isclose(home[CTRL_JOINTS.index("left_elbow_joint")], -0.7, atol=1e-3)
    m = build_spec(cfg=cfg).compile(); e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=5, max_impulse=0, train=False)
    e.reset(hits=[], delay=1); o = e._raw_obs(); torso = m.body("torso_link").id
    for _ in range(200):
        o, r, d, info = e.step(c.act(o[None])[0]); assert not info["fallen"]
    wrist = e.d.xpos[m.body("left_wrist_roll_link").id] - e.d.xpos[torso]
    assert wrist[2] > 0.1 and wrist[0] > 0.1   # fist up and in front of the torso, not hanging by the hip


def _eval_opponent():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("eval_opponent", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_opponent.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_eval_tiers_draw_paired_episodes():
    """Hard tier: six strikes/pushes per 20 s bout at 8-14 N*s, deterministic, with the env/DR and hit-duration streams
    separate from the hits; start times are drawn from the clean ones only. The standard tier still draws the hits and
    delays it drew before the clean-start filter (only t0 moves)."""
    eo = _eval_opponent(); n = int(round(eo.START_SPAN / eo.START_STEP))
    good = {("kfab_202", d): np.ones(n, bool) for t in eo.TIERS.values() for d in t["distances"]}
    eps = eo.episodes(["kfab_202"], 3, "hard", good)
    assert len(eps) == 12 and all(len(e["hits"]) == 6 for e in eps) and len({e["seed"] for e in eps}) == 12
    imps = [h[2] for e in eps for h in e["hits"]]; assert 8 <= min(imps) and max(imps) <= 14
    assert eps == eo.episodes(["kfab_202"], 3, "hard", good) and len({e["env_seed"] for e in eps} | {e["hit_seed"] for e in eps}) == 24
    assert all(h[3] < 9 for e in eo.episodes(["kfab_202"], 3, "hard", good, seconds=10.0) for h in e["hits"])   # --seconds keeps every hit
    g = good[("kfab_202", 0.4)].copy(); g[:150] = False   # starts before 3 s overlap the R1 -> every bout starts at >= 3 s
    t0s = [e["t0"] for e in eo.episodes(["kfab_202"], 40, "hard", {**good, ("kfab_202", 0.4): g}) if e["distance"] == 0.4]
    assert min(t0s) >= 3.0 and len(set(t0s)) > 30   # spread over the clean starts, not piled on the first one
    e0 = eo.episodes(["kfab_202"], 1, "standard", good)[0]; rng = np.random.default_rng(10_000); hits = []
    for t in sorted(rng.uniform(1.0, 9.0, 2)):   # the generator as it was before the filter
        yaw = rng.choice([np.pi / 2, -np.pi / 2, np.pi]) + rng.uniform(-0.3, 0.3)
        hits.append(("push", str(rng.choice(["l_shoulder", "r_shoulder", "pelvis", "chest"])), float(rng.uniform(4, 6)), float(t), float(yaw)))
    t0 = rng.uniform(0, 6); assert e0["hits"] == hits and e0["delay"] == int(rng.integers(0, 3)) and abs(e0["t0"] - t0) < eo.START_STEP


def test_clean_starts_skip_an_opponent_inside_the_r1():
    """At 0.40 m some clip start times put the opponent's limbs inside the R1; the mask marks exactly those."""
    eo = _eval_opponent(); cfg = {**CFG, "opponent_mode": "policy"}
    good = eo.clean_starts(cfg, ["kfab_202"], [0.4], {"home": eo.REF_POSES["home"]})[("kfab_202", 0.4)]; assert 0 < good.sum() < len(good)
    m = build_spec(opponent=True, cfg=cfg).compile(); e = R1HitEnv(seed=0, cfg=cfg, model=m, episode_s=1, max_impulse=0, train=False)
    e.opp = FightReplay(m, "kfab_202", distance=0.4, cfg=cfg)
    for i in [int(np.flatnonzero(good)[0]), int(np.flatnonzero(~good)[0])]:
        e.opp_t0 = i * eo.START_STEP; e.reset(hits=[], delay=0)
        inside = any((m.geom_bodyid[c.geom1] in e.opp.bodies) != (m.geom_bodyid[c.geom2] in e.opp.bodies) and 0 not in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
                     for c in [e.d.contact[j] for j in range(e.d.ncon)])
        assert inside == (not good[i])


def test_hard_tier_dr_runs_on_private_model_copies():
    """Domain-randomised episodes step on per-thread model copies: the shared model is untouched, the DR cache is cleaned
    up, and the results don't depend on the thread count."""
    from r1gym.policy import PDStand
    from r1gym.env import _DR_NOMINAL
    eo = _eval_opponent(); cfg = {**CFG, "opponent_mode": "policy"}; m = build_spec(opponent=True, cfg=cfg).compile()
    n = int(round(eo.START_SPAN / eo.START_STEP)); good = {("kfab_202", d): np.ones(n, bool) for d in eo.TIERS["hard"]["distances"]}
    eps = eo.episodes(["kfab_202"], 1, "hard", good)
    for e in eps: e["hits"] = [("strike", "chest", 30.0, 0.1, 0.0)] if e["distance"] == 0.4 else []   # one bout ends in a fall
    mass, keys = m.body_mass.copy(), set(_DR_NOMINAL)
    rows = eo.run(PDStand(), eps, m, cfg, T=1.0, threads=2, dr=True)
    assert len(rows) == 4 and np.array_equal(m.body_mass, mass) and set(_DR_NOMINAL) == keys
    assert rows == eo.run(PDStand(), eps, m, cfg, T=1.0, threads=3, dr=True)
    assert rows[0]["fell_s"] is not None and rows[0]["distance"] == 0.4


def test_steered_opponent_stays_in_front_and_the_r1_faces_it():
    """The opponent (steerable tracker) is kept in front of the R1 along the bout's axis at the bout distance instead of
    circling to its side, so the R1 faces it without turning (demo scenario, 6 s). Neither shipped policy uses the turn
    command (results/face_opponent_ab.json): both gym.json files leave it off."""
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.scenario import demo_env
    assert controller_cfg(load_controller("step=results/policy_mjlab/policy.onnx"))["face_opponent"] is False
    c = load_controller("guard=results/policy_mjlab_guard/policy.onnx"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
    m = build_spec(opponent=True, cfg=cfg).compile(); e, o = demo_env(m, cfg, seconds=6.0); op = e.opp
    assert op.tracker.steerable and not e.face
    brg, dist, face = [], [], []
    for _ in range(300):
        o, r, d, info = e.step(c.act(o[None])[0]); assert not info["fallen"]
        v = e.d.xpos[op.pel][:2] - e.d.xpos[e.r.pel][:2]; brg.append(np.degrees(np.arctan2(v[1], v[0]))); dist.append(np.linalg.norm(v))
        R = e.d.xmat[e.r.pel].reshape(3, 3); face.append(abs(np.degrees(np.angle(np.exp(1j * (np.arctan2(v[1], v[0]) - np.arctan2(R[1, 0], R[0, 0])))))))
    # the unsteered (stage-3) opponent ended up at the R1's side in this scenario; the 3 s side push moves the R1 and the
    # opponent follows at <= 0.3 m/s, so the tail is wider than the median
    assert np.median(np.abs(brg)) < 25 and abs(np.median(dist) - 0.65) < 0.15 and np.median(face) < 30, (np.median(np.abs(brg)), np.median(dist), np.median(face))


def test_face_command_turns_toward_the_opponent():
    """r1gym/env.py _face: opponent to the R1's left -> positive yaw-rate command, to its right -> negative, capped at the
    policy's trained range; only for the mjlab contract."""
    from r1gym import CONFIGS
    c = {**load_cfg(CONFIGS / "r1_mjlab.yaml"), "face_opponent": True}; m = build_spec(cfg=c, opponent=True).compile()
    e = R1HitEnv(seed=0, cfg=c, model=m, episode_s=1, max_impulse=0, train=False); e.opp = FightReplay(m, "kfab_200", distance=0.6, cfg={**c, "opponent_mode": "puppet"}); e.reset(hits=[], delay=0)
    adr = m.jnt_qposadr[m.joint("opp_floating_base_joint").id]
    for y, sign in [(1.0, 1), (-1.0, -1), (0.0, 0)]:
        e.d.qpos[adr:adr + 2] = [0.6 if y == 0 else 0.0, y]; mujoco.mj_kinematics(m, e.d); e._face()
        assert np.sign(round(e.command[2], 3)) == sign and abs(e.command[2]) <= e.FACE_MAX
    e2 = R1HitEnv(seed=0, cfg=CFG, model=build_spec(cfg=CFG, opponent=True).compile(), episode_s=1, max_impulse=0, train=False); assert not e2.face
    e.face_band = [0.35, 0.1]; e.turning = False   # deadband: stand still until 20 deg off, then turn until within 6 deg
    for ang, turning in [(0.2, False), (0.5, True), (0.2, True), (0.05, False)]:
        e.d.qpos[adr:adr + 2] = [0.6 * np.cos(ang), 0.6 * np.sin(ang)]; mujoco.mj_kinematics(m, e.d); e._face()
        assert e.turning == turning and (e.command[2] != 0) == turning, (ang, e.turning, e.command[2])


def test_robots_collide_on_their_visible_shells():
    """robot_contact: shell (default): each robot's visual meshes collide with the other robot's only, Unitree's capsules
    stop colliding across robots but still carry the feet and the opponent's self-collision; in the demo the shells
    overlap by under 2 cm (with capsules they passed ~5 cm into each other)."""
    import importlib.util, pathlib
    from r1gym.policy import load_controller, controller_cfg
    from r1gym.scenario import demo_env
    spec = importlib.util.spec_from_file_location("clip_check", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "clip_check.py")
    cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
    c = load_controller("step=results/policy_mjlab/policy.onnx"); cfg = {**controller_cfg(c), "opponent_mode": "policy"}
    m = build_spec(opponent=True, cfg=cfg).compile(); opp = {b for b in range(m.nbody) if m.body(b).name.startswith("opp_")}
    can = lambda a, b: bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
    mesh = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    caps = [g for g in range(m.ngeom) if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH and m.geom_bodyid[g] and (m.geom_contype[g] or m.geom_conaffinity[g])]
    rs, os_ = [g for g in mesh if m.geom_bodyid[g] not in opp], [g for g in mesh if m.geom_bodyid[g] in opp]
    rc, oc = [g for g in caps if m.geom_bodyid[g] not in opp], [g for g in caps if m.geom_bodyid[g] in opp]
    assert len(rs) == len(os_) > 20 and all(can(a, b) for a in rs[:5] for b in os_[:5])
    assert not any(can(a, b) for a in rs + os_ for b in rc + oc + [0]) and not any(can(a, b) for a in rc for b in oc)
    assert all(can(g, 0) for g in rc + oc) and can(oc[0], oc[-1])   # floor contact, opponent self-collision
    ms = build_spec(opponent=True, cfg={**cfg, "robot_contact": "shell"}).compile(); e, o = demo_env(m, cfg, seconds=6.0)
    r = cc.bout(c, cfg, m, ms, e, o, 300); assert not r["fell"] and max(r["overlap"]) < 0.02, max(r["overlap"])


def test_hold_command_walks_back_to_the_spot():
    """r1gym/env.py _hold: once the pelvis is past hold_deadband[0] from the bout's spot, (vx, vy) point back to it in
    the heading frame (inside the trained ranges) until it is within hold_deadband[1]; off without an opponent."""
    from r1gym import CONFIGS
    c = {**load_cfg(CONFIGS / "r1_mjlab.yaml"), "hold_position": True, "hold_deadband": [0.25, 0.08]}; m = build_spec(cfg=c, opponent=True).compile()
    e = R1HitEnv(seed=0, cfg=c, model=m, episode_s=1, max_impulse=0, train=False); e.opp = FightReplay(m, "kfab_200", distance=0.6, cfg={**c, "opponent_mode": "puppet"})
    e.reset(hits=[], delay=0); q = e.r.free_q; spot = e.spot.copy()
    for off, moving in [((0.1, 0.0), False), ((0.0, 0.4), True), ((0.1, 0.0), True), ((0.05, 0.0), False)]:
        e.d.qpos[q:q + 2] = spot + off; e._hold()
        assert e.returning == moving and (np.abs(e.command[:2]).sum() > 0) == moving, (off, e.returning, e.command)
    e.d.qpos[q:q + 2] = spot + (0.0, 0.4); e._hold(); assert e.command[1] < 0 and abs(e.command[0]) < 1e-6   # left of the spot -> step right
    e.d.qpos[q + 3:q + 7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]; e._hold()                          # facing +y: the spot is behind
    assert e.command[0] < 0 and abs(e.command[1]) < 1e-6 and e.command[0] >= e.HOLD_VX[0]
    e.opp = None; e.command[:] = 0; e._hold(); assert not e.command.any()
