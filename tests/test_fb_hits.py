"""Full-body M2 v2 hit account (M2V2_PLAN_v3 M5; gpu/fb_hits.py): constants = r1gym/hits.py, aim.py and run 2's; MjContactSensors =
MuJoCo's own mjSENS_CONTACT built as mjlab builds it; zone forces and closing speeds; on random synthetic contact logs (per-side keys,
gaps around the merge gap, zero-force touches, closing speeds of both signs) FullBodyHitAccount's episodes, landed / clean calls and
window outcomes equal HitMeter.events / windows run on the same logs, and its HIT_R2 + m2 pay (block-shell-first, run 2's bz) equals the python
reference on the HitMeter's own per-side episodes, in float64 and float32; close_envs scores and resets only its envs."""
import sys
from pathlib import Path
import numpy as np, pytest, torch, mujoco
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gpu")); sys.path.insert(0, str(ROOT / "scripts"))
import fb_hits as HT, fb_m5_check as C
import r1gym.hits as RH

@pytest.fixture(scope="module")
def m(): return C.model()[0]

def test_constants_equal_the_eval():
    import gpu.sparring_task as ST
    from r1gym.aim import AIM
    assert HT.ZONES == RH.ZONES and HT.LIMBS == RH.STRIKE_LIMBS and HT.HIT_P == {k: float(v) for k, v in RH.DEFAULTS.items()}
    assert HT.WIN_TAIL_S == AIM["window_tail_s"] and HT.BLOCK_BY_APEX_S == ST.HIT_R2["block_by_apex_s"] and HT.FORCE_SIGN == ST.FORCE_SIGN and (HT.T_TOL, HT.W_TOL) == (ST.T_TOL, ST.W_TOL)
    assert HT.SENSORS[:4] == ("l_hand", "l_forearm", "l_foot", "l_shin") and [HT.LIMBS[p] for p in HT.PARTNER] == ["forearm", "hand", "shin", "foot"]

def test_zone_and_limb_maps_equal_the_hitmeter(m):
    hm, rv = RH.HitMeter(m), RH.HitMeter(m, reverse=True); s, r = HT.MjContactSensors(m), HT.MjContactSensors(m, reverse=True)
    assert len(s.prim) == len(r.prim) == 32 and len(s.bprim) == sum(HT.ZONES[z] in HT.BODY_ZONES for z in s.zone_idx) == 20
    assert [HT.ZONES[z] for z in s.zone_idx] == [hm.zone_of[g] for g in s.prim] and [HT.ZONES[z] for z in r.zone_idx] == [rv.zone_of[g] for g in r.prim]
    for sens, h in ((s, hm), (r, rv)):
        for g, (sd, l) in h.keys.items():
            assert sens.limb[g] == (HT.SIDES.index(sd) * 4 + HT.LIMBS.index(l) if sd and l in HT.LIMBS else -1), (g, sd, l)
    feet = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom) if s.limb[g] == HT.SENSORS.index("l_foot")]
    assert set(feet) == {"opp_left_ankle_pitch_link", "opp_left_ankle_roll_link"}   # the foot sensor's secondary is the ankle_pitch subtree
    import gpu.sparring_task as ST
    blk = sorted(m.body(m.geom_bodyid[g]).name for g, b in zip(s.prim, s.block_idx) if b); assert blk == sorted(dict(ST.SHELLS)["block"])   # run 2's 'block' zone

def _contact_states(m, n=6, seed=0):
    """Two R1s overlapping (opponent 0.30-0.36 m away, turned to face, random arm and leg joints): states with robot contacts."""
    rng = np.random.default_rng(seed); d = mujoco.MjData(m); hm = RH.HitMeter(m); out = []; oq = m.jnt_qposadr[m.body("opp_pelvis").jntadr[0]]
    for _ in range(5000):
        if len(out) >= n: break
        q = m.qpos0.copy(); q[2] = 0.74; q[oq:oq + 3] = (rng.uniform(0.28, 0.36), rng.uniform(-0.05, 0.05), 0.74); q[oq + 3:oq + 7] = (0, 0, 0, 1)
        for j in range(m.njnt):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE: lo, hi = m.jnt_range[j]; q[m.jnt_qposadr[j]] = rng.uniform(lo, hi) * 0.6
        d.qpos[:] = q; d.qvel[:] = rng.normal(0, 0.5, m.nv); mujoco.mj_forward(m, d); g1, g2 = d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]
        k = (hm.is_r1[g1] & hm.is_opp[g2]) | (hm.is_r1[g2] & hm.is_opp[g1])
        if k.sum() >= 3: out.append((q.copy(), d.qvel.copy()))
    return out

def test_mj_sensors_equal_mujoco_contact_sensor(m):
    """mjlab's construction (one sensor per primary geom and field, reduce netforce, num_slots 1; secondary body / subtree) on the eval
    model with named shells: found and force equal ours exactly, for the 8 limb sensors and the body sensor."""
    import fb_m4_check as C4
    _, cfg, _, _ = C4.model_cfg(); from r1gym.model import build_spec
    sp = build_spec(opponent=True, cfg=cfg)
    for g in sp.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and (g.contype, g.conaffinity) == (0, 4): g.name = g.meshname + "_shell"
    s = HT.MjContactSensors(m); names = [m.mesh(m.geom_dataid[g]).name + "_shell" for g in s.prim]
    sec = {f"{sd}_{l}": (mujoco.mjtObj.mjOBJ_XBODY if HT.LIMB_SECONDARY[l][0] == "subtree" else mujoco.mjtObj.mjOBJ_BODY, f"opp_{'left' if sd == 'l' else 'right'}_{HT.LIMB_SECONDARY[l][1]}")
           for sd in HT.SIDES for l in HT.LIMBS}
    for k, (rt, rn) in {**sec, "body": (mujoco.mjtObj.mjOBJ_XBODY, "opp_pelvis")}.items():
        for p, nm in enumerate(names):
            if k == "body" and s.prim[p] not in s.bprim: continue
            for f, bit in (("found", 0), ("force", 1)):
                sp.add_sensor(name=f"{k}/{nm}/{f}", type=mujoco.mjtSensor.mjSENS_CONTACT, objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=nm, reftype=rt, refname=rn, intprm=[1 << bit, 3, 1])
    M = sp.compile(); d = mujoco.MjData(M); n_hit = n_body = 0
    for q, v in _contact_states(m):
        d.qpos[:] = q; d.qvel[:] = v; mujoco.mj_forward(M, d); F, fo, _, _, Fb, _ = s.read(d)
        for l, k in enumerate(HT.SENSORS):
            for p, nm in enumerate(names):
                assert d.sensor(f"{k}/{nm}/found").data[0] == fo[l, p] and np.array_equal(d.sensor(f"{k}/{nm}/force").data, F[l, p]); n_hit += int(fo[l, p] > 0)
        for p, g in enumerate(s.bprim):
            nm = names[s.prim.index(g)]; assert np.array_equal(d.sensor(f"body/{nm}/force").data, Fb[p]); n_body += int(d.sensor(f"body/{nm}/found").data[0] > 0)
    assert n_hit >= 3 and n_body >= 3, (n_hit, n_body)

def test_zone_forces_and_closing_speeds():
    zi = [0, 0, 1]; F = torch.zeros(1, 1, 3, 1, 3, dtype=torch.float64); F[0, 0, 0, 0] = torch.tensor([-3.0, 0, 0]); F[0, 0, 1, 0] = torch.tensor([0, -4.0, 0])
    Fz, Tz, st = HT.zone_forces(F, zi); assert Fz[0, 0, :, 0].tolist() == [7.0, 0.0, 0.0, 0.0, 0.0] and Tz[0, 0, :, 0].tolist() == [True] + [False] * 4 and st[0, 0, 0, 0] == 1 and st[0, 0, 1, 0] == -1
    v = torch.zeros(1, 1, 3, 1, 3, dtype=torch.float64); v[0, 0, 0, 0] = torch.tensor([2.0, 0, 0]); v[0, 0, 1, 0] = torch.tensor([0, 1.5, 0])
    vc = HT.closing_speeds(F, v, st); assert vc[0, 0, 0, 0] == 1.5 and vc[0, 0, 1, 0] == 0   # the strongest primary (4 N): moving into it at 1.5 m/s
    v[0, 0, 1, 0] = torch.tensor([0, -1.5, 0]); assert HT.closing_speeds(F, v, st)[0, 0, 0, 0] == -1.5 and HT.closing_speeds(F, v, st, signed=False)[0, 0, 0, 0] == 1.5
    fo = torch.zeros(1, 1, 3, 1); fo[0, 0, 2, 0] = 1; assert HT.zone_forces(torch.zeros_like(F), zi, fo)[1][0, 0, 1, 0]   # a zero-force contact still touches

# ---- the account vs HitMeter on random contact logs
def _random_logs(n_env, steps, rng):
    """Per env a HitMeter log over steps x 4 substeps: bursts per (side, limb, zone), forces 0-60 N (some exactly 0 after the first
    touch: a zero netforce has no direction, so a sensor cannot give a zero-force first touch a closing speed; G0-11 measures it on
    real bouts), gaps 0.02-0.2 s, closing speed at each key's first touch in [-1, 4] m/s; and intents (t_on, t_apex, side's hand).
    An arm-zone touch lands on a block shell (primary 3), an upper-arm shell (primary 5) or both: blks {(sensor, t)} of block touches."""
    logs, wins, blks = [], [], []
    for _ in range(n_env):
        on = {}
        for side in HT.SIDES:
            for l in HT.LIMBS:
                for z in HT.ZONES:
                    if rng.random() < 0.55: continue
                    k = int(rng.integers(1, 30)); sub = set()
                    while k < steps * 4:
                        L = int(rng.integers(1, 36)); sub |= set(range(k, min(k + L, steps * 4 + 1))); k += L + int(rng.integers(4, 40))
                    on[(side, l, z)] = sub
        log, prev, blk = [], set(), {}
        for k in range(1, steps * 4 + 1):
            acc, vc = {}, {}
            for key, sub in on.items():
                if k in sub:
                    acc[key] = 0.0 if key in prev and rng.random() < 0.08 else float(rng.uniform(0, 60) * (rng.random() < 0.7) + rng.uniform(0.01, 3))
                    if key not in prev: vc[key] = round(float(rng.uniform(-1, 4)), 3)
                    if key[2] == "arm": blk[(key, round(k * HT.DT, 4))] = int(rng.integers(0, 3))   # 0 block, 1 upper arm, 2 both
            prev = set(acc)
            if acc: log.append((round(k * HT.DT, 4), acc, vc))
        logs.append(log); blks.append(blk)
        W = []
        for _ in range(int(rng.integers(3, 9))):
            t_on = round(float(rng.uniform(0.0, steps * 0.02 - 0.5)), 3); W.append((t_on, round(t_on + float(rng.uniform(0.1, 0.4)), 3), str(rng.choice(["l_hand", "r_hand"]))))
        wins.append(sorted(W))
    return logs, wins, blks

def _account_on_logs(logs, wins, blks, steps, dtype):
    N, K = len(logs), max(len(w) for w in wins); zi = list(range(5)) + [3]; log = []
    acc = HT.FullBodyHitAccount(N, zi, slots=K, dtype=dtype, log=log, block=[False, False, False, True, False, False])
    ids, on, ap, sd = (torch.full((N, K), -1, dtype=torch.long), torch.zeros(N, K, dtype=dtype), torch.zeros(N, K, dtype=dtype), torch.zeros(N, K, dtype=torch.long))
    for i, W in enumerate(wins):
        for k, (a, b, l) in enumerate(W): ids[i, k], on[i, k], ap[i, k], sd[i, k] = k, a, b, HT.SIDES.index(l[0])
    by_t = [{t: (a, v) for t, a, v in L} for L in logs]
    for s in range(steps):
        F, V, fo = torch.zeros(N, 8, 6, 4, 3, dtype=dtype), torch.zeros(N, 8, 6, 4, 3, dtype=dtype), torch.zeros(N, 8, 6, 4, dtype=dtype)
        for i in range(N):
            for j in range(4):
                x = by_t[i].get(round((4 * s + j + 1) * HT.DT, 4))
                if x is None: continue
                for (side, l, z), f in x[0].items():
                    c = HT.SIDES.index(side) * 4 + HT.LIMBS.index(l); zz = HT.ZONES.index(z); w = blks[i].get(((side, l, z), round((4 * s + j + 1) * HT.DT, 4)), 0)
                    for pp, share in ((zz, 1.0),) if z != "arm" else (((3, 1.0),), ((5, 1.0),), ((3, 0.5), (5, 0.5)))[w]:
                        F[i, c, pp, j, 0] = -f * share; fo[i, c, pp, j] = 1; V[i, c, pp, j, 0] = x[1].get((side, l, z), 0.0)
        acc.process(F, torch.full((N,), 0.02 * (s + 1), dtype=torch.float64), v_rel=V, found=fo, win=dict(id=ids, on=on, apex=ap, side=sd))
    acc.close_envs(list(range(N))); return log

@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_account_equals_hitmeter_on_random_logs(m, dtype):
    rng = np.random.default_rng(7); steps = 150; logs, wins, blks = _random_logs(24, steps, rng); log = _account_on_logs(logs, wins, blks, steps, dtype)
    hm = RH.HitMeter(m); n_ep = n_win = n_pay = n_up = 0
    for i, (L, W) in enumerate(zip(logs, wins)):
        hm.log, hm._ev = L, {}
        ref = sorted((e["limb"], e["zone"], e["t0"], hm.landed(e), hm.clean(e)) for e in hm.events() if e["limb"] in HT.LIMBS)
        ours = sorted((e["limb"], e["zone"], round(e["t0"], 4), e["landed"], e["clean"]) for e in log if e["kind"] == "episode" and e["env"] == i)
        assert ours == ref, i
        imp = {(e["limb"], e["zone"], e["t0"]): e["impulse_Ns"] for e in hm.events()}
        for e in log:
            if e["kind"] == "episode" and e["env"] == i: assert abs(imp[(e["limb"], e["zone"], round(e["t0"], 4))] - e["impulse_Ns"]) < (1e-4 if dtype == torch.float64 else 1e-3)
        oc = hm.windows([(a, round(b + HT.WIN_TAIL_S, 4), l) for a, b, l in W]); ev = hm.events(by_side=True); mine = {x["id"]: x for x in log if x["kind"] == "window" and x["env"] == i}
        assert [mine[k]["outcome"] for k in range(len(W))] == oc, i
        ab = lambda sd, g, t0: blks[i].get(((sd, g, "arm"), round(t0, 4)), 1) in (0, 2)   # a block shell touched as the arm episode opened (run 2 bz)
        assert [mine[k]["paid"] for k in range(len(W))] == [C.pay_ref(ev, a, b, l, hm.p, ab) for a, b, l in W], i
        n_ep += len(ref); n_win += len(W); n_pay += sum(mine[k]["paid"] for k in range(len(W))); n_up += sum(C.pay_ref(ev, a, b, l, hm.p) and not mine[k]["paid"] for k, (a, b, l) in enumerate(W))
    assert n_ep > 500 and n_win > 100 and n_pay > 0 and n_up > 0, (n_ep, n_win, n_pay, n_up)   # n_up: upper-arm-first windows unpaid

def test_close_envs_scores_and_resets_only_its_envs():
    acc = HT.FullBodyHitAccount(2, list(range(5)), dtype=torch.float64, log=[]); F = torch.zeros(2, 8, 5, 4, 3, dtype=torch.float64); F[:, 0, 0, :, 0] = -50.0
    V = torch.zeros_like(F); V[:, 0, 0, :, 0] = 2.0; acc.process(F, torch.full((2,), 0.02, dtype=torch.float64), v_rel=V)
    assert float(acc.land_t[0]) == 0.02 and acc.A.open[:, 0, 0].all(); acc.close_envs([1])
    E = [e for e in acc.log if e["kind"] == "episode"]; assert len(E) == 1 and E[0]["env"] == 1 and E[0]["clean"] and E[0]["impulse_Ns"] == 1.0
    assert bool(acc.A.open[0, 0, 0]) and not bool(acc.A.open[1].any()) and float(acc.land_t[1]) == -np.inf and float(acc.ep["clean"][1].sum()) == 0
