"""Per-body contact logging: which R1 body touched what, with normal force [N]."""
import numpy as np, mujoco

def body_contacts(m, d, bodies_of_interest=None):
    out = []; f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if bodies_of_interest is not None and b1 not in bodies_of_interest and b2 not in bodies_of_interest: continue
        mujoco.mj_contactForce(m, d, i, f6)
        out.append((m.body(b1).name or "world", m.body(b2).name or "world", float(abs(f6[0])), c.pos.copy()))
    return out

def per_body_force(m, d, r1_bodies):
    """dict body_name -> summed normal force from non-floor contacts (hits/grabs from the opponent)."""
    agg = {}
    for a, b, f, _ in body_contacts(m, d):
        for x, y in ((a, b), (b, a)):
            if x != "world" and not x.startswith("opp_") and y.startswith("opp_"): agg[x] = agg.get(x, 0.0) + f
    return agg
