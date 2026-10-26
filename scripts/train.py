"""PPO for a hit-robust R1 balance/recovery policy (residual on top of the stand PD), with a strike/push/grab
curriculum and per-episode sensor-delay randomisation. CPU proof run; full-scale GPU command in docs/GPU.md."""
import sys, time, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn as nn
from r1gym.env import BatchEnv, ProcBatchEnv, OBS_DIM, ACT_DIM

class Actor(nn.Module):
    def __init__(s, init_std=0.06):
        super().__init__()
        s.mu = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, ACT_DIM))
        # exploration std in action units (joint target = 0.25 * a). The stiff stand itself falls from white noise at
        # std >= 0.15 (~0.04 rad) with no hits at all (scripts/noise_tolerance.py), so a residual-on-stand policy can't
        # explore much: 0.06 is the largest level with 0/16 falls. v1-v3 used exp(-3.5) ~ 0.03.
        s.logstd = nn.Parameter(torch.full((ACT_DIM,), float(np.log(init_std))))
        nn.init.zeros_(s.mu[-1].weight); nn.init.zeros_(s.mu[-1].bias)      # start == PD stand baseline
    def forward(s, o): return s.mu(o)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--envs", type=int, default=64); ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--out", default="results/policy")
    ap.add_argument("--imp0", type=float, default=4.0); ap.add_argument("--imp_max", type=float, default=40.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=0, help=">0: one process per worker")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--init", default="", help="warm-start from a policy.pt (actor + obs normaliser)")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--opp_prob", type=float, default=0.0, help="fraction of episodes with a fight-clip opponent in contact range")
    ap.add_argument("--dr", type=int, default=0); ap.add_argument("--reward", default="v2"); ap.add_argument("--gap", type=float, nargs=2, default=[1.0, 2.5])
    ap.add_argument("--grow_below", type=float, default=0.25, help="raise max impulse x1.15 when episode fall rate is below this")
    ap.add_argument("--init_std", type=float, default=0.06, help="initial exploration std (action units; joint target = 0.25 * a)")
    ap.add_argument("--reset_std", action="store_true", help="with --init: reset exploration std to --init_std instead of keeping the checkpoint's")
    ap.add_argument("--entropy", type=float, default=0.0, help="entropy bonus coefficient")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed)
    env = (ProcBatchEnv(a.envs, seed=a.seed * 1000, workers=a.workers, max_impulse=a.imp0, opp_prob=a.opp_prob, dr=bool(a.dr), reward=a.reward, gap=tuple(a.gap)) if a.workers else BatchEnv(a.envs, seed=a.seed * 1000, max_impulse=a.imp0, opp_prob=a.opp_prob, dr=bool(a.dr), reward=a.reward, gap=tuple(a.gap)))
    torch.set_num_threads(4 if a.workers else torch.get_num_threads()); obs = env.reset()
    actor = Actor(a.init_std); critic = nn.Sequential(nn.Linear(OBS_DIM, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
    dev = torch.device(a.device); actor.to(dev); critic.to(dev)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=a.lr)
    # running obs normaliser (frozen into the ONNX export)
    mean, var, cnt = np.zeros(OBS_DIM), np.ones(OBS_DIM), 1e-4
    if a.init:
        ck = torch.load(a.init, weights_only=False); actor.load_state_dict(ck["actor"]); mean, var, cnt = ck["mean"], ck["var"], 1e6
        if a.reset_std:
            with torch.no_grad(): actor.logstd.fill_(float(np.log(a.init_std)))
    imp = a.imp0; t0 = time.time(); it = 0; log = []; steps = 0; ep_falls = ep_done = 0
    gamma, lam, clip = 0.99, 0.95, 0.2
    while time.time() - t0 < a.minutes * 60:
        B = {k: [] for k in "o a lp r d v".split()}
        for _ in range(a.horizon):
            bm, bv = obs.mean(0), obs.var(0); n = len(obs); tot = cnt + n; delta = bm - mean
            mean = mean + delta * n / tot; var = (var * cnt + bv * n + delta ** 2 * cnt * n / tot) / tot; cnt = tot
            on = torch.tensor((obs - mean) / np.sqrt(var + 1e-8), dtype=torch.float32, device=dev).clamp(-5, 5)
            with torch.no_grad():
                mu = actor(on); std = actor.logstd.exp(); act = mu + std * torch.randn_like(mu)
                lp = torch.distributions.Normal(mu, std).log_prob(act).sum(-1); v = critic(on).squeeze(-1)
            obs, r, d, info = env.step(act.cpu().numpy())
            for x in info:
                if x["fallen"] or x["timeout"]: ep_done += 1; ep_falls += x["fallen"]
            # time-limit truncation is not a terminal state: bootstrap it with the value estimate (as rsl_rl does);
            # otherwise the critic learns that time running out is like falling, which the obs can't explain
            to = torch.tensor([x["timeout"] for x in info], dtype=torch.float32, device=dev)
            r = torch.tensor(r, device=dev) + gamma * v * to
            for k, val in zip("o a lp r d v".split(), (on, act, lp, r, torch.tensor(d, dtype=torch.float32, device=dev), v)): B[k].append(val)
            steps += a.envs
        on = torch.tensor((obs - mean) / np.sqrt(var + 1e-8), dtype=torch.float32, device=dev).clamp(-5, 5)
        with torch.no_grad(): nv = critic(on).squeeze(-1)
        R, D, V = torch.stack(B["r"]), torch.stack(B["d"]), torch.stack(B["v"]); adv = torch.zeros_like(R); g = 0
        for t in reversed(range(a.horizon)):
            nxt = nv if t == a.horizon - 1 else V[t + 1]
            dl = R[t] + gamma * nxt * (1 - D[t]) - V[t]; g = dl + gamma * lam * (1 - D[t]) * g; adv[t] = g
        ret = adv + V; O = torch.cat(B["o"]); A = torch.cat(B["a"]); LP = torch.cat(B["lp"]); ADV = adv.flatten(); RET = ret.flatten()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        for _ in range(5):
            for idx in torch.randperm(len(O), device=dev).split(max(512, len(O) // 8)):
                mu = actor(O[idx]); dist = torch.distributions.Normal(mu, actor.logstd.exp())
                ratio = (dist.log_prob(A[idx]).sum(-1) - LP[idx]).exp()
                pl = -torch.min(ratio * ADV[idx], ratio.clamp(1 - clip, 1 + clip) * ADV[idx]).mean()
                vl = ((critic(O[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pl + 0.5 * vl - a.entropy * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0); opt.step()
        it += 1
        if ep_done >= 64:   # curriculum on episode fall rate
            fr = ep_falls / ep_done
            if fr < a.grow_below: imp = min(a.imp_max, imp * 1.15)
            elif fr > 0.6: imp = max(a.imp0, imp / 1.1)
            env.set_max_impulse(imp)
            rec = dict(it=it, steps=steps, minutes=round((time.time() - t0) / 60, 2), fall_rate=round(fr, 3), max_impulse=round(imp, 2),
                       mean_rew=round(float(R.mean()), 3), sps=round(steps / (time.time() - t0)), std=round(float(actor.logstd.exp().mean()), 4))
            print(json.dumps(rec), flush=True); log.append(rec); ep_falls = ep_done = 0
        if it % 20 == 0: save(actor, mean, var, out, log)
    save(actor, mean, var, out, log)

def save(actor, mean, var, out, log):
    torch.save({"actor": {k: v.cpu() for k, v in actor.state_dict().items()}, "mean": mean, "var": var}, out / "policy.pt")
    json.dump(log, open(out / "train_log.json", "w"), indent=0)

if __name__ == "__main__": main()
