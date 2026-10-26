import sys;sys.path.insert(0,__import__('os').path.join(__import__('os').path.dirname(__file__),'..'))
import numpy as np, mujoco, copy, itertools
from r1gym.env import R1HitEnv
from r1gym.model import load_cfg, build_spec
base=load_cfg()
def trial(leg,ank,kd,mu,s=0.02):
  c=copy.deepcopy(base); c['stand_pd']['leg']={'kp':leg,'kd':kd}; c['stand_pd']['ankle']={'kp':ank,'kd':kd}
  sp=build_spec(cfg=c); m=sp.compile()
  for g in range(m.ngeom):
    if m.geom_priority[g]==1: m.geom_friction[g,0]=mu
  e=R1HitEnv(seed=3,cfg=c,model=m,train=False); f=0
  for k in range(10):
    e.reset(hits=[],delay=0); e.d.qpos[e.r.qadr]+=e.rng.normal(0,s,24); mujoco.mj_forward(e.m,e.d)
    for t in range(250):
      o,r,d,i=e.step(np.zeros(24))
      if d: f+=1;break
  return f
for leg,ank,kd,mu in []:
  print(leg,ank,kd,mu,trial(leg,ank,kd,mu),flush=True)
print('---')
for leg,ank,kd in [(200,300,5),(300,300,10),(300,400,8),(250,350,6),(300,300,5),(400,400,10)]:
  print(leg,ank,kd,[trial(leg,ank,kd,1.0,s) for s in (0.02,0.04)],flush=True)
