#!/bin/bash
# Full-body M2 v2 on the GPU pod (M2V2_PLAN_v3 section 4: P0-P7; m12, m13, B4, B6; A8-A9). One RTX 4090 (or L40S), >= 16 vCPU.
#   bash pod_fb.sh plan      the stages, the uploads and the taskset plan; with FB_DRY=1 also every stage's commands, printed not run
#   bash pod_fb.sh P0 | P1 | P2 | P3 | P456 | P7 | all     a failed gate stops (fail handling: plan 3.2)
#   FB_REPO=. bash gpu/pod_fb.sh md5 <file>   (locally, at the prereg commit) the uploads' md5 list P0 checks ($FB_ROOT/uploads.md5 on the pod)
# Env: FB_REPO (the repo uploaded at the prereg commit, default /root/repo), FB_ROOT (work dir, /root), FB_CPUS (nproc), FB_SOLVER /
# FB_HULLS (M6's choice; P0 times the fallback lists), FB_TOPP (T-opp baseline per live s, M6; WATCH6 E6), FB_TWO_PROC=1 (P0 showed
# >= 80% per-process throughput with two processes per GPU), FB_DRY=1 (print every command instead of running it), FB_FAKE_GPU=1 (no
# nvidia-smi: the M9 dry run), PREREG_SHA16 (prereg_m2_fb.json's LF sha256[:16], M11).
# Taskset (m13): training on cores 0..T-1 (two processes split them), background CPU evals (E7 probes, T1-c, LOCO evals) on T..N-1,
# T = N / 2: disjoint, printed as 'PIN train <cpus>' / 'PIN bg <cpus>'.
# Asserts (P0): uploads md5 (md5sum -c uploads.md5); versions (mujoco, mujoco-warp, warp-lang, torch, mjlab, rsl-rl-lib) recorded vs
# local mujoco 3.17.0 / mjlab 1.2.0; assert_sim: the task's MujocoCfg = SIM + FB_SOLVER, njmax 600, nconmax 128, and tf32 off
# (fb_opponent.pin_precision + precision_ok); the M9 model diff rerun with the pod's mujoco (FB_POD=1 scripts/fb_m9_check.py).
# Evaluations (M10): scripts/fb_p1.py (P1; the R0-EO pod reference at $FB_R0EO, default $W/r0eo_pod, A8.2), fb_t1c.py (P2's T1-c,
# both arms on this pod's CPU, A8.2), fb_loco_eval.py (P3; start and G re-run here beside the pilots, A8.2).
set -u
R=${FB_REPO:-/root/repo}; DRY=${FB_DRY:-0}; FAKE=${FB_FAKE_GPU:-0}; W=${FB_ROOT:-/root}; [ "$DRY" = 1 ] && W=${TMPDIR:-/tmp}/fb_pod_dry
[ "${1:-}" = md5 ] || mkdir -p $W; P=$W/progress_fb.txt; OUT=$W/out_fb; NE=${FB_NUM_ENVS:-4096}; N=${FB_CPUS:-$(nproc)}
T=$(( N / 2 )); TRAIN_CPUS="0-$(( T - 1 ))"; BG_CPUS="$T-$(( N - 1 ))"; TRAIN_A="0-$(( T / 2 - 1 ))"; TRAIN_B="$(( T / 2 ))-$(( T - 1 ))"
SEEDS="42 43 44"; CHECKS="50 100 $(seq -s ' ' 200 100 4000)"                 # = scripts/fb_watch.py W6 checks
ITERS_A0=100; ITERS_A=1500; ITERS_B=4000; LOCO_A0=100; LOCO_A=800; LOCO_B=1100  # plan 2.5 / P3
U=$W/unitree_rl_mjlab; RUNS=$U/logs/rsl_rl/r1_fullbody; HULLS_LIST="${FB_HULLS:-mix} all256 full"
UPLOADS="results/policy_mjlab_guard_footwork_r/model_7496.pt results/policy_mjlab_guard_footwork_r/policy.onnx results/policy_mjlab_guard_footwork_r/gym.json
 results/policy_mjlab_spar3_a3s/model_8995.pt results/policy_mjlab_spar3_a3s/policy.onnx results/policy_mjlab_spar3_blind3/model_8995.pt results/policy_mjlab_spar3_blind3/policy.onnx
 results/policy_mjlab_spar3_a3/model_8995.pt results/opponent_tracker/model_13749.pt results/opponent_tracker/aim.json
 gpu/motions/fb/fb_motion.npz gpu/motions/fb/fb_l1.npz gpu/motions/fb/fb_l1_loco.npz gpu/motions/fb/fb_hulls.npz gpu/motions/fb/fb_l1_track.csv gpu/motions/fb/fb_l1_track_index.json
 gpu/motions/fb/fb_l1_track_loco.csv gpu/motions/fb/fb_l1_track_loco_index.json results/sparring/prereg_m2_fb.json"
act() { [ -f $W/venv/bin/activate ] && . $W/venv/bin/activate; return 0; }   # every python after P0's install runs in the pod venv (any stage)
act
log() { echo "$* $(date -u +%T)" >> $P; echo "$*"; }
run() { if [ "$DRY" = 1 ]; then echo "RUN $*"; else "$@"; fi; }                 # every heavy command goes through run
bgrun() { run taskset -c $BG_CPUS "$@"; }
pyq() { if [ "$DRY" = 1 ]; then echo "(dry)"; else python -c "$1" 2>/dev/null; fi; }   # read-only queries (versions, checkpoint fields)
fail() { log "FAILED $1"; pack; exit 1; }
pack() { if [ "$DRY" = 1 ]; then echo "RUN pack $OUT -> m2fb.tar + m2fb.md5"; return; fi; mkdir -p $OUT; cp $W/*.log $W/*.json $P $OUT/ 2>/dev/null; (cd $OUT && tar cf ../m2fb.tar . && cd .. && md5sum m2fb.tar > m2fb.md5); }
cdu() { cd $U 2>/dev/null || [ "$DRY" = 1 ] || fail no_unitree_checkout; }
last() { if [ "$DRY" = 1 ]; then echo "$1/model_LAST.pt"; else f=$(ls -t $RUNS/*_$1/model_*.pt | head -1); echo "$(basename $(dirname $f))/$(basename $f)"; fi; }

plan() {
  echo "STAGE P0 setup, uploads + md5, versions, assert_sim, tf32 off, M9 model diff (pod mujoco), start checkpoints, timing smoke: hulls {$HULLS_LIST} x envs {4096, 8192} x solver {M6, 25/30} + a two-process test, 20 it each"
  echo "STAGE P1 parity: T0 alone, restricted 0.3 s, P1-R4g (12288 worlds x 1000 steps x G, H, A3 vs R0-EO), P1-Hit (32 worlds), B3 asserts, 3-it rsl_rl smoke"
  echo "STAGE P2 T1 (R1-Track-Spar-Reangle: the reangle family only, A9.3) 4000 it, T1-LOCO 3000 it, T1-c on the pod CPU (bg cores)"
  echo "STAGE P3 LOCO pilot sighted + blind: A0 $LOCO_A0 + A $LOCO_A + B $LOCO_B, evals on bg cores (88000-89151 / 89500-90651)"
  echo "STAGE P456 seeds $SEEDS: sighted + blind, A0 $ITERS_A0 + A $ITERS_A + B $ITERS_B, WATCH6 (fb_watch.py --rule 6) at $(echo $CHECKS | wc -w) checks, E7 probes on 63000-63383 (bg)"
  echo "STAGE P7 pack, md5"
  echo "PIN train $TRAIN_CPUS"; echo "PIN train-a $TRAIN_A"; echo "PIN train-b $TRAIN_B"; echo "PIN bg $BG_CPUS"
  for f in $UPLOADS; do echo "UPLOAD $f"; done
}

gpu_ok() {
  if [ "$FAKE" = 1 ]; then log "gpu FAKE (FB_FAKE_GPU=1): no nvidia-smi"; return 0; fi
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $P || fail no_gpu
}

assert_sim() {   # B4 + m12 on the task's own cfg objects (pod: the real mjlab)
  if [ "$DRY" = 1 ]; then echo "RUN assert_sim (fullbody_task.sim_cfg = SIM + FB_SOLVER, njmax 600, nconmax 128; precision_ok)"; return 0; fi
  python -I - > $W/assert_sim.log 2>&1 <<'EOF' || fail assert_sim
import sys; sys.path[:0] = ["gpu", "scripts"]
import fullbody_task as FT, fb_opponent as FO
c = FT.sim_cfg(); m = c.mujoco; it, ls = FT.solver_cfg()
assert (m.timestep, m.integrator, m.cone, m.impratio, m.ccd_iterations) == (0.005, "implicitfast", "pyramidal", 1.0, 35), m
assert (m.iterations, m.ls_iterations) == (it, ls) and (it, ls) in FT.SOLVER_FALLBACK and c.njmax == 600 and c.nconmax == 128, (c, it, ls)
FO.pin_precision(); assert FO.precision_ok(), "tf32 not off"
print("ASSERT_SIM ok", m.timestep, m.integrator, m.cone, m.impratio, m.ccd_iterations, it, ls, c.njmax, c.nconmax, "tf32 off")
EOF
  log "$(cat $W/assert_sim.log | tail -1)"
}

P0() {
  log "P0 start repo $R cpus $N train $TRAIN_CPUS bg $BG_CPUS"; gpu_ok; cd $R || fail no_repo
  for f in $UPLOADS gpu/fullbody_task.py gpu/fb_population.py scripts/fb_watch.py scripts/fb_m9_check.py; do [ -f $f ] || [ "$DRY" = 1 ] || fail "missing_$f"; done
  run md5sum -c $W/uploads.md5 > $W/md5.log 2>&1 || fail uploads_md5
  H=$( [ -f results/sparring/prereg_m2_fb.json ] && tr -d '\r' < results/sparring/prereg_m2_fb.json | sha256sum | cut -c1-16 || echo none ); log "prereg sha256_16 $H"
  [ "$DRY" = 1 ] || { [ -n "${PREREG_SHA16:-}" ] && [ "$H" = "$PREREG_SHA16" ] || fail "prereg_hash_${H}_expected_${PREREG_SHA16:-unset}"; }
  run bash -c "cd $W && git clone -q https://github.com/unitreerobotics/unitree_rl_mjlab && cd unitree_rl_mjlab && git checkout -q 1425b15" || fail clone
  run bash -c "uv venv -q -p 3.11 $W/venv && . $W/venv/bin/activate && cd $U && uv pip install -q -e . && uv pip install -q 'mujoco==3.17.0' onnx 'onnxruntime==1.31.0' tensorboard" > $W/setup.log 2>&1 || fail install
  act
  log "versions $(pyq 'import importlib.metadata as m; print(*[f"{p} {m.version(p)}" for p in ("mujoco", "mujoco-warp", "warp-lang", "torch", "mjlab", "rsl-rl-lib")])') (local: mujoco 3.17.0, mjlab 1.2.0)"
  assert_sim
  FB_POD=1 run python -I scripts/fb_m9_check.py --only model,inputs --out $W/m9_pod.json > $W/m9_pod.log 2>&1 || fail m9_model_diff_on_pod_mujoco
  # the fullbody task runs from the repo ($R/gpu/fullbody_task.py, cwd $U: unitree's train.py, logs/), never a copy (its imports and
  # data are the repo's); only the T1 tracker task files go next to unitree's scripts (P2), with T0 staged as a run dir and the L1 motions converted
  for f in gpu/r1_track_task.py gpu/r1_track_spar_cfg.py gpu/r1_track_spar_t1.py; do run cp $f $U/scripts/; done
  run mkdir -p $U/logs/rsl_rl/r1_tracking_spar/t0 $U/motions; run cp results/opponent_tracker/model_13749.pt $U/logs/rsl_rl/r1_tracking_spar/t0/
  for m in fb_l1_track fb_l1_track_loco; do
    run bash -c "cd $U && python scripts/r1_track_spar_cfg.py convert $R/gpu/motions/fb/$m.csv motions/$m.npz && cp $R/gpu/motions/fb/${m}_index.json motions/" > $W/convert_$m.log 2>&1 || fail convert_$m
  done
  for b in sighted blind; do   # A0 warm starts (actor = the start with its blind flag; critic widened by PRIV zero columns) and the donors' arm std (WATCH6 E0)
    run python -I gpu/fullbody_task.py start $([ $b = blind ] && echo --blind) --out $RUNS/start_$b/model_0.pt > $W/start_$b.log 2>&1 || fail start_$b
    pyq "import torch; print(float(torch.load('$RUNS/start_$b/model_0.pt', map_location='cpu', weights_only=False)['actor_state_dict']['std_param'][14:].mean()))" > $W/donor_std_$b.txt
  done
  cdu; ok=0   # training runs from the unitree checkout (its train.py; logs/rsl_rl = $RUNS)
  for h in $HULLS_LIST; do for ne in 4096 8192; do for sv in "${FB_SOLVER:-10,20}" 25,30; do
    FB_HULLS=$h FB_SOLVER=$sv FB_STAGE=A0 run taskset -c $TRAIN_CPUS python $R/gpu/fullbody_task.py R1-Fb-A --env.scene.num-envs $ne --agent.max-iterations 20 \
      --agent.resume True --agent.load-run start_sighted --agent.load-checkpoint model_0.pt --agent.run-name p0_${h}_${ne}_${sv/,/_} > $W/p0_${h}_${ne}_${sv/,/_}.log 2>&1 && ok=$(( ok + 1 )) || log "P0 timing failed $h $ne $sv"
  done; done; done
  FB_STAGE=A0 run bash -c "taskset -c $TRAIN_A python $R/gpu/fullbody_task.py R1-Fb-A --env.scene.num-envs $NE --agent.max-iterations 20 --agent.resume True --agent.load-run start_sighted --agent.load-checkpoint model_0.pt --agent.run-name p0_two_a > $W/p0_two_a.log 2>&1 &
    taskset -c $TRAIN_B python $R/gpu/fullbody_task.py R1-Fb-A --env.scene.num-envs $NE --agent.max-iterations 20 --agent.resume True --agent.load-run start_sighted --agent.load-checkpoint model_0.pt --agent.run-name p0_two_b > $W/p0_two_b.log 2>&1; wait"
  [ "$DRY" = 1 ] || [ $ok -gt 0 ] || fail p0_no_timing_run_completed
  [ "$DRY" = 1 ] || ! grep -aqiE "overflow|nan detected|maxmatch" $W/p0_*.log || fail p0_overflow_nan_maxmatch
  log "P0 done: pick the most env-steps/s; > 6 s/it: a second pod, > 9 s/it: time an H100 / L40S (throughput rules, section 4)"
}

P1() { log "P1 parity"; cd $R; FB_ROOT=$W FB_UNITREE=$U run python -I scripts/fb_p1.py --all --out $W/p1.json --r0eo ${FB_R0EO:-$W/r0eo_pod} --workers $(( N - T )) > $W/p1.log 2>&1 || fail p1; }

P2() {
  log "P2 T1 (reangle family only, A9.3) and T1-LOCO"; cdu
  run taskset -c $TRAIN_CPUS python scripts/r1_track_spar_t1.py R1-Track-Spar-Reangle --env.scene.num-envs $NE --agent.max-iterations 4000 --motion-file motions/fb_l1_track.npz \
    --agent.resume True --agent.load-run t0 --agent.load-checkpoint model_13749.pt --agent.run-name t1 > $W/t1.log 2>&1 || fail t1
  cd $R; FB_ROOT=$W bgrun python -I scripts/fb_t1c.py --workers $(( N - T )) --out $W/t1c.json > $W/t1c.log 2>&1 || log "T1-c failed: 100% T0 (plan 2.2)"
  cdu; run taskset -c $TRAIN_CPUS python scripts/r1_track_spar_t1.py R1-Track-Spar-Reangle-LOCO --env.scene.num-envs $NE --agent.max-iterations 3000 --motion-file motions/fb_l1_track_loco.npz \
    --agent.resume True --agent.load-run t0 --agent.load-checkpoint model_13749.pt --agent.run-name t1_loco > $W/t1_loco.log 2>&1 || fail t1_loco
}

train() {   # $1 task, $2 run name, $3 seed, $4 stage (A0 / A / B), $5 iterations, $6 resume <run dir>/<ckpt>, $7 cpus
  [ "$DRY" = 1 ] && echo "RUN FB_STAGE=$4 FB_SEED=$3 taskset -c $7 python fullbody_task.py $1 --agent.max-iterations $5 --agent.seed $3 --agent.load-run $(dirname $6) --agent.load-checkpoint $(basename $6) --agent.run-name $2" && return 0
  cdu; FB_STAGE=$4 FB_SEED=$3 run taskset -c $7 python $R/gpu/fullbody_task.py $1 --env.scene.num-envs $NE --agent.max-iterations $5 --agent.logger tensorboard \
    --agent.seed $3 --agent.resume True --agent.load-run $(dirname $6) --agent.load-checkpoint $(basename $6) --agent.run-name $2 > $W/$2.log 2>&1
}
nitr() { grep -ac 'Iteration time:' $W/$1.log 2>/dev/null || echo 0; }
watched() {   # $1 task $2 run $3 seed $4 stage $5 iters $6 resume $7 cpus $8 blind (0/1) [$9 the A run, stage B]; 0 done, 3 stopped
  local extra="--start-it $(pyq "import torch; print(torch.load('$RUNS/$6', map_location='cpu', weights_only=False)['iter'])")" pid rc at
  if [ "$4" = B ]; then extra="$extra --a-final-log $W/$9.log"; else extra="$extra --donor-arm-std $(cat $W/donor_std_$([ $8 = 1 ] && echo blind || echo sighted).txt 2>/dev/null || echo dry)"; fi
  [ "$8" = 1 ] && extra="$extra --blind"
  if [ "$DRY" = 1 ]; then train "$1" "$2" "$3" "$4" "$5" "$6" "$7"; echo "RUN fb_watch.py watch --rule 6 --log $W/$2.log --stage $4 --topp \$FB_TOPP $extra at each of $CHECKS <= $5"; return 0; fi
  train "$1" "$2" "$3" "$4" "$5" "$6" "$7" & pid=$!
  for at in $CHECKS; do
    [ $at -le $5 ] || break
    while kill -0 $pid 2>/dev/null && [ "$(nitr $2)" -lt $at ]; do sleep 20; done
    [ "$(nitr $2)" -ge $at ] || break
    python -I $R/scripts/fb_watch.py watch --rule 6 --log $W/$2.log --at $at --stage $4 --topp ${FB_TOPP:?FB_TOPP} $extra --out $W/watch_$2_$at.json > /dev/null 2>&1; rc=$?
    log "watch $2 at $at rc $rc"
    if [ $rc -eq 2 ]; then kill $pid; fail "watch_config_error_$2_$at"; fi
    if [ $rc -eq 3 ] && [ "$8" = 0 ]; then log "EARLY_STOP $2 at $at"; kill $pid; wait $pid; return 3; fi
  done
  wait $pid
}
one() {   # $1 seed, $2 tag, $3 A0 $4 A $5 B iterations, $6 task prefix, $7 blind (0/1), $8 cpus: A0 -> A -> B restarts of one twin
  local sfx=$([ $7 = 1 ] && echo -Blind) nm=$2_s$1$([ $7 = 1 ] && echo _blind) st=start_$([ $7 = 1 ] && echo blind || echo sighted)/model_0.pt
  watched $6-A$sfx ${nm}_A0 $1 A0 $3 $st $8 $7 && watched $6-A$sfx ${nm}_A $1 A $4 "$(last ${nm}_A0)" $8 $7 \
    && watched $6-B$sfx ${nm}_B $1 B $5 "$(last ${nm}_A)" $8 $7 ${nm}_A || log "STOPPED ${nm}: export _stopped (a stopped B falls back to the A final)"
}
pair() {   # sighted + blind twin; concurrently on the two halves of the training cores only with FB_TWO_PROC=1 (P0's two-process test)
  if [ "${FB_TWO_PROC:-0}" = 1 ]; then one $1 $2 $3 $4 $5 $6 0 $TRAIN_A & one $1 $2 $3 $4 $5 $6 1 $TRAIN_B & wait
  else one $1 $2 $3 $4 $5 $6 0 $TRAIN_CPUS; one $1 $2 $3 $4 $5 $6 1 $TRAIN_CPUS; fi
}
P3() { log "P3 LOCO pilot"; pair 42 loco $LOCO_A0 $LOCO_A $LOCO_B R1-Fb-LOCO; cd $R; FB_ROOT=$W bgrun python -I scripts/fb_loco_eval.py --workers $(( N - T )) --out $W/loco.json > $W/loco_eval.log 2>&1 || { log "LOCO eval: $(tail -1 $W/loco_eval.log)"; fail loco; }; }   # P3's bars gate P4-P6 (plan 4: proceed iff all four)
P456() { log "P4-P6 main pairs"; for s in $SEEDS; do pair $s main $ITERS_A0 $ITERS_A $ITERS_B R1-Fb; done; }
P7() { log "P7 pack"; pack; log "ALL_DONE"; }
md5() { local o=${1:?usage: pod_fb.sh md5 <out file>}; (cd $R && md5sum $UPLOADS) > $o || { echo "md5: an upload is missing"; exit 1; }; echo "MD5 $(wc -l < $o) files -> $o"; }

case "${1:-plan}" in
  plan) plan; if [ "$DRY" = 1 ]; then echo "--- dry run ($W)"; : > $P; P0 && P1 && P2 && P3 && P456 && P7 && echo "DRY OK"; fi ;;
  P0|P1|P2|P3|P456|P7) $1 ;;
  md5) md5 "${2:-}" ;;
  all) P0 && P1 && P2 && P3 && P456 && P7 ;;
  *) echo "usage: pod_fb.sh plan|P0|P1|P2|P3|P456|P7|all|md5 <file>"; exit 64 ;;
esac
