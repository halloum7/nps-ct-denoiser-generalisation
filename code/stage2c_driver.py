# ---- Stage 2c driver (notebook cell). LIB = source of stage2c_lib.py ; N_ITER fixed from the pilot rule.
import os, sys, glob, json, time, shutil, subprocess
T0 = time.time(); N_ITER = int(os.environ.get("N_ITER_FIXED", "__N__"))
open("/kaggle/working/s2c.py", "w").write(LIB)
exec(LIB)
for f in glob.glob("/kaggle/input/**/stage2c/*", recursive=True):          # resume from an earlier session's output
    dst = os.path.join(OUT, os.path.basename(f))
    if not os.path.exists(dst): shutil.copy(f, dst)
S = equivalence_smoke(); print("EQUIV_JSON=" + json.dumps(S), flush=True)
torch.cuda.empty_cache()
est = {"REDCNN": S["REDCNN_sec_per_it"] * N_ITER * 1.15, "DnCNN": S["REDCNN_sec_per_it"] * N_ITER * 1.15}
deadline = T0 + 11.3 * 3600
logs = [open(f"/kaggle/working/worker{g}.log", "w") for g in (0, 1)]
P = [subprocess.Popen([sys.executable, "/kaggle/working/s2c.py", "worker", str(g), str(N_ITER), str(deadline)],
                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "EST_SEC": json.dumps(est)},
                      stdout=logs[g], stderr=subprocess.STDOUT) for g in (0, 1)]
for p in P: p.wait()
for g in (0, 1):
    logs[g].close(); print(open(f"/kaggle/working/worker{g}.log").read()[-20000:])
done = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{OUT}/*.json"))
print("STAGE2C_STATUS=" + json.dumps({"n_done": len(done), "n_total": len(all_jobs()), "return_codes": [p.returncode for p in P],
                                       "hours": (time.time() - T0) / 3600}))
