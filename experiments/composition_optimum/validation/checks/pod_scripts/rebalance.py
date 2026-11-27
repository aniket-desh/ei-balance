"""rebalance.py STRIPE:POD[,POD...] ... : global, duplicate-safe reassignment of stripes' queued networks.

1. Kill every runner (`run.py --part S/24`) for the named stripes on every pod (SIGKILL; networks keep running).
2. Gather, from every pod, the networks running (`--one`) and finished (records present).
3. Any network running on more than one pod, or running while finished elsewhere, keeps only its oldest copy.
4. Queued = stripe networks neither finished nor running anywhere; dealt round-robin to the listed pods
   (first is the stripe's own pod), each getting a runner with --only. given.txt on the stripe's own pod
   lists every queued name not assigned to it, so a later restart.sh there cannot pick them up.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S = Path(__file__).resolve().parent
PODS = {l.split()[0]: l.split() for l in (S / "pods.txt").read_text().splitlines() if l.strip()}
OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={S}/known_hosts", "-o", "ConnectTimeout=20"]
V = "/root/research/ei/experiments/composition_optimum/validation"
LOCAL = Path("/Users/aniket/Documents/research/ei-balance/experiments/composition_optimum/validation")


def ssh(stripe, cmd):
    _, _, _, host, port, _ = PODS[stripe]
    return subprocess.run(["ssh", *OPTS, "-p", port, f"root@{host}", cmd], capture_output=True, text=True).stdout


plan = {a.split(":")[0]: a.split(":")[1].split(",") for a in sys.argv[1:]}
stripes_ = list(plan)
kill = "; ".join(f'pkill -9 -f "^/root/venv/bin/python run.py --part {s}/24"' for s in stripes_) + "; true"
with ThreadPoolExecutor(24) as ex:
    list(ex.map(lambda p: ssh(p, kill), PODS))
STATE = "ps -eo pid=,etimes=,args= | grep '[-]-one ' | awk '{print \"R\", $1, $2, $6}'; ls " + V + "/records | sed 's/.json.gz$//; s/^/D /'"
with ThreadPoolExecutor(24) as ex:
    states = dict(zip(PODS, ex.map(lambda p: ssh(p, STATE), PODS)))
done, running = set(), {}
for pod, out in states.items():
    for line in out.splitlines():
        t = line.split()
        if t[0] == "D":
            done.add(t[1])
        elif t[0] == "R":
            running.setdefault(t[3], []).append((int(t[2]), int(t[1]), pod))
sys.path.insert(0, str(LOCAL))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("runmod", LOCAL / "run.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
parts = m.stripes(json.loads((LOCAL / "jobs.json").read_text()), 24)
killed = []
for name, copies in running.items():
    copies.sort(reverse=True)                      # oldest (largest elapsed) first
    drop = copies[1:] if name not in done else copies
    for _, pid, pod in drop:
        ssh(pod, f"kill -9 {pid}")
        killed.append((name, pod))
print("duplicate copies killed:", killed)
for s, pods in plan.items():
    mine = [j["name"] for j in parts[int(s) - 1]]
    queued = [n for n in mine if n not in done and n not in running]
    share = {p: queued[k::len(pods)] for k, p in enumerate(pods)}
    ssh(s, f"echo -n {','.join(n for n in queued if n not in share[s])}, > /root/research/given.txt")
    for p, names in share.items():
        if not names:
            continue
        nrun = int(ssh(p, 'ps -eo args | grep -c "[-]-one " || true').strip() or 0)
        w = max(2, 15 - nrun)
        log, code = ("run.log", "exit_code") if p == s else (f"run_help{s}.log", f"exit_code_help{s}")
        ssh(p, f'cd {V}; rm -f /root/research/{code}; nohup sh -c "/root/venv/bin/python run.py --part {s}/24 --only {",".join(names)} '
               f'--workers {w} --memory-gib 56 --scratch-gib 400 >> /root/research/{log} 2>&1; echo \\$? > /root/research/{code}" '
               f'</dev/null >/dev/null 2>&1 & echo "$(date -u +%FT%TZ) rebalance: stripe {s}, {len(names)} networks, workers {w}" >> /root/research/restart.log')
    print(f"stripe {s}: running {sum(n in running for n in mine)}, done {sum(n in done for n in mine)}, queued {len(queued)} -> "
          + ", ".join(f"pod-of-stripe {p}: {len(v)}" for p, v in share.items()))
