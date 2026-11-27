"""split.py HEAVY HELPER : move half of a stripe's queued networks to a pod that has finished its own queue.

On the heavy pod the runner is killed (SIGKILL, running networks continue) and the stripe's networks
that are neither done nor running are listed; alternate ones go back to a new runner on the heavy
pod and to a runner for the same stripe on the helper pod (both with --only, so no network can run twice).
"""
import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
PODS = {l.split()[0]: l.split() for l in (S / "pods.txt").read_text().splitlines() if l.strip()}
OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={S}/known_hosts", "-o", "ConnectTimeout=20"]
V = "/root/research/ei/experiments/composition_optimum/validation"


def ssh(stripe, cmd):
    _, _, _, host, port, _ = PODS[stripe]
    return subprocess.run(["ssh", *OPTS, "-p", port, f"root@{host}", cmd], capture_output=True, text=True, check=True).stdout


def running(stripe):
    return int(ssh(stripe, 'ps -eo args | grep -c "[-]-one " || true').strip())


heavy, helper = sys.argv[1], sys.argv[2]
LIST = f"""cd {V}; /root/venv/bin/python - <<'PY'
import json, subprocess, importlib.util
spec = importlib.util.spec_from_file_location("runmod", "run.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
mine = m.stripes(json.load(open("jobs.json")), 24)[{heavy} - 1]
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
run = {{l.split("--one ")[1].split()[0] for l in ps.splitlines() if "--one " in l}}
print(len(run), ",".join(j["name"] for j in mine if not m.done(j["name"]) and j["name"] not in run))
PY"""
ssh(heavy, f'RP=$(pgrep -f "^/root/venv/bin/python run.py --part {heavy}/24" || true); [ -n "$RP" ] && kill -9 $RP; sleep 2; rm -f /root/research/exit_code; true')
nrun, names = ssh(heavy, LIST).split()
names = names.split(",")
keep, give = names[0::2], names[1::2]


def launch(stripe, part, only, workers, tag):
    log, code = ("run.log", "exit_code") if tag == "" else (f"run_help{part}.log", f"exit_code_help{part}")
    cmd = (f'cd {V}; nohup sh -c "/root/venv/bin/python run.py --part {part}/24 --only {",".join(only)} --workers {workers} '
           f'--memory-gib 56 --scratch-gib 400 >> /root/research/{log} 2>&1; echo \\$? > /root/research/{code}" </dev/null >/dev/null 2>&1 & '
           f'echo "$(date -u +%FT%TZ) {tag or "restart"}: stripe {part}, {len(only)} networks, workers {workers}" >> /root/research/restart.log')
    ssh(stripe, cmd)


ssh(heavy, f"echo -n {','.join(give)}, >> /root/research/given.txt")
launch(heavy, heavy, keep, max(1, 15 - int(nrun)), "")
launch(helper, heavy, give, max(1, 15 - running(helper)), "helper")
print(f"stripe {heavy}: {len(names)} queued, {nrun} running; kept {len(keep)}, gave {len(give)} to pod of stripe {helper}")
