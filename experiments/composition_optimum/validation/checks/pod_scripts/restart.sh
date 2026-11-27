#!/bin/bash
# restart.sh HOST PORT STRIPE TOTAL_WORKERS MEMORY_GIB
# Replace a stripe's runner with one whose resource model admits more concurrent networks.
# Running networks are left alone (the old runner is killed with SIGKILL so its cleanup does not
# terminate them); the new runner gets exactly the stripe's networks that are neither done nor running.
S=$(cd "$(dirname "$0")" && pwd)
ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$S/known_hosts -o ConnectTimeout=20 -p $2 root@$1 "STRIPE=$3 W=$4 MEM=$5 bash -s" <<'EOF'
set -e
cd /root/research/ei/experiments/composition_optimum/validation
RP=$(pgrep -f "^/root/venv/bin/python run.py --part $STRIPE/24" || true)
[ -n "$RP" ] && kill -9 $RP
sleep 2
rm -f /root/research/exit_code
/root/venv/bin/python - <<PY > /root/research/restart_list.txt
import json, subprocess, importlib.util
spec = importlib.util.spec_from_file_location("runmod", "run.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
jobs = json.load(open("jobs.json"))
mine = m.stripes(jobs, 24)[$STRIPE - 1]
ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
running = {l.split("--one ")[1].split()[0] for l in ps.splitlines() if "--one " in l}
import os
given = set(open("/root/research/given.txt").read().split(",")) if os.path.exists("/root/research/given.txt") else set()
todo = [j["name"] for j in mine if not m.done(j["name"]) and j["name"] not in running and j["name"] not in given]
print(len(running), len(todo), ",".join(todo))
PY
read NRUN NTODO LIST < /root/research/restart_list.txt
NW=$((W - NRUN)); [ $NW -lt 1 ] && NW=1
echo "$(date -u +%FT%TZ) restart: killed runner $RP, running $NRUN, todo $NTODO, workers $NW, memory $MEM" | tee -a /root/research/restart.log
cd /root/research/ei/experiments/composition_optimum/validation
if [ "$NTODO" -gt 0 ]; then
  nohup sh -c "/root/venv/bin/python run.py --part $STRIPE/24 --only $LIST --workers $NW --memory-gib $MEM --scratch-gib 400 >> /root/research/run.log 2>&1; echo \$? > /root/research/exit_code" </dev/null >/dev/null 2>&1 &
else
  echo 0 > /root/research/exit_code
fi
sleep 3
pgrep -fa "run.py --part" | cut -c1-90 | grep -v "^.*sh -c" || true
ps -eo args | grep -c "[-]-one "
EOF
