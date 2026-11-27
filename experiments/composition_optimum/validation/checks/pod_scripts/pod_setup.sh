#!/bin/bash
# usage: pod_setup.sh HOST PORT STRIPE COUNT WORKERS MEMORY_GIB [TIMER_SECONDS]
# 1. arms and verifies the self-stop timer, 2. uploads source, 3. installs, 4. launches one stripe detached.
set -euo pipefail
HOST=$1; PORT=$2; STRIPE=$3; COUNT=$4; WORKERS=$5; MEM=$6; TIMER=${7:-21600}
S=$(cd "$(dirname "$0")" && pwd)
OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$S/known_hosts -o ConnectTimeout=20"
SSH="ssh $OPTS -p $PORT root@$HOST"
# Timer first, fully detached. Verified from the process list by exact args (the container already runs
# "sleep infinity", and pgrep -f would match this shell's own command line).
$SSH "mkdir -p /root/research; if ps -eo args | grep -q '^[s]leep $TIMER\$'; then echo timer already armed; else (nohup sh -c '. /etc/rp_environment; export RUNPOD_API_KEY RUNPOD_POD_ID; sleep $TIMER; runpodctl stop pod \$RUNPOD_POD_ID || runpodctl pod stop \$RUNPOD_POD_ID' </dev/null >/root/research/stop-timer.log 2>&1 &); sleep 1; fi; . /etc/rp_environment; echo pod=\$RUNPOD_POD_ID; echo TIMER_PROCESS: \$(ps -eo pid,etimes,args | grep '[0-9] sleep $TIMER\$'); which runpodctl; nproc; free -g | sed -n 2p; df -h / | tail -1"
scp -q $OPTS -P $PORT $S/val_source.tgz root@$HOST:/root/research/
$SSH 'set -e; cd /root/research && mkdir -p ei && tar xzf val_source.tgz -C ei && sha256sum val_source.tgz; (python3 -m venv /root/venv >/dev/null 2>&1 && /root/venv/bin/pip --version >/dev/null 2>&1) || (apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq python3-venv >/dev/null 2>&1; python3 -m venv /root/venv); which g++ >/dev/null 2>&1 || (apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq g++ >/dev/null 2>&1); /root/venv/bin/pip install -q numpy==2.5.5 scipy==1.18.1 psutil threadpoolctl 2>&1 | tail -2; g++ --version | head -1; /root/venv/bin/python -c "import numpy,scipy;print(numpy.__version__,scipy.__version__)"'
$SSH "cd /root/research/ei/experiments/composition_optimum/validation && (nohup sh -c '/root/venv/bin/python run.py --part $STRIPE/$COUNT --workers $WORKERS --memory-gib $MEM --scratch-gib 100 > /root/research/run.log 2>&1; echo \$? > /root/research/exit_code' </dev/null >/dev/null 2>&1 &); sleep 8; tail -3 /root/research/run.log"
