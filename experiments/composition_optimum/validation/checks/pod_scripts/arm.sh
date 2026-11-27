#!/bin/bash
# arm.sh HOST PORT TIMER : arm a detached self-stop timer and print the verified timer process
S=$(cd "$(dirname "$0")" && pwd); HOST=$1; PORT=$2; TIMER=$3
OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$S/known_hosts -o ConnectTimeout=20"
ssh $OPTS -p $PORT root@$HOST "mkdir -p /root/research; if ps -eo args | grep -q '^[s]leep $TIMER\$'; then echo already armed; else (nohup sh -c '. /etc/rp_environment; export RUNPOD_API_KEY RUNPOD_POD_ID; sleep $TIMER; runpodctl stop pod \$RUNPOD_POD_ID || runpodctl pod stop \$RUNPOD_POD_ID' </dev/null >/root/research/stop-timer.log 2>&1 &); sleep 1; fi; . /etc/rp_environment; echo \"pod=\$RUNPOD_POD_ID timer=[\$(ps -eo pid,etimes,args | grep '[0-9] sleep $TIMER\$')] runpodctl=\$(which runpodctl) nproc=\$(nproc) disk=\$(df -h / | tail -1 | tr -s ' ' | cut -d' ' -f2,4)\""
