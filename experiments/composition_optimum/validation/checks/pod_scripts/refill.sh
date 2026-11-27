#!/bin/bash
# refill.sh : for every stripe still running, restart its runner (restart.sh) when fewer than W-2 networks are running
# and its current runner still has queued networks, so the worker cap tracks finished orphans.
S=$(cd "$(dirname "$0")" && pwd)
OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$S/known_hosts -o ConnectTimeout=20"
while read stripe id name host port kind; do
  if [ "$kind" = cpu3c32 ]; then W=31; M=112; else W=15; M=56; fi
  (
  st=$(ssh $OPTS -p $port root@$host 'n=$(ps -eo args | grep -c "[-]-one "); q=$(tail -1 /root/research/run.log | grep -o "([0-9]* pending" | tr -dc 0-9); [ -f /root/research/exit_code ] && e=1 || e=0; echo "$n ${q:-x} $e"')
  set -- $st
  if [ "$3" = 0 ] && [ "$1" -lt $((W-2)) ] && [ "$2" != 0 ]; then
    echo "stripe $stripe refill: $($S/restart.sh $host $port $stripe $W $M 2>&1 | head -1)"
  else
    echo "stripe $stripe skip: running $1 queued $2 exit_file $3"
  fi
  ) &
done < $S/pods.txt
wait
