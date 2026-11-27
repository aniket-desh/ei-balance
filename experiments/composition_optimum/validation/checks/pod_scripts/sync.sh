#!/bin/bash
# Pull finished records/arrays/audits from every pod in pods.txt; one status line per pod.
S=$(cd "$(dirname "$0")" && pwd)
DEST=/Users/aniket/Documents/research/ei-balance/experiments/composition_optimum/validation
OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$S/known_hosts -o ConnectTimeout=20"
mkdir -p $DEST/records $DEST/arrays $DEST/audit
while read stripe id name host port kind; do
  (
  for d in records arrays audit; do
    rsync -a --exclude '*.log' -e "ssh $OPTS -p $port" root@$host:/root/research/ei/experiments/composition_optimum/validation/$d/ $DEST/$d/ 2>/dev/null
  done
  st=$(ssh $OPTS -p $port root@$host 'cd /root/research; echo "exit=$(cat exit_code 2>/dev/null || echo running) done=$(ls ei/experiments/composition_optimum/validation/records 2>/dev/null | wc -l) fail=$(grep -c FAILED run.log) load=$(cut -d" " -f1 /proc/loadavg) one=$(ps -eo args | grep -c "[-]-one ") timer=$(ps -eo etimes,args | grep -c "[0-9] sleep 21600$") $(tail -1 run.log | cut -c1-70)"' 2>/dev/null)
  echo "stripe $stripe $name: $st"
  ) &
done < $S/pods_active.txt
wait
echo "local records: $(find $DEST/records -name 'cv_*' | wc -l)"
