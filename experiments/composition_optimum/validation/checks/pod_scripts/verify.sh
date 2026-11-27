#!/bin/bash
# verify.sh STRIPE : pull, then compare remote and local SHA-256 of every record, array and audit file on that pod
S=$(cd "$(dirname "$0")" && pwd)
DEST=/Users/aniket/Documents/research/ei-balance/experiments/composition_optimum/validation
OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$S/known_hosts -o ConnectTimeout=20"
line=$(grep "^$1 " $S/pods.txt); host=$(echo $line | cut -d' ' -f4); port=$(echo $line | cut -d' ' -f5); name=$(echo $line | cut -d' ' -f3)
for d in records arrays audit; do rsync -a --exclude '*.log' -e "ssh $OPTS -p $port" root@$host:/root/research/ei/experiments/composition_optimum/validation/$d/ $DEST/$d/; done
ssh $OPTS -p $port root@$host 'cd /root/research/ei/experiments/composition_optimum/validation && find records arrays audit -type f ! -name "*.log" | sort | xargs sha256sum' | sort -k2 > $S/remote_$1.sha
(cd $DEST && awk '{print $2}' $S/remote_$1.sha | xargs shasum -a 256 | sort -k2) > $S/local_$1.sha
echo "stripe $1 ($name): remote files $(wc -l < $S/remote_$1.sha), mismatches $(diff <(awk '{print $1, $2}' $S/remote_$1.sha) <(awk '{print $1, $2}' $S/local_$1.sha) | grep -c '^[<>]'), exit_code $(ssh $OPTS -p $port root@$host 'cat /root/research/exit_code 2>/dev/null || echo running')"
