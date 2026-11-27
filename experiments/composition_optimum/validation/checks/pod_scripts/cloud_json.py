"""Build validation/cloud_execution.json from created.txt, stops.txt, hosts.txt, pods.txt and verify_*.txt."""
import json
from pathlib import Path

S = Path(__file__).resolve().parent
DEST = Path("/Users/aniket/Documents/research/ei-balance/experiments/composition_optimum/validation")
created = {l.split()[0]: l.split() for l in (S / "created.txt").read_text().split("\n") if l.strip()}
stops = {l.split()[0]: l.split() for l in (S / "stops.txt").read_text().split("\n") if l.strip()}
hosts = {l.split()[0]: l.split() for l in (S / "hosts.txt").read_text().split("\n") if l.strip()}
stripe_of = {l.split()[2][-2:]: l.split()[0] for l in (S / "pods.txt").read_text().split("\n") if l.strip()}
pods, total, files, mism = [], 0., 0, 0
for k in sorted(created):
    num, pid, kind, vcpu, rate, ctime = created[k]
    st = stops.get(k)
    up = int(st[5]) if st else None
    cost = round(up / 3600 * float(rate), 3) if up else None
    total += cost or 0
    ver = None
    stripe = stripe_of.get(k)
    vf = S / f"verify_{stripe}.txt"
    if stripe and vf.exists():
        t = vf.read_text().split()
        ver = dict(line=vf.read_text().strip(), remote_files=int(t[t.index("files") + 1].rstrip(",")), mismatches=int(t[t.index("mismatches") + 1].rstrip(",")))
        files += ver["remote_files"]
        mism += ver["mismatches"]
    pods.append(dict(name=hosts[k][2], id=pid, cpu_type=kind, vcpu=int(vcpu), rate_usd_per_hour=float(rate), cloud="SECURE", data_center="EUR-IS-1",
                     created_utc=ctime, role=f"cohort stripe {stripe}/24" if stripe else "replay check (control 4)",
                     stopped_utc=st[3] if st else None, uptime_seconds_at_stop=up, final_status=st[7] if st else None,
                     estimated_cost_usd=cost, collection=ver))
# Global check after all collection: every remote file against the final local copy, and files that exist on two pods.
import hashlib
copies = {}
for f in sorted(S.glob("remote_*.sha")):
    stripe = f.stem.split("_")[1]
    for line in f.read_text().splitlines():
        h, path = line.split()
        copies.setdefault(path, []).append((stripe, h))
local_hash = {p: hashlib.sha256((DEST / p).read_bytes()).hexdigest() for p in copies}
unique = [p for p, c in copies.items() if len(c) == 1]
dups = {p: c for p, c in copies.items() if len(c) > 1}
dup_rows = []
for p, c in sorted(dups.items()):
    dup_rows.append(dict(path=p, pods=[f"stripe {s}" for s, _ in c], hashes_identical=len({h for _, h in c}) == 1,
                         local_matches_a_copy=local_hash[p] in {h for _, h in c}))
networks_twice = sorted({Path(p).name.replace(".json.gz", "").replace(".npz", "").replace(".json", "") for p in dups})
global_check = dict(files=len(copies), unique_files=len(unique), unique_files_matching_local=sum(local_hash[p] == copies[p][0][1] for p in unique),
                    files_on_two_pods=len(dups), networks_on_two_pods=networks_twice,
                    duplicate_arrays_byte_identical=all(r["hashes_identical"] for r in dup_rows if r["path"].startswith("arrays/")),
                    duplicate_records_local_matches_a_copy=all(r["local_matches_a_copy"] for r in dup_rows),
                    note="Records of a network run twice differ in bytes (host name and timestamps inside the gzip JSON); their arrays "
                         "(per-neuron statistics) are byte-identical, so the simulations agree. The local record is the copy pulled last.")
replay = S / "verify_replay.txt"
out = dict(account=dict(team="CBAI", note="verified with get-account before any pod was created; list-pods never called"),
           image="runpod/base:1.0.2-ubuntu2404", disk_gb=120, ports=["22/tcp"], self_stop_timer_seconds=21600,
           final_readback=dict(utc="2026-11-27T18:16Z", method="get-pod on all 25 pod ids after the last stop", all_exited=True),
           wall_clock=dict(first_create_utc="2026-11-27T16:01:33Z", last_stop_utc=max(p["stopped_utc"] for p in pods if p["stopped_utc"])),
           pods=pods, total_rate_usd_per_hour=round(sum(p["rate_usd_per_hour"] for p in pods), 2),
           estimated_total_cost_usd=round(total, 2),
           pod_hours=round(sum((p["uptime_seconds_at_stop"] or 0) for p in pods) / 3600, 2),
           cost_note="uptime at stop (from get-pod) times the on-demand rate quoted at creation; stopped pods keep billing for their container disk",
           checksums=dict(cohort_files_compared=files, mismatches=mism, method="sha256sum on the pod vs shasum -a 256 locally, every record, array and audit json, before stopping",
                          replay=replay.read_text().strip() if replay.exists() else None, passed=mism == 0 and files > 0,
                          at_stop="each pod: remote vs local SHA-256 of every record, array and audit json, just before its stop",
                          global_after_collection=global_check))
(DEST / "cloud_execution.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(dict(total=out["estimated_total_cost_usd"], files=files, mismatches=mism, stopped=sum(p["final_status"] == "EXITED" for p in pods)), indent=0))
