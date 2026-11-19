"""Run a disjoint shard with one CPU process and one BLAS thread per network."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / 'runtime'
sys.path.insert(0, str(RUNTIME))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'


def complete(name):
    path = HERE / 'runs' / (name + '.json')
    return path.is_file() and json.loads(path.read_text())['status'] == 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', choices=('local','remote','all'), default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--memory-gib', type=float, default=4)
    parser.add_argument('--scratch-gib', type=float, default=20)
    parser.add_argument('--part', default='1/1', help='1-based stripe within the selected shard, e.g. 2/3')
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--one', help=argparse.SUPPRESS)
    parser.add_argument('--library', help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = HERE / 'jobs.json'
    jobs = json.loads(manifest.read_text())
    for directory in ('runs','tmp'):
        (HERE / directory).mkdir(exist_ok=True)
    if args.one:
        import engine_v4 as engine
        engine.HERE = HERE  # Keep fresh outputs separate from archival campaign records.
        job = next(j for j in jobs if j['name'] == args.one)
        if args.library:
            job['native_library'] = args.library
        print(json.dumps(engine.run_one(job)), flush=True)
        return

    stripe, count = map(int, args.part.split('/'))
    if not 1 <= stripe <= count or min(args.workers,args.memory_gib,args.scratch_gib) <= 0:
        parser.error('Positive resource limits and a valid 1-based stripe are required.')
    jobs = [j for j in jobs if args.shard == 'all' or j['shard'] == args.shard][stripe-1::count]
    pending = [j for j in jobs if not complete(j['name'])]
    if not pending:
        print('All selected network records are complete.', flush=True)
        return
    biggest = max(j['config']['n'] for j in pending)
    gib = 1024**3
    mem_cap = int(min(args.memory_gib*gib, .75*psutil.virtual_memory().available))
    disk_cap = int(min(args.scratch_gib*gib, .75*shutil.disk_usage(HERE).free))
    per_worker = 192*1024**2 + 24*biggest**2 + 32*biggest*2000
    disk_per_worker = max(round(j['config']['duration']/j['config']['sample_dt'])*j['config']['n']*4 +
                          round(j['config']['duration']/min(j['blocks']))*j['config']['n']**2*8 for j in pending)
    if sys.platform == 'darwin':
        # macOS may keep the raw trajectory and block covariances resident
        # during contractions. Include the mappings in the observed-RSS budget.
        per_worker = max(per_worker, 384*1024**2 + 64*biggest**2 + disk_per_worker)
    usable_ram = mem_cap - 128*1024**2  # supervisor and imports also count towards RSS
    workers = min(len(pending), args.workers, os.cpu_count() or 1, usable_ram//per_worker, disk_cap//disk_per_worker)
    if workers < 1:
        parser.error('The largest selected network exceeds the supplied RAM or scratch capacity.')
    identity = f'{socket.gethostname()}_{args.shard}_{stripe}of{count}_{os.getpid()}'
    audit = {'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(), 'selected_names':[j['name'] for j in jobs],
             'workers':workers,'memory_cap_bytes':mem_cap,'scratch_cap_bytes':disk_cap,'platform':platform.platform(),
             'python':sys.version,'start_unix':time.time(),'max_rss_bytes':0,'failures':[]}
    library = None
    if args.native:
        from native_action import build
        library, command = build(HERE / 'tmp' / identity)
        audit.update(compile_command=command, library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    import engine_v4 as engine
    import numpy as np
    import scipy
    audit.update(numpy=np.__version__, scipy=scipy.__version__)
    model_source = engine.ROOT/'recovered/bundle/ei_full/ei_balance_handoff/src/eicavity'
    sources = [Path(__file__),RUNTIME/'engine_v4.py',RUNTIME/'native_action.py',RUNTIME/'constant_csr.cpp',
               model_source/'network.py',model_source/'degree.py',model_source/'single_site.py']
    audit['source_sha256'] = {str(f.resolve().relative_to(engine.ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    audit_path = HERE / 'runs' / f'{identity}_audit.json'
    engine.atomic(audit_path, audit)
    print(f'{len(pending)} pending, {workers} workers, RAM cap {mem_cap/gib:.2f} GiB, audit {audit_path.name}', flush=True)
    active = {}
    next_checkpoint = time.monotonic() + 15
    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                name = job['name']
                lock = (HERE/'tmp'/(name+'.lock')).open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    raise RuntimeError(f'{name} is already running on this filesystem. Use disjoint stripes.')
                if complete(name):
                    lock.close()
                    continue
                cmd = [sys.executable, __file__, '--one', name]
                if library is not None:
                    cmd.extend(['--library',str(library)])
                proc = subprocess.Popen(cmd)
                active[proc] = (name,lock)
                print(f'started {name}', flush=True)
            rss = psutil.Process().memory_info().rss
            for proc,(name,lock) in list(active.items()):
                status = proc.poll()
                if status is not None:
                    if status != 0 or not complete(name):
                        audit['failures'].append(name)
                    lock.close()
                    del active[proc]
                else:
                    try:
                        rss += psutil.Process(proc.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
            audit['max_rss_bytes'] = max(audit['max_rss_bytes'],rss)
            if rss > mem_cap:
                raise MemoryError('Measured aggregate worker RSS exceeded the cap; retry with fewer workers.')
            if time.monotonic() >= next_checkpoint:
                audit.update(last_check_unix=time.time(),active=[name for name,lock in active.values()],pending=len(pending))
                engine.atomic(audit_path, audit)
                next_checkpoint = time.monotonic() + 15
            if active:
                time.sleep(1)
    except BaseException as exc:
        audit['abort_reason'] = repr(exc)
        raise
    finally:
        for proc,(name,lock) in active.items():
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
            lock.close()
        audit.update(finished_unix=time.time(),complete=sum(complete(j['name']) for j in jobs))
        engine.atomic(audit_path, audit)
    if audit['failures']:
        raise SystemExit('Failed networks: '+', '.join(audit['failures']))


if __name__ == '__main__':
    main()
