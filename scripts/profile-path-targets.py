#!/usr/bin/env python3
"""Paired NVIDIA process-footprint probe; GPU memory only, not execution time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import time


def capture(binary, cropped):
    env = os.environ.copy()
    # Select both modes explicitly: Linux enables cropping when the variable is absent.
    env['GPUI_GPU_EXPERIMENTS'] = 'cropped-paths' if cropped else ''
    process = subprocess.Popen([str(binary)], env=env, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(30):
                raise RuntimeError('renderer did not become ready')
        ready = process.stdout.readline().strip()
        if ready != f'ready {process.pid}':
            raise RuntimeError(f'bad completion witness: {ready!r}')
        samples = []
        for _ in range(12):
            result = subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'], text=True, timeout=5)
            matches = [float(line.split(',')[1]) for line in result.splitlines()
                       if int(line.split(',')[0]) == process.pid]
            if len(matches) != 1 or matches[0] <= 0:
                raise RuntimeError('one process GPU footprint is required')
            samples.append(matches[0])
            time.sleep(0.1)
        process.stdin.write('\n')
        process.stdin.flush()
        _, stderr = process.communicate(timeout=10)
        if process.returncode:
            raise RuntimeError(f'renderer exited {process.returncode}: {stderr}')
        return {'mib_samples': samples, 'median_mib': statistics.median(samples),
                'stderr': stderr, 'cropped': cropped}
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 3:
        parser.error('at least three pairs in each of two sessions are required')
    binary = args.binary.resolve()
    report = {'binary': str(binary), 'sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'metric': 'process GPU footprint MiB (NVIDIA driver)',
              'scope': '2560x1440 headless renderer, 512x1408 retained path target; no app latency claim',
              'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
                                              '--format=csv,noheader'], text=True).strip(),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'sessions': []}
    reductions = []
    for session in range(2):
        pairs = []
        for pair in range(args.pairs):
            order = [False, True] if (session + pair) % 2 == 0 else [True, False]
            runs = [capture(binary, mode) for mode in order]
            by_mode = {run['cropped']: run for run in runs}
            reduction = 1 - by_mode[True]['median_mib'] / by_mode[False]['median_mib']
            reductions.append(reduction)
            pairs.append({'order': order, 'runs': runs, 'reduction': reduction})
            print(f'session {session + 1}, pair {pair + 1}: {reduction:.1%}', flush=True)
        report['sessions'].append(pairs)
    report['median_reduction'] = statistics.median(reductions)
    report['significant_memory_benefit'] = all(value >= 0.10 for value in reductions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report['significant_memory_benefit']:
        raise SystemExit('no consistent >=10% memory improvement')


if __name__ == '__main__':
    main()
