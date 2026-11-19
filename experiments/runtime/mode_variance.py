"""Measure temporal variance along unit right eigenvectors of a fixed-point linearization.

Run on a completed record with retain_temporary_trajectory=true. Each color in
the proposed mode plot is v^H C v for one unit probe and centered covariance C.
Nonorthogonal probes can see the same variance, so their colors need not sum to
the trace. Using input (x) or rate (phi(x)) coordinates changes the probes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.linalg import eig
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'recovered/bundle/ei_full/ei_balance_handoff/src'))
from eicavity.network import NetworkConfig, make_network
from eicavity.single_site import rate, rate_prime


def mode_variances(trajectory, directions, transform=None, chunk_size=512):
    """Full, first-half and second-half variances with divisor T in each window."""
    conjugate_probes = directions.conj()
    n_time = len(trajectory)
    if n_time < 4:
        raise ValueError('Each recording half needs at least two samples.')
    means, variances = [], []
    boundaries = (0, n_time // 2, n_time)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        # Pool centered chunk moments to preserve between-chunk fluctuations
        # without subtracting a large squared DC component from a second moment.
        mean = np.zeros(directions.shape[1], dtype=conjugate_probes.dtype)
        squared_deviations = np.zeros(directions.shape[1])
        count = 0
        for offset in range(start, stop, chunk_size):
            x = np.asarray(trajectory[offset:min(offset + chunk_size, stop)], dtype=float)
            values = x if transform is None else transform(x)
            scores = values @ conjugate_probes
            chunk_mean = scores.mean(axis=0)
            difference = chunk_mean - mean
            size = len(scores)
            squared_deviations += np.sum(np.abs(scores - chunk_mean) ** 2, axis=0)
            squared_deviations += (count * size / (count + size)) * np.abs(difference) ** 2
            mean += (size / (count + size)) * difference
            count += size
        means.append(mean)
        variances.append(squared_deviations / count)
    fraction = boundaries[1] / n_time
    full = fraction * variances[0] + (1 - fraction) * variances[1]
    full += fraction * (1 - fraction) * np.abs(means[0] - means[1]) ** 2
    return np.stack((full, *variances))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path, help='Completed runtime/runs/<name>.json')
    parser.add_argument('--trajectory', type=Path, help='Retained <name>_trajectory.npy (default: runtime/tmp)')
    parser.add_argument('--fixed-point', type=Path, help='A candidate x* as a .npy vector. By default solve from the temporal mean of x.')
    parser.add_argument('--variable', choices=('input', 'rate'), default='input')
    parser.add_argument('--output', type=Path, help='Output .npz (default: beside the input record)')
    args = parser.parse_args(argv)
    record = json.loads(args.record.read_text())
    if record['status'] != 'complete':
        parser.error('A complete network record is required.')
    name = record['name']
    output = args.output or args.record.with_name(f'{name}_{args.variable}_mode_variance.npz')
    if output.suffix != '.npz':
        parser.error('The output filename must end in .npz.')
    trajectory_path = args.trajectory or args.record.parent.parent / 'tmp' / f'{name}_trajectory.npy'
    if not trajectory_path.is_file():
        parser.error(f'{trajectory_path} is missing. Use a freshly retained recording for this network.')
    trajectory = np.load(trajectory_path, mmap_mode='r', allow_pickle=False)
    job = record['job']
    config = NetworkConfig(**job['config'])
    if trajectory.shape != (round(config.duration / config.sample_dt), config.n):
        parser.error('Trajectory dimensions disagree with the network record.')
    J, drive, _, labels, _ = make_network(config, np.asarray(job['mean_input']))
    if hashlib.sha256(J).hexdigest() != record['connectivity_sha256']:
        parser.error('Regenerated connectivity differs from the network that produced the record.')
    # A same-size recording from another job can share the expected dimensions.
    # Compare the actual activity to the moments archived for these windows.
    estimator = record['estimators'][0]
    measured_samples = estimator['nblocks'] * round(estimator['block_duration'] / config.sample_dt)
    measured_mean = trajectory[:measured_samples].mean(axis=0, dtype=float)
    for population in estimator['population_local_stats']:
        actual = measured_mean[labels == population['population']].mean()
        if not np.isclose(actual, population['h'], rtol=1e-11, atol=1e-12):
            parser.error('Trajectory population means differ from the network record.')

    def drift(x):
        return -x + J @ rate(x) + drive

    if args.fixed_point is None:
        def jacobian(x):
            matrix = J * rate_prime(x)[None, :]
            matrix.flat[::len(x) + 1] -= 1
            return matrix

        initial = measured_mean if measured_samples == len(trajectory) else trajectory.mean(axis=0, dtype=float)
        solution = root(drift, initial, jac=jacobian, options={'xtol': 1e-10})
        fixed_point = solution.x
        origin = 'root initialized at the temporal mean input'
    else:
        fixed_point = np.load(args.fixed_point, allow_pickle=False)
        if fixed_point.shape != (config.n,) or not np.isrealobj(fixed_point):
            parser.error('The candidate fixed point must be one real input per neuron.')
        origin = str(args.fixed_point)
    residual = float(np.max(np.abs(drift(fixed_point))))
    if not np.isfinite(residual) or residual > 1e-8:
        parser.error(f'Fixed-point residual {residual:.3g} exceeds 1e-8. Supply a verified root with --fixed-point.')

    gains = rate_prime(fixed_point)
    eigenvalues, directions = eig(J * gains[None, :])  # Re(lambda) = 1 is the stability boundary.
    if args.variable == 'rate':
        # Push forward input directions under the local map dr = diag(phi') dx.
        directions *= gains[:, None]
    norms = np.linalg.norm(directions, axis=0)
    if np.any(norms == 0):
        parser.error('The selected coordinate map annihilated a probe direction.')
    directions /= norms
    variances = mode_variances(trajectory, directions, rate if args.variable == 'rate' else None)
    if not np.isfinite(variances).all():
        parser.error('Nonfinite projected variance. Check the input recording.')
    with TemporaryDirectory(dir=output.parent, prefix='.mode-variance-') as stage:
        pending = Path(stage) / output.name
        np.savez(pending, eigenvalues=eigenvalues, variance=variances, windows=['full', 'first_half', 'second_half'],
                 e_weight=np.sum(np.abs(directions[labels == 0]) ** 2, axis=0),
                 variable=args.variable, fixed_point=fixed_point, fixed_point_origin=origin,
                 fixed_point_residual_inf=residual, connectivity_sha256=record['connectivity_sha256'],
                 analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 record_name=name, seed=config.seed, sample_dt=config.sample_dt, n_samples=len(trajectory))
        pending.replace(output)
    print(f'{output}: {len(eigenvalues)} unit right directions, max fixed-point residual {residual:.3g}')


if __name__ == '__main__':
    main()
