"""Extract the three cell arrays used for prospective gain and variance scoring."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ('labels', 'q_x', 'mean_gain')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = HERE/'measured'
    destination.mkdir(exist_ok=True)
    manifest = {}
    for original in json.loads((HERE/'checksums.json').read_text()):
        name = original['name']
        full = HERE/'runs'/f'{name}_B500_neuron_statistics.npz'
        assert digest(full) == original['cell_statistics_sha256']
        target = destination/f'{name}.npz'
        with np.load(full, allow_pickle=False) as full_arrays:
            np.savez_compressed(target, **{field:full_arrays[field] for field in FIELDS})
            with np.load(target, allow_pickle=False) as extracted:
                for field in FIELDS:
                    assert extracted[field].dtype == full_arrays[field].dtype
                    np.testing.assert_array_equal(extracted[field], full_arrays[field])
        manifest[name] = {**original, 'compact_sha256':digest(target)}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
