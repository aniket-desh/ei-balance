"""Append the scope decision while preserving all prior archive bytes."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / 'tmp/before_scope_boundary.zip'
OUTPUT = ROOT / 'correction/ei_finite_size_theory_working.zip'
HISTORY = 'archive_history/before_next_round_scope/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    with zipfile.ZipFile(PREVIOUS) as archive:
        assert archive.testzip() is None
        old = {p: archive.read(p) for p in archive.namelist()
               if not p.endswith('/')}
    for entry in json.loads(old['MANIFEST.json']):
        assert sha(old[entry['path']]) == entry['sha256']
    data = dict(old)
    additions = [ROOT / 'research_round/README.md']
    additions += sorted(p for p in (ROOT / 'research_round/scope').rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts)
    for path in additions:
        key = str(path.relative_to(ROOT))
        value = path.read_bytes()
        if key in old and value != old[key]:
            data[HISTORY + key] = old[key]
        data[key] = value
    status = json.loads(old['ARCHIVE_STATUS.json'])
    status.update({
        'scope_read_first': 'research_round/scope/NEXT_ROUND_SCOPE.md',
        'scope_constraint': 'Finish and validate the conditional balanced EI paper; optional extensions do not become prerequisites.',
        'previous_archive_sha256': sha(PREVIOUS.read_bytes()),
        'scope_update_changes_scientific_results': False,
    })
    data[HISTORY + 'ARCHIVE_STATUS.json'] = old['ARCHIVE_STATUS.json']
    data['ARCHIVE_STATUS.json'] = (json.dumps(status, indent=2) + '\n').encode()
    data[HISTORY + 'MANIFEST.json'] = old['MANIFEST.json']
    del data['MANIFEST.json']
    manifest = [{'path': p, 'bytes': len(v), 'sha256': sha(v)}
                for p, v in sorted(data.items())]
    data['MANIFEST.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    temporary = OUTPUT.with_suffix('.building.zip')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path, value in sorted(data.items()):
            archive.writestr(path, value)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert set(old).issubset(archive.namelist())
        for entry in json.loads(archive.read('MANIFEST.json')):
            assert sha(archive.read(entry['path'])) == entry['sha256']
        for key, value in old.items():
            if archive.read(key) != value:
                assert archive.read(HISTORY + key) == value
    temporary.replace(OUTPUT)
    print(json.dumps({'path': str(OUTPUT), 'entries': len(data),
                      'bytes': OUTPUT.stat().st_size,
                      'sha256': sha(OUTPUT.read_bytes()),
                      'all_prior_paths_and_bytes_preserved': True,
                      'CRC_and_manifest_checks_passed': True}, indent=2))


if __name__ == '__main__':
    main()
