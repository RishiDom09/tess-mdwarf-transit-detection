"""Portable paths, immutable run identity, and atomic artifact receipts."""
import hashlib
import json
import os
from pathlib import Path
from .settings import REPOSITORY

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(str(temporary), str(path))

def inventory():
    paths = [p for folder in ('src', 'configs', 'manifests', 'scripts')
             for p in (REPOSITORY/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')]
    return {str(p.relative_to(REPOSITORY)): digest(p) for p in sorted(paths)}

def freeze(root):
    root = Path(root).resolve()
    pin = REPOSITORY/'manifests/checksums.json'
    for name, sha in read(pin).items():
        if digest(REPOSITORY/name) != sha:
            raise ValueError('Published configuration/manifest changed: '+name)
    contract = {'schema': 1, 'files': inventory()}
    target = root/'contract.json'
    if target.exists():
        if read(target) != contract:
            raise ValueError('Code, settings or manifests changed; use a new run directory')
    else:
        write(target, contract)
    return digest(target)

def receipt(folder, names, identity, extra=None):
    folder = Path(folder)
    write(folder/'completion.json', {'identity': identity,
          'hashes': {n: digest(folder/n) for n in names}, 'extra': extra or {}})

def completed(folder, identity):
    folder = Path(folder)
    if not (folder/'completion.json').exists():
        return False
    r = read(folder/'completion.json')
    if r['identity'] != identity:
        raise ValueError('Artifact belongs to a different run: '+str(folder))
    for name, sha in r['hashes'].items():
        if digest(folder/name) != sha:
            raise ValueError('Committed artifact changed: '+str(folder/name))
    return True
