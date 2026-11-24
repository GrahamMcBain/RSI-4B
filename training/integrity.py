"""Small fail-closed provenance and crash-safe storage helpers (stdlib only)."""
import fcntl
import hashlib
import importlib.metadata
import platform
import json
import os
import tempfile
import unicodedata
from contextlib import contextmanager
from pathlib import Path


class IntegrityError(ValueError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    _atomic_text(path, _json(value) + '\n')


def atomic_jsonl(path, rows):
    # Serialize the entire snapshot before touching the old destination.
    _atomic_text(path, ''.join(_json(row) + '\n' for row in rows))


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _invalid_constant(value):
    raise IntegrityError('Non-finite JSON constant: ' + value)


def strict_json(text):
    return json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_invalid_constant)


def read_jsonl(path):
    path = Path(path)
    data = path.read_bytes()
    # Even a syntactically complete last object without a newline is not a
    # committed record. Preserve the file untouched for operator inspection.
    if data and not data.endswith(b'\n'):
        raise IntegrityError(f'{path}: uncommitted/torn last line; file preserved, use a new output or inspect it explicitly')
    rows = []
    for number, line in enumerate(data.splitlines(), 1):
        try:
            row = strict_json(line.decode('utf-8'))
            if not isinstance(row, dict):
                raise IntegrityError('Expected an object')
        except (ValueError, UnicodeError) as error:
            raise IntegrityError(f'{path}:{number}: invalid JSONL; file preserved: {error}') from error
        rows.append(row)
    return rows


def prompt_key(messages):
    """Hash normalized conversation context, excluding only a final assistant target."""
    if not isinstance(messages, list):
        raise IntegrityError('messages must be a list')
    context = messages[:-1] if messages and messages[-1].get('role') == 'assistant' else messages
    def normalize(value):
        if isinstance(value, str):
            return ' '.join(unicodedata.normalize('NFKC', value).split())
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value
    return fingerprint(normalize(context))


@contextmanager
def file_lock(path):
    """Nonblocking advisory lock; never unlink the stable lock inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise IntegrityError(f'Output is locked by another writer: {path}') from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def unique_rows(rows, key='sample_id', expected=None):
    result = {}
    for row in rows:
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier:
            raise IntegrityError(f'Missing/invalid {key}')
        if identifier in result:
            raise IntegrityError(f'Duplicate {key}: {identifier}')
        if expected is not None and identifier not in expected:
            raise IntegrityError(f'Unexpected {key}: {identifier}')
        result[identifier] = row
    return result


def bind_manifest(output, manifest):
    """Caller holds output lock. Never adopt legacy or mismatched output."""
    output = Path(output)
    sidecar = Path(str(output) + '.manifest.json')
    if sidecar.exists():
        old = strict_json(sidecar.read_text(encoding='utf-8'))
        if old != manifest:
            raise IntegrityError(f'Manifest mismatch for {output}; use a new output namespace')
    elif output.exists():
        raise IntegrityError(f'Output has no manifest: {output}; refusing legacy/stale row reuse')
    else:
        atomic_json(sidecar, manifest)
    return fingerprint(manifest)


def model_provenance(root, adapter=None):
    """Bind base verification receipt and the actual inference adapter files."""
    base = Path(root) / 'artifacts/base'
    receipt = Path(root) / 'artifacts/base-verification.json'
    result = {'base_path': str(base.resolve()), 'base_verification_sha256': sha(receipt)}
    # Tokenizer/config changes must not hide behind an unchanged receipt.
    result['base_settings'] = {p.name: sha(p) for p in sorted(base.iterdir())
                               if p.is_file() and p.suffix in ('.json', '.txt', '.jinja')}
    if adapter:
        directory = Path(adapter).resolve()
        result['adapter'] = {'path': str(directory), 'files': {
            name: sha(directory / name) for name in ('adapters.safetensors', 'adapter_config.json')}}
    else:
        result['adapter'] = None
    return result


def script_provenance(script):
    versions = {}
    for package in ('mlx', 'mlx-lm', 'transformers', 'tokenizers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'script_sha256': sha(script), 'integrity_sha256': sha(__file__),
            'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                        'packages': versions}}


def validate_batch_prefix(batches, existing):
    """Atomic snapshots permit only whole original batches in a contiguous prefix."""
    missing_seen = False
    for batch in batches:
        present = sum(identifier in existing for identifier in batch)
        if present not in (0, len(batch)) or (present and missing_seen):
            raise IntegrityError('Output is not a complete original-batch prefix; refusing changed batch composition')
        if not present:
            missing_seen = True
