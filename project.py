# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""project.py: the LSDub project folder and its input-hash cache ("keep everything that renders").

Every stage writes into <root>/<name>/<stage>/<key>/ and marks it finished with a done.json. The key is a hash of the
node id, the engine version, the node's canonical inputs and the hashes of what came from upstream, so a re-run with
the same inputs finds its folder and skips the work, and a changed input writes a NEW key folder. Nothing is deleted.
Per-shot stages keep one folder per shot (shots/shot_###/<key>/) and skip shots whose done.json says pass=true;
force_shots ("21,23") re-opens just those (their done.json is set aside as done.<n>.json, the renders stay).

Pure Python: no ComfyUI imports here, so the cache is unit-tested on its own.
"""
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field

from .lsdub_engine import __version__ as ENGINE_VERSION

MODES = ('run', 'test_learn')
STAGES = ('slots/video', 'slots/still', 'slots/song', 'slots/lyrics', 'audio/split', 'audio/lead', 'audio/align',
          'video/shots', 'video/cast', 'video/triage', 'mouth/kits', 'mouth/masks', 'shots',
          'qa/verify', 'qa/scorecard', 'qa/syncqa', 'qa/fixit', 'final', 'logs')


# ---------------------------------------------------------------------------------------------------- hashing
def _canon(x):
    """JSON-safe canonical form: dict keys sorted, tuples as lists, floats rounded to 6 places (stable across runs)."""
    if isinstance(x, dict): return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)): return [_canon(v) for v in x]
    if isinstance(x, float): return round(x, 6)
    if hasattr(x, 'cache_key'): return x.cache_key()
    if x is None or isinstance(x, (bool, int, str)): return x
    return repr(x)


def canonical(x):
    return json.dumps(_canon(x), sort_keys=True, separators=(',', ':'))


def text_hash(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


def file_hash(path, chunk=1 << 20):
    """sha1 of a file's bytes (an asset dropped into a slot)."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''): h.update(b)
    return h.hexdigest()


def stage_key(node_id, inputs=None, upstream=(), engine_version=ENGINE_VERSION):
    """key = sha1(node_id, engine_version, canonical inputs, upstream hashes), 16 hex chars."""
    blob = canonical({'node': node_id, 'engine': engine_version, 'inputs': inputs or {}, 'upstream': list(upstream)})
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def parse_force_shots(s):
    """'21, 23' / '21 23' / [21, 23] -> (21, 23); '' -> ()."""
    if not s: return ()
    if isinstance(s, (list, tuple)): return tuple(sorted({int(x) for x in s}))
    return tuple(sorted({int(x) for x in s.replace(',', ' ').split() if x.strip()}))


# ---------------------------------------------------------------------------------------------------- project
@dataclass(frozen=True)
class Project:
    """LSDUB_PROJECT: where everything for one cut lives, and the run-wide settings."""
    root: str
    name: str
    mode: str = 'run'
    fps: float = 25.0
    song_offset: float = 0.0
    vocalisations: str = 'skip'
    seed: int = 0
    preset: str = 'none'
    backend: str = 'local'
    force_shots: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if self.mode not in MODES: raise ValueError(f'mode must be one of {MODES}, got {self.mode!r}')
        if not self.name or os.sep in self.name or self.name in ('.', '..'): raise ValueError(f'bad project name {self.name!r}')

    @property
    def dir(self):
        return os.path.join(self.root, self.name)

    def cache_key(self):
        """what downstream keys depend on: not force_shots (forcing a shot must not move every key), not mode (only the
        dub reads it, so test_learn and run share their analysis) and not seed (only the generator slots use it, and
        they key on it themselves)."""
        d = asdict(self)
        for k in ('force_shots', 'root', 'mode', 'seed', 'backend'): d.pop(k)
        return text_hash(canonical(d))

    def ensure(self):
        os.makedirs(self.dir, exist_ok=True)
        for s in ('slots', 'audio', 'video', 'mouth', 'shots', 'qa', 'final', 'logs'): os.makedirs(os.path.join(self.dir, s), exist_ok=True)
        p = os.path.join(self.dir, 'project.json')
        d = asdict(self)
        d['force_shots'] = list(self.force_shots)
        d['engine_version'] = ENGINE_VERSION
        tmp = p + '.tmp'
        with open(tmp, 'w') as f: json.dump(d, f, indent=1)
        os.replace(tmp, p)
        return self

    def stage(self, stage, key):
        return StageCache(os.path.join(self.dir, stage, key))

    def shot(self, shot_id, key):
        return StageCache(os.path.join(self.dir, 'shots', f'shot_{int(shot_id):03d}', key))

    def to_json(self):
        d = asdict(self)
        d['force_shots'] = list(self.force_shots)
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d['force_shots'] = tuple(d.get('force_shots') or ())
        d.pop('engine_version', None)
        return cls(**d)


class StageCache:
    """one <stage>/<key>/ folder: done.json marks it finished (pass true/false + whatever the stage reports)."""

    def __init__(self, path):
        self.dir = path

    @property
    def done_path(self):
        return os.path.join(self.dir, 'done.json')

    def path(self, *parts):
        os.makedirs(self.dir, exist_ok=True)
        return os.path.join(self.dir, *parts)

    def done(self):
        try:
            with open(self.done_path) as f: return json.load(f)
        except (OSError, ValueError):
            return None

    def is_done(self, need_pass=True):
        d = self.done()
        return d is not None and (bool(d.get('pass', False)) or not need_pass)

    def mark_done(self, passed=True, **info):
        os.makedirs(self.dir, exist_ok=True)
        tmp = self.done_path + '.tmp'
        with open(tmp, 'w') as f: json.dump({'pass': bool(passed), 'time': time.time(), **_canon(info)}, f, indent=1)
        os.replace(tmp, self.done_path)

    def reopen(self):
        """set done.json aside (done.<n>.json) so the stage runs again; the outputs stay on disk."""
        if not os.path.exists(self.done_path): return False
        n = 1
        while os.path.exists(os.path.join(self.dir, f'done.{n}.json')): n += 1
        os.replace(self.done_path, os.path.join(self.dir, f'done.{n}.json'))
        return True


def take_force(project, scope='dub'):
    """force_shots acts ONCE per value: -> the shots to re-open now (the project's force_shots when that value differs
    from the last one applied in this scope, else ()); the applied value is recorded in <project>/force.json."""
    p = os.path.join(project.dir, 'force.json')
    try:
        with open(p) as f: done = json.load(f)
    except (OSError, ValueError):
        done = {}
    now = ','.join(map(str, project.force_shots))
    if done.get(scope, '') == now: return ()
    done[scope] = now
    os.makedirs(project.dir, exist_ok=True)
    tmp = p + '.tmp'
    with open(tmp, 'w') as f: json.dump(done, f)
    os.replace(tmp, p)
    return tuple(project.force_shots)


def shots_todo(project, key, shot_ids, forced=None):
    """which shots of a per-shot stage still need work: the forced shots (default: project.force_shots) are re-opened
    first; a shot is skipped only when its done.json says pass=true (a failed shot is redone)."""
    for s in (project.force_shots if forced is None else forced):
        if s in shot_ids: project.shot(s, key).reopen()
    return [s for s in shot_ids if not project.shot(s, key).is_done()]
