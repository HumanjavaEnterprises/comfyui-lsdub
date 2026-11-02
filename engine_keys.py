# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""engine_keys.py: what part of the engine a stage runs, so a change to the paint does not throw away a take or a kit.

A stage's key used to carry the whole engine's tree hash: any change anywhere re-ran every stage. The stages before the dub now carry the
hash of the engine files their code can actually reach, read off the source: from the functions the stage calls, through every name they use,
across modules, classes whole, including imports inside functions. A step named by string (steps.call("paint")) is followed through
steps.STEPS. What a stage cannot reach cannot change what it makes; what it can reach moves its key, and everything downstream keys on the
hash of the stage's output. The dub and after keep the whole engine's version.

tests/test_engine_keys.py holds this to account: every engine function a node calls is in STAGES, nothing in the engine dispatches by name
outside steps.py, and a change to a file moves exactly the stages that reach it."""
import ast
import hashlib
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lsdub_engine')

# stage -> the engine functions its node calls (module:function)
STAGES = {
    'LSDubLyricsSlot': (),
    'LSDubSongSlot': ('io.ff:run',),
    'LSDubStillSlot': ('io.ff:run',),
    'LSDubVideoSlot': ('io.ff:run',),
    'LSDubPerformance': ('core.medium:read', 'io.ff:run'),
    'LSDubShotSplit': ('audits.qa:detect_shots', 'audits.qa:sheet'),
    'LSDubCast': ('audits.qa:build_db', 'audits.qa:read_frames', 'audits.qa:identify', 'audits.qa:sheet', 'core.cast:Faces', 'core.cast:Embedder'),
    'LSDubTriage': ('audits.qa:triage', 'audits.qa:write_triage', 'core.cast:Faces', 'core.cast:Embedder', 'core.cast:load_db'),
    'LSDubBuildLipKit': ('core.mouth_library:main', 'core.mouth_library:gen', 'core.mouth_library:viseme_prompt'),
    'LSDubAlign': ('core.lyric_align:main', 'core.lyric_align:plain_lines', 'core.lyric_align:stt', 'core.viseme_track:main', 'core.vocalisations:main',
                   'core.vocal_retime:retime', 'core.vocal_retime:read_mono', 'core.lead_sheet:read_sheet', 'core.lead_sheet:honour', 'core.lead_sheet:write',
                   'core.lyric_check:word_table', 'core.lyric_check:confidence', 'core.lyric_check:short_reason', 'core.lyric_check:WEAK_BELOW', 'core.steps:call'),
}


def _is_main(n):
    """`if __name__ == "__main__":`: only when the file is run as a script, which no stage does."""
    return isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and getattr(n.test.left, 'id', '') == '__name__'


class Reach:
    def __init__(self, root=ROOT):
        self.root, self.files, self.trees = root, {}, {}
        for d, _, fs in os.walk(root):
            for f in fs:
                if f.endswith('.py'):
                    p = os.path.join(d, f)
                    m = os.path.relpath(p, root)[:-3].replace(os.sep, '.')
                    m = m[: -len('.__init__')] if m.endswith('.__init__') else m
                    self.files[m] = p
        self.seen, self.mods = set(), set()
        self.steps = self._steps()

    def tree(self, m):
        if m not in self.trees:
            with open(self.files[m], encoding='utf-8') as f:
                self.trees[m] = ast.parse(f.read())
        return self.trees[m]

    def _steps(self):
        for n in self.tree('core.steps').body:
            if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'STEPS':
                d = ast.literal_eval(n.value)
                return {k: v[0] for k, v in d.items()}
        raise RuntimeError('engine_keys: core/steps.py has no STEPS table')

    def _base(self, m, level):
        parts = m.split('.') if m != '__init__' else []
        pkg = parts if m and self.files.get(m, '').endswith('__init__.py') else parts[:-1]
        return pkg[: len(pkg) - (level - 1)]

    def imports(self, m, nodes):
        """alias -> (module, name or None) for the imports among nodes."""
        out = {}
        for top in nodes:
            for n in ast.walk(top):
                if isinstance(n, ast.ImportFrom) and n.level:
                    b = self._base(m, n.level) + (n.module.split('.') if n.module else [])
                    for a in n.names:
                        sub = '.'.join(b + [a.name])
                        if sub in self.files:
                            out[a.asname or a.name] = (sub, None)
                        elif '.'.join(b) in self.files:
                            out[a.asname or a.name] = ('.'.join(b), a.name)
        return out

    def defs(self, m):
        return {n.name: n for n in self.tree(m).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}

    def module(self, m):
        """a reached module: its top level runs on import, whatever it defines."""
        if ('', m) in self.seen:
            return
        self.seen.add(('', m))
        self.mods.add(m)
        body = [n for n in self.tree(m).body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not _is_main(n)]
        self.walk(m, body)

    def name(self, m, name):
        if (m, name) in self.seen or m not in self.files:
            return
        self.seen.add((m, name))
        self.module(m)
        d = self.defs(m).get(name)
        if d is not None:
            self.walk(m, [d])
            return
        for n in self.tree(m).body:  # a constant or an alias at the top level
            if isinstance(n, ast.ImportFrom):
                al = self.imports(m, [n])
                if name in al:
                    self.name(*al[name]) if al[name][1] else self.module(al[name][0])

    def walk(self, m, nodes):
        al = {**self.imports(m, self.tree(m).body), **self.imports(m, nodes)}
        top = self.defs(m)
        for root in nodes:
            for n in ast.walk(root):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in al and al[n.value.id][1] is None:
                    self.name(al[n.value.id][0], n.attr)
                elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute):  # pkg.sub.fn through an imported package
                    pass
                elif isinstance(n, ast.Name):
                    if n.id in al:
                        self.name(*al[n.id]) if al[n.id][1] else self.module(al[n.id][0])
                    elif n.id in top:
                        self.name(m, n.id)
                elif isinstance(n, ast.Call):
                    f = n.func
                    called = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', '')
                    if called == 'call' and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value in self.steps:
                        self.name(self.steps[n.args[0].value], 'main')


def reached(entries, root=ROOT):
    r = Reach(root)
    for e in entries:
        m, f = e.split(':')
        r.name(m, f)
    return r.mods, r


_cache = {}


def part(stage, version, root=ROOT):
    """`version`, the release, plus the hash of the engine files stage can reach."""
    if (stage, root) not in _cache:
        mods, r = reached(STAGES[stage], root)
        h = hashlib.sha256()
        for m in sorted(mods):
            with open(r.files[m], 'rb') as f:
                h.update(m.encode() + b'\0' + f.read() + b'\0')
        _cache[stage, root] = h.hexdigest()[:12]
    return f'{version}+{_cache[stage, root]}'
