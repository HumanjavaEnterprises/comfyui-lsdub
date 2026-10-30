# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""ff.py: the ONE place the engine starts an external process, always as an argv list, never through a shell.

  run / popen   ffmpeg and ffprobe (every node path uses only these)
                ffmpeg falls back to the static binary of the `imageio-ffmpeg` package when none is on PATH; ffprobe
                (which imageio-ffmpeg does not ship) falls back to an emulation over `ffmpeg -i` for the probes the
                pack makes (format=duration, format_tags=..., stream=index/width/height/r_frame_rate; csv/json/default)
  tool          the opt-in CLI aligners: rhubarb (lip cues) and mfa (Montreal Forced Aligner); never on a node path
  python        a second Python interpreter for speech-to-text (`lsdub align --whisper-python`); CLI only, validated

Every other engine step runs in-process."""
import json
import os
import re
import shutil
import subprocess
from fractions import Fraction

PIPE, DEVNULL, CalledProcessError = subprocess.PIPE, subprocess.DEVNULL, subprocess.CalledProcessError
ALLOWED = ('ffmpeg', 'ffprobe')
TOOLS = ('rhubarb', 'mfa')


def _resolve(exe, allowed):
    name = os.path.basename(exe)
    if name not in allowed:
        raise ValueError(f'ff: only {allowed} may be started, got {exe!r}')
    path = exe if os.path.isabs(exe) else which(exe)
    if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise FileNotFoundError(f'ff: {exe} is not installed (or not executable)')
    return path


def _argv(argv, allowed=ALLOWED):
    argv = [str(a) for a in argv]
    if not argv:
        raise ValueError('ff: empty argv')
    return [_resolve(argv[0], allowed), *argv[1:]]


def which(name):
    """the executable for a tool name: PATH first; for ffmpeg, then the imageio-ffmpeg static binary. None if absent."""
    p = shutil.which(name)
    if p or name != 'ffmpeg': return p
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001  (not installed, or no binary for this platform)
        return None
    return p if p and os.path.isfile(p) else None


def _emulate_probe(argv):
    """True when argv is an ffprobe call that must be emulated (ffprobe absent, ffmpeg present)."""
    return (str(argv[0]) == 'ffprobe' and not shutil.which('ffprobe')) if argv else False


def run(argv, **kw):
    """subprocess.run for ffmpeg / ffprobe (argv list, shell=False)."""
    kw.pop('shell', None)
    if _emulate_probe(argv): return probe(argv, **kw)
    return subprocess.run(_argv(argv), shell=False, **kw)  # noqa: S603


def popen(argv, **kw):
    """subprocess.Popen for ffmpeg / ffprobe (argv list, shell=False), e.g. a raw-frame pipe into the encoder."""
    kw.pop('shell', None)
    if _emulate_probe(argv): raise FileNotFoundError('ff: ffprobe is not installed (only ff.run can emulate it)')
    return subprocess.Popen(_argv(argv), shell=False, **kw)  # noqa: S603


# --- ffprobe emulation over `ffmpeg -i` (stderr parse) ------------------------------------------------------------
_DUR = re.compile(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)')
_STREAM = re.compile(r'^\s*Stream #\d+:(\d+)\S*: (Video|Audio|Subtitle|Data|Attachment): (.*)$')
_KIND = {'Video': 'v', 'Audio': 'a', 'Subtitle': 's', 'Data': 'd', 'Attachment': 't'}


def _rate(fps):
    """'25' -> '25/1', '29.97' -> '30000/1001', '12.5' -> '25/2' (ffmpeg prints the rate rounded)."""
    f = float(fps)
    if abs(f - round(f)) < 1e-3: return f'{round(f)}/1'
    n = round(f * 1.001)
    if n and abs(n * 1000 / 1001 - f) < 6e-3: return f'{n * 1000}/1001'
    q = Fraction(f).limit_denominator(1001)
    return f'{q.numerator}/{q.denominator}'


def _parse_info(err):
    """`ffmpeg -i` stderr -> {'format': {...}, 'tags': {...}, 'streams': [{...}]} for input #0."""
    info = {'format': {}, 'tags': {}, 'streams': []}
    in0, tags_open = False, False
    for ln in err.splitlines():
        if ln.startswith('Input #'):
            in0 = ln.startswith('Input #0')
            continue
        if not in0 or ln.startswith('Output #'):
            in0 = False if ln.startswith('Output #') else in0
            continue
        m = _DUR.search(ln)
        if m:
            h, mi, se = m.groups()
            info['format']['duration'] = f'{int(h) * 3600 + int(mi) * 60 + float(se):.6f}'
            tags_open = False
            continue
        if 'Duration: N/A' in ln:
            info['format']['duration'] = 'N/A'
            tags_open = False
            continue
        if ln.strip() == 'Metadata:' and not info['streams'] and 'duration' not in info['format']:
            tags_open = True
            continue
        if tags_open and ' : ' in ln:
            k, v = ln.split(' : ', 1)
            k = k.strip()
            if k: info['tags'][k] = v.strip()
            elif info['tags']: info['tags'][list(info['tags'])[-1]] += '\n' + v.strip()
            continue
        m = _STREAM.match(ln)
        if m:
            idx, kind, rest = int(m.group(1)), m.group(2), m.group(3)
            s = {'index': idx, 'codec_type': kind.lower(), '_k': _KIND[kind]}
            if kind == 'Video':
                wh = re.search(r'(?:^|[ ,])(\d{1,5})x(\d{1,5})(?:[ ,]|$)', rest)
                if wh: s['width'], s['height'] = int(wh.group(1)), int(wh.group(2))
                fr = re.search(r'([\d.]+)(?:k)? tbr', rest) or re.search(r'([\d.]+) fps', rest)
                if fr: s['r_frame_rate'] = _rate(fr.group(1))
            info['streams'].append(s)
    return info


def _opt(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def probe(argv, capture_output=False, text=False, check=False, stdout=None, stderr=None, **_):
    """ffprobe stand-in for the probes the pack makes: -select_streams {a,v,v:0}, -show_entries {format=K,
    format_tags=K, stream=K,...}, -of {csv=p=0, json, default=nw=1:nk=1}; the input is the last argument."""
    argv = [str(a) for a in argv]
    path, sel, ents, of = argv[-1], _opt(argv, '-select_streams'), _opt(argv, '-show_entries') or '', _opt(argv, '-of') or 'default'
    ff = which('ffmpeg')
    if not ff: raise FileNotFoundError('ff: neither ffprobe nor ffmpeg is installed')
    r = subprocess.run([ff, '-hide_banner', '-nostdin', '-i', path], capture_output=True, text=True,  # noqa: S603
                       encoding='utf-8', errors='replace', shell=False)
    ok = 'Input #0' in r.stderr
    out, err = '', '' if ok else (r.stderr.strip().splitlines() or ['ffprobe emulation: no input'])[-1] + '\n'
    if ok:
        info = _parse_info(r.stderr)
        sec, _, keys = ents.partition('=')
        keys = [k for k in keys.split(',') if k]
        if sec == 'stream':
            ss = info['streams']
            if sel:
                kind, _, n = sel.partition(':')
                ss = [s for s in ss if s['_k'] == kind]
                if n: ss = ss[int(n):int(n) + 1]
            rows = [{k: s[k] for k in keys if k in s} for s in ss]
            if of.startswith('json'):
                out = json.dumps({'streams': rows}, indent=4) + '\n'
            elif of.startswith('csv'):
                out = ''.join(','.join(str(v) for v in row.values()) + '\n' for row in rows)
            else:
                out = ''.join(f'{k}={v}\n' if 'nk=1' not in of else f'{v}\n' for row in rows for k, v in row.items())
        else:
            src = info['tags'] if sec == 'format_tags' else info['format']
            row = {k: src[k] for k in keys if k in src}
            if of.startswith('json'):
                out = json.dumps({'format': ({'tags': row} if sec == 'format_tags' else row)}, indent=4) + '\n'
            elif of.startswith('csv'):
                out = ','.join(row.values()) + '\n' if row else ''
            else:
                pre = 'TAG:' if sec == 'format_tags' else ''
                out = ''.join(f'{v}\n' if 'nk=1' in of else f'{pre}{k}={v}\n' for k, v in row.items())
    code = 0 if ok else 1
    if check and code: raise subprocess.CalledProcessError(code, argv, out, err)
    if not (capture_output or stdout == PIPE):
        out = None
    if not (capture_output or stderr == PIPE):
        err = None
    if not text:
        out = out.encode() if out is not None else None
        err = err.encode() if err is not None else None
    return subprocess.CompletedProcess(argv, code, out, err)


def tool(argv, **kw):
    """subprocess.run for an opt-in aligner (rhubarb, mfa): argv[0] is the tool name or an absolute path to it."""
    kw.pop('shell', None)
    return subprocess.run(_argv(argv, TOOLS), shell=False, **kw)  # noqa: S603


def python(interpreter, module, args, **kw):
    """run `<interpreter> -m <module> args` where interpreter is an existing executable named python*; module must be
    one of the engine's own (lsdub_engine.*)."""
    kw.pop('shell', None)
    if not module.startswith('lsdub_engine.'):
        raise ValueError(f'ff: only engine modules may be run, got {module!r}')
    exe = os.path.realpath(interpreter)
    if not os.path.basename(exe).startswith('python') or not os.path.isfile(exe) or not os.access(exe, os.X_OK):
        raise ValueError(f'ff: {interpreter!r} is not a Python interpreter')
    return subprocess.run([exe, '-m', module, *[str(a) for a in args]], shell=False, **kw)  # noqa: S603
