#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""The install-path test: LSDub in a clean ComfyUI, the way a user gets it. CPU only. CI runs it on every PR to main, before
every publish, weekly, and by hand (.github/workflows/install-path.yml); scripts/release.py waits for it.

  python scripts/install_path_check.py --comfyui PATH/TO/ComfyUI [--pack .] [--port 8199] [--demo-zip lsdub-demo-0.1.0.zip]
                                       [--hf-mirror DIR] [--save-mirror DIR] [--steps nodes,align,dub,mouth]

Run it with the ComfyUI venv's python (ComfyUI's requirements + this pack's requirements.txt, nothing else). It:
  1. copies the pack (this checkout, without .git) into ComfyUI/custom_nodes/comfyui-lsdub and fetches the demo assets;
  2. starts ComfyUI headless (--cpu) and checks that every LSDub node in the source registered;
  3. align: LSDub Align with aligner ctc from an EMPTY models/lsdub/wav2vec2-espeak. The recogniser is downloaded for real
     (exactly the four pinned files at the pinned revision, no safetensors), and phonemizer is never needed;
  4. dub: the demo workflow (BYO Clip and Song) at its defaults: Dub, Verify, Save Cut;
  5. mouth: the same Dub with Mouth Tweaks warp_events vocal, warp_hold nucleus-follow, warp_closures aligned-press,
     warp_aperture p95 (the Dub's auto path).
A dubbed shot that errors fails the test even though the Dub itself carries on with the source (on_fail=source).

--hf-mirror DIR serves the four recogniser files from DIR on a local Hugging Face endpoint (CI's cache, so the files are
fetched from the Hub once per revision); the client path (hf_hub_download into the empty folder) is unchanged.
--save-mirror DIR copies the downloaded files there after a real download, for that cache. Exit 0 = pass."""
import argparse
import glob
import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
RECOGNISER_DIR = os.path.join('lsdub', 'wav2vec2-espeak')
MOUTH = dict(warp_events='vocal', warp_hold='nucleus-follow', warp_closures='aligned-press', warp_aperture='p95')
BAD_LOG = [(re.compile(r'phonemizer', re.I), 'phonemizer was needed'),
           (re.compile(r'phone network not loaded'), 'the phone recogniser did not load (vocal events fell back to flux)'),
           (re.compile(r'shot \d+ failed: '), 'a dubbed shot raised (the Dub fell back to the source)'),
           (re.compile(r'IMPORT FAILED.*lsdub', re.I), 'the pack failed to import')]


class Fail(SystemExit):
    def __init__(self, msg): super().__init__(f'INSTALL-PATH FAIL: {msg}')


def say(msg): print(f'[install-path] {msg}', flush=True)


def pinned(pack):
    """the recogniser's pinned revision and file shas, read from the pack under test (lsdub_engine/_models.py)."""
    src = open(os.path.join(pack, 'lsdub_engine', '_models.py'), encoding='utf-8').read()
    i = src.index(f'"{REPO}"')
    rev = re.search(r'"revision":\s*"([0-9a-f]{40})"', src[i:]).group(1)
    files = dict(re.findall(r'"([\w.]+)":\s*"([0-9a-f]{64})"', src[i:src.index('}', src.index('"files"', i)) + 1]))
    return rev, files


def stand_in(fd, z, sha):
    """unpack a demo zip the way fetch_demo.py does, with the zip's own checksum (only after fetch_demo itself has failed)."""
    h = hashlib.sha256(open(z, 'rb').read()).hexdigest()
    if h != sha: raise Fail(f'demo stand-in {z}: sha256 {h} != {sha}')
    src = open(fd[1], encoding='utf-8').read()
    tmp = fd[1] + '.standin.py'
    import zipfile
    root = {n.split('/')[0] for n in zipfile.ZipFile(z).namelist()}.pop()             # lsdub-demo-<version>
    src = re.sub(r"^VERSION = '.*'$", f"VERSION = '{root.rsplit('-', 1)[-1]}'", src, flags=re.M)
    open(tmp, 'w', encoding='utf-8').write(re.sub(r"^SHA256 = '.*'$", f"SHA256 = '{sha}'", src, flags=re.M))
    try: subprocess.run([fd[0], tmp] + fd[2:] + ['--zip', z], check=True, stdout=subprocess.DEVNULL)
    finally: os.remove(tmp)


def node_ids(pack):
    ids = set()
    for p in glob.glob(os.path.join(pack, '*.py')):
        ids |= set(re.findall(r"node_id='(LSDub\w+)'", open(p, encoding='utf-8').read()))
    return ids


# ------------------------------------------------------------------------------------------------ the local hub mirror
def serve_mirror(folder, rev, files):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _file(self):
            m = re.match(rf'^/{re.escape(REPO)}/resolve/([^/]+)/([\w.]+)$', self.path.split('?')[0])
            if not m or m.group(1) not in (rev, 'main') or m.group(2) not in files: return None
            return os.path.join(folder, m.group(2))

        def _head(self, p):
            self.send_response(200)
            self.send_header('X-Repo-Commit', rev)
            self.send_header('ETag', f'"{files[os.path.basename(p)]}"')
            self.send_header('Content-Length', str(os.path.getsize(p)))
            self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()

        def do_HEAD(self):
            p = self._file()
            if not p or not os.path.isfile(p):
                self.send_error(404)
                return
            self._head(p)

        def do_GET(self):
            p = self._file()
            if not p or not os.path.isfile(p):
                self.send_error(404)
                return
            self._head(p)
            with open(p, 'rb') as f: shutil.copyfileobj(f, self.wfile)

    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f'http://127.0.0.1:{srv.server_address[1]}'


# ------------------------------------------------------------------------------------------------ ComfyUI
def api(port, path, body=None, timeout=30):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (loopback)
        return json.loads(r.read())


def run_prompt(port, prompt, label, timeout):
    try:
        pid = api(port, '/prompt', {'prompt': prompt, 'client_id': 'install-path'})['prompt_id']
    except urllib.error.HTTPError as e:
        raise Fail(f'{label}: ComfyUI refused the workflow: {e.read().decode(errors="replace")[:2000]}') from None
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = api(port, f'/history/{pid}').get(pid)
        if h and h.get('status', {}).get('completed') is not None and (h['status'].get('completed') or h['status'].get('status_str') == 'error'):
            if h['status'].get('status_str') != 'success':
                msgs = [m for m in h['status'].get('messages', []) if m[0] == 'execution_error']
                raise Fail(f'{label}: the workflow failed: {json.dumps(msgs)[:3000]}')
            say(f'{label}: done in {time.time() - t0:.0f} s')
            return h['outputs']
        time.sleep(3)
    raise Fail(f'{label}: no result after {timeout} s')


def texts(outputs, node):
    t = outputs.get(node, {}).get('text', [])
    return '\n'.join(t if isinstance(t, list) else [str(t)])


def workflow(name, aligner='sphinx', mouth=False, align_only=False):
    w = {
        '1': ('LSDubProject', dict(name=name, root='', mode='run', fps=25, song_offset=0, preset='landing', seed=0, force_shots='', backend='local')),
        '2': ('LoadVideo', dict(file='lsdub_demo_landing_clip.mp4')),
        '3': ('LSDubVideoSlot', dict(prompt='', negative_prompt='', seconds=5, project=['1', 0], video=['2', 0])),
        '4': ('LoadAudio', dict(audio='lsdub_demo_landing_song.wav')),
        '5': ('LoadAudio', dict(audio='lsdub_demo_landing_lead.wav')),
        '6': ('LSDubSongSlot', dict(tags='', seconds=60, bpm=120, keyscale='C major', project=['1', 0], audio=['4', 0], vocal=['5', 0])),
        '8': ('LSDubLyricsSlot', dict(text='Pleiadians bending space and time\nFour hundred light-years far away\nAlways so serious, never dance a day',
                                      project=['1', 0])),
        '9': ('LSDubAlign', dict(pronunciations='', stt_model='large-v3', transcript_json='lsdub_demo_landing_stt.json', aligner=aligner,
                                 project=['1', 0], song=['6', 0], lyrics=['8', 0])),
        '13': ('PreviewAny', dict(source=['9', 1])),
    }
    if not align_only:
        w.update({
            '7': ('LSDubShotSplit', dict(project=['1', 0], source=['3', 0])),
            '10': ('LSDubTriage', dict(samples=6, project=['1', 0], source=['3', 0], shots=['7', 0], song=['6', 0])),
            '11': ('PreviewAny', dict(source=['10', 2])),
            '14': ('LSDubBuildLipKit', dict(identity='red_nova', kit_dir='lsdub_demo_kits/red_nova.zip', shot_id=-1,
                                            keep='person, same face shape, skin tone, hair, eyewear and makeup', seeds=3, project=['1', 0])),
            '15': ('LSDubDub', dict(only_shots='', quality=18, on_fail='source', audio='source', vocalisations='skip', project=['1', 0],
                                    source=['3', 0], shots=['10', 0], song=['6', 0], align=['9', 0], kit=['14', 0])),
            '16': ('LSDubVerify', dict(project=['1', 0], source=['3', 0], render=['15', 0], shots=['10', 0], song=['6', 0], align=['9', 0])),
            '17': ('PreviewAny', dict(source=['16', 1])),
            '20': ('LSDubSaveCut', dict(watermark=True, quality=18, project=['1', 0], render=['15', 0])),
        })
        if mouth:
            w['21'] = ('LSDubTweaks', dict(lead=0.06, word_close=0.06, face_target=512.0, snap=True, auto=True, continuity=True, settings='', **MOUTH))
            w['15'][1]['tweaks'] = ['21', 0]
    return {k: {'class_type': c, 'inputs': i} for k, (c, i) in w.items()}


def check_log(log, since, label):
    with open(log, encoding='utf-8', errors='replace') as f:
        f.seek(since)
        new = f.read()
    for rx, why in BAD_LOG:
        m = rx.search(new)
        if m:
            ln = new[max(0, new.rfind('\n', 0, m.start()) + 1):new.find('\n', m.end())]
            raise Fail(f'{label}: {why}: {ln.strip()[:500]}')
    return new


def check_dub(comfy, name, outputs, label):
    proj = os.path.join(comfy, 'output', 'lsdub', name)
    renders = glob.glob(os.path.join(proj, 'shots', '*', '*', 'render.mp4')) + glob.glob(os.path.join(proj, 'shots', '*', '*', '*.mp4'))
    errors = []
    for d in glob.glob(os.path.join(proj, 'shots', '*', '*', 'done.json')):
        j = json.load(open(d))
        if j.get('error'): errors.append(f"{os.path.relpath(d, proj)}: {j['error']}")
    if errors: raise Fail(f'{label}: dubbed shots raised: {errors}')
    if not renders: raise Fail(f'{label}: no shot was dubbed (no render under {proj}/shots)')
    cuts = [f for f in glob.glob(os.path.join(comfy, 'output', '**', '*.mp4'), recursive=True) if name in f and 'final' in f]
    if not cuts: raise Fail(f'{label}: Save Cut wrote no file')
    rep = texts(outputs, '17')
    if 'shot' not in rep.lower(): raise Fail(f'{label}: Verify gave no report: {rep[:300]!r}')
    say(f'{label}: {len(renders)} render file(s), cut {os.path.relpath(cuts[0], comfy)}; Verify: {rep.splitlines()[0][:120] if rep else ""}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--comfyui', required=True)
    ap.add_argument('--pack', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--port', type=int, default=8199)
    ap.add_argument('--demo-zip')
    ap.add_argument('--hf-mirror')
    ap.add_argument('--save-mirror')
    ap.add_argument('--demo-fallback', help='a demo zip to use if fetch_demo.py fails (the step still fails)')
    ap.add_argument('--demo-fallback-sha', default='', help="the fallback zip's sha256")
    ap.add_argument('--steps', default='nodes,align,dub,mouth')
    ap.add_argument('--timeout', type=int, default=3600, help='seconds per workflow')
    a = ap.parse_args()
    comfy, pack, steps = os.path.abspath(a.comfyui), os.path.abspath(a.pack), set(a.steps.split(','))
    rev, files = pinned(pack)
    say(f'pack {pack}; recogniser {REPO}@{rev[:8]} ({len(files)} files)')

    dst = os.path.join(comfy, 'custom_nodes', 'comfyui-lsdub')
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(pack, dst, ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'dist', '*.zip'))
    failures = []
    fd = [sys.executable, os.path.join(dst, 'scripts', 'fetch_demo.py'), '--comfyui', comfy]
    r = subprocess.run(fd + (['--zip', os.path.abspath(a.demo_zip)] if a.demo_zip else []), capture_output=True, text=True)
    if r.returncode:
        failures.append(f'demo: scripts/fetch_demo.py fails for a user: {(r.stderr or r.stdout).strip().splitlines()[-1][:400]}')
        say(failures[-1])
        if not a.demo_fallback: raise Fail(failures[-1])
        say(f'demo: carrying on with the stand-in {a.demo_fallback} (sha256 {a.demo_fallback_sha[:12]}) so the other steps still report')
        stand_in(fd, a.demo_fallback, a.demo_fallback_sha)
    else:
        say('demo: fetch_demo.py ok')
    rec = os.path.join(comfy, 'models', RECOGNISER_DIR)
    if os.path.exists(rec): raise Fail(f'{rec} is not empty: the align step needs an empty models folder')

    env = dict(os.environ, PYTHONUNBUFFERED='1')
    if a.hf_mirror and all(os.path.isfile(os.path.join(a.hf_mirror, f)) for f in files):
        env['HF_ENDPOINT'] = serve_mirror(os.path.abspath(a.hf_mirror), rev, files)
        say(f'recogniser files from the cached mirror {a.hf_mirror} on {env["HF_ENDPOINT"]}')
    else:
        say('recogniser files from the Hugging Face Hub')
    log = os.path.join(comfy, 'install-path-comfyui.log')
    lf = open(log, 'w')
    proc = subprocess.Popen([sys.executable, 'main.py', '--cpu', '--listen', '127.0.0.1', '--port', str(a.port), '--disable-auto-launch'],
                            cwd=comfy, env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None: raise Fail(f'ComfyUI exited ({proc.returncode}); see {log}')
            try:
                info = api(a.port, '/object_info', timeout=60)
                break
            except (OSError, urllib.error.URLError):
                if time.time() - t0 > 600: raise Fail('ComfyUI did not start in 600 s') from None
                time.sleep(2)
        want, got = node_ids(pack), {k for k in info if k.startswith('LSDub')}
        check_log(log, 0, 'startup')
        if want - got: raise Fail(f'nodes not registered: {sorted(want - got)} (see {log})')
        say(f'nodes: all {len(want)} LSDub nodes registered')

        def step(label, fn):
            try: fn()
            except Fail as e:
                failures.append(str(e.code))
                say(str(e.code))

        def align():
            pos = os.path.getsize(log)
            out = run_prompt(a.port, workflow('ip-align', aligner='ctc', align_only=True), 'align (ctc, empty models folder)', a.timeout)
            check_log(log, pos, 'align')
            have = sorted(os.path.relpath(p, rec) for p in glob.glob(os.path.join(rec, '**', '*'), recursive=True) if os.path.isfile(p))
            snap = os.path.join(rec, 'models--' + REPO.replace('/', '--'), 'snapshots', rev)
            if any(p.endswith('.safetensors') for p in have): raise Fail(f'align: an unpinned safetensors was downloaded: {have}')
            for f, sha in files.items():
                p = os.path.join(snap, f)
                if not os.path.isfile(p): raise Fail(f'align: pinned file {f} missing from {snap}; have {have}')
                h = hashlib.sha256()
                with open(p, 'rb') as fh:
                    for b in iter(lambda: fh.read(1 << 20), b''): h.update(b)
                if h.hexdigest() != sha: raise Fail(f'align: {f} sha256 {h.hexdigest()} != pinned {sha}')
            blobs = [p for p in have if '/blobs/' in p or p.startswith('models--') and '/blobs' in p]
            if len(blobs) > len(files): raise Fail(f'align: more than the {len(files)} pinned files were downloaded: {blobs}')
            rep = texts(out, '13')
            if 'aligner ctc' not in rep: raise Fail(f'align: the lyric check does not say aligner ctc: {rep[:400]!r}')
            say(f'align: four pinned files at {rev[:8]}, sha256 ok, no safetensors; {rep.splitlines()[2] if len(rep.splitlines()) > 2 else rep[:100]}')
            if a.save_mirror and 'HF_ENDPOINT' not in env:
                os.makedirs(a.save_mirror, exist_ok=True)
                for f in files: shutil.copyfile(os.path.join(snap, f), os.path.join(a.save_mirror, f))
                say(f'align: saved the four files to {a.save_mirror} for the cache')

        def dub(name, label, kw):
            pos = os.path.getsize(log)
            out = run_prompt(a.port, workflow(name, **kw), label, a.timeout)
            check_log(log, pos, label)
            check_dub(comfy, name, out, label)

        if 'align' in steps: step('align', align)
        if 'dub' in steps: step('dub', lambda: dub('ip-dub-defaults', 'dub at defaults', {}))
        if 'mouth' in steps:
            step('mouth', lambda: dub('ip-dub-mouth', 'dub with the mouth Combos (' + ', '.join(f'{k} {v}' for k, v in MOUTH.items()) + ')',
                                      dict(aligner='ctc', mouth=True)))
        if failures: raise Fail(f'{len(failures)} step(s) failed:\n  ' + '\n  '.join(failures))
        say('PASS')
    finally:
        proc.terminate()
        try: proc.wait(30)
        except subprocess.TimeoutExpired: proc.kill()
        lf.close()


if __name__ == '__main__':
    main()
