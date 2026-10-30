# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""MOUTH lane: the lip kit, the mouth controls, occluder masks, the dub itself (per shot, resumable) and continuity.

LSDub Dub, per shot of the triage: DUB / DUB-MEDIUM -> `sing --auto` (crop to a 512 px face, warp at 2x fps, paste
the lower face back, measured gate; the native full-frame warp is also run and the higher measured sync wins), then
the continuity pass; WIDE / NO -> the source frames, untouched. Every shot is saved under shots/shot_###/<key>/ with a
done.json, so an interrupted dub resumes and a re-run only redoes changed, failed or forced shots. The shots are then
assembled into the cut with the source's audio, and the cut is watermarked."""
import json
import os
import shutil
import sys

from comfy_api.v0_0_2 import io

from . import lsdub_types as T
from . import media
from .media import kit_folder
from .lsdub_engine import presets
from .lsdub_engine._inproc import call
from .project import file_hash, shots_todo, stage_key, text_hash

DUB_VERDICTS = ('DUB', 'DUB-MEDIUM')
SING_EXTRA = ['--warp-track-mode', 'chain']                 # the standard sing recipe's track mode


def _kit_flags(project, identity):
    """the preset's warp flags for this face (opt-in per face), minus the lip-lock geometry, which is per take."""
    f = list(presets.get(project.preset).paint_flags.get(identity, ()))
    if '--warp-lip-lock' in f:
        i = f.index('--warp-lip-lock')
        del f[i:i + 2]
    return f


class LSDubBuildLipKit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubBuildLipKit', display_name='LSDub Lip Kit', category='LSDub/mouth',
            description='One singer\'s mouth kit (13 mouth shapes from their own face): the "mouth LoRA" every dub needs. '
                        'Bring a kit folder, or build one from a shot of the source with Qwen-Image-Edit-2511 (open '
                        'weights; local GPU or Comfy Cloud per the project, ~39 edits).',
            hidden=[io.Hidden.api_key_comfy_org],
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                io.String.Input('identity', default='', tooltip='Who this kit is for: the singer\'s cast name.'),
                io.String.Input('kit_dir', default='', tooltip='A finished kit folder, or a .zip of one (a zip travels with '
                                '`comfy deploy run`; a folder does not). Leave empty to build one from the source.'),
                T.LSDubSource.Input('source', optional=True, tooltip='Build from this video: the frame with the most closed '
                                                                    'mouth of the chosen shot is the base face.'),
                T.LSDubShots.Input('shots', optional=True, tooltip='The shots, to pick shot_id from.'),
                io.Int.Input('shot_id', default=-1, min=-1, max=100000, optional=True,
                             tooltip='The shot to build from (a close-up of this singer); -1 = the whole source.'),
                io.String.Input('keep', default='person, same face shape, skin tone, hair, eyewear and makeup', optional=True,
                                tooltip='Who to keep: describes the singer, after "the same ...", so the edits keep the face.'),
                io.Int.Input('seeds', default=3, min=1, max=6, optional=True,
                             tooltip='Takes per mouth shape; the best-measured take is kept.'),
            ],
            outputs=[T.LSDubKit.Output(display_name='kit', tooltip='The lip kit: folder, identity, hash.')],
        )

    @classmethod
    def _hash(cls, kit_dir):
        h = []
        for d, _, fs in sorted(os.walk(kit_dir)):
            for f in sorted(fs):
                p = os.path.join(d, f)
                h.append(f'{os.path.relpath(p, kit_dir)}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}')
        return text_hash('\n'.join(h))

    @classmethod
    def _build_key(cls, project, identity, source, shots, shot_id, keep, seeds):
        span = None
        if shots is not None and shot_id is not None and shot_id >= 0:
            s = next((x for x in shots.shots if x['shot'] == shot_id), None)
            span = [s['f0'], s['f1']] if s else None
        return stage_key('LSDubBuildLipKit', {'id': identity.strip(), 'span': span, 'keep': keep, 'seeds': seeds,
                                              'seed': project.seed, 'backend': project.backend}, [source.hash])

    @classmethod
    def fingerprint_inputs(cls, project, identity, kit_dir, source=None, shots=None, shot_id=-1, keep='', seeds=3):
        kit_dir = kit_folder(kit_dir) if kit_dir else kit_dir
        if kit_dir: return stage_key('LSDubBuildLipKit', {'id': identity, 'dir': kit_dir,
                                                          'h': cls._hash(kit_dir) if os.path.isdir(kit_dir) else ''})
        return cls._build_key(project, identity, source, shots, shot_id, keep, seeds) if source is not None else 'none'

    @classmethod
    async def execute(cls, project, identity, kit_dir, source=None, shots=None, shot_id=-1,
                      keep='person, same face shape, skin tone, hair, eyewear and makeup', seeds=3):
        from . import devices as _dev
        _dev.ensure_face_models()
        if not identity.strip(): raise ValueError('LSDub Lip Kit: set identity (the singer\'s cast name)')
        if kit_dir:
            kit_dir = kit_folder(kit_dir)
            if not os.path.isdir(kit_dir): raise FileNotFoundError(f'LSDub Lip Kit: no folder {kit_dir}')
            return io.NodeOutput(T.Kit(dir=kit_dir, identity=identity.strip(), hash=cls._hash(kit_dir)))
        if source is None: raise ValueError('LSDub Lip Kit: set kit_dir, or connect a source to build a kit from')
        import asyncio
        import argparse
        from . import generate, graphs, media
        from .lsdub_engine.core import mouth_library as ML
        st = project.stage(f'mouth/kits/{identity.strip()}', cls._build_key(project, identity, source, shots, shot_id, keep, seeds))
        D = st.path('kit')
        if not st.is_done():
            clip = source.path
            if shots is not None and shot_id >= 0:
                s = next((x for x in shots.shots if x['shot'] == shot_id), None)
                if s is None: raise ValueError(f'LSDub Lip Kit: no shot {shot_id}')
                clip = extract_shot(source.path, s['f0'], s['f1'], st.path('clip.mp4'))
            await asyncio.to_thread(call, ML.main, ['prep', '--video', clip, '--out-dir', D], True, 'lip kit prep')
            loop = asyncio.get_running_loop()
            key = media.hidden_api_key(cls)

            def runner(vis_text, keep_, seed, base, out_dir):        # one Qwen edit, run on ComfyUI's loop
                g = graphs.still('@img', ML.viseme_prompt(vis_text, keep_), seed, megapixels=1.0, prefix='lsdub/kit',
                                 negative='cartoon, painting, blurry face, distorted mouth, extra teeth, flicker, text, watermark, '
                                          'deformed, plastic skin')
                fut = asyncio.run_coroutine_threadsafe(generate.run(g, project.backend, uploads={'img': base}, out_dir=out_dir,
                                                                    api_key=key), loop)
                return fut.result()
            args = argparse.Namespace(out_dir=D, seeds=[11 * (i + 1) + project.seed for i in range(seeds)], only=None,
                                      no_register=False, keep=keep, max_drift=12.0, open_cap=0.0)
            await asyncio.to_thread(ML.gen, args, runner)
            st.mark_done(True, backend=project.backend)
        return io.NodeOutput(T.Kit(dir=D, identity=identity.strip(), base_frame=os.path.join(D, 'base.png'), hash=cls._hash(D)))


class LSDubTweaks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubTweaks', display_name='LSDub Mouth Tweaks', category='LSDub/mouth',
            description='The mouth controls for the dub. Defaults are the proven recipe; change one thing at a time.',
            inputs=[
                io.Float.Input('lead', default=0.06, min=0.0, max=0.3, step=0.01,
                               tooltip='How far the mouth leads the sound, s. Lower = later mouth.'),
                io.Float.Input('word_close', default=0.06, min=0.0, max=0.3, step=0.01,
                               tooltip='Closure between words, s. Higher = crisper word edges.'),
                io.Float.Input('face_target', default=512.0, min=256.0, max=1024.0, step=16.0,
                               tooltip='Face height the shot is upscaled to before warping, px.'),
                io.Boolean.Input('snap', default=True, tooltip='Snappy mouth: 2x frame rate warp and sharper attacks.'),
                io.Boolean.Input('auto', default=True,
                                 tooltip='Also try the full-frame warp and keep whichever measures better in sync.'),
                io.Boolean.Input('continuity', default=True,
                                 tooltip='After the dub, repair single-frame pops against the source.'),
                io.String.Input('settings', default='', multiline=True,
                                tooltip='Advanced: extra mouth-warp flags, passed through as is (e.g. --warp-sigma 0.8).'),
                io.String.Input('jaw_csv', default='', optional=True,
                                tooltip='Optional: a jaw-curve CSV for the whole song (Audio2Face-3D blendshape export, '
                                        'columns frame,time_s,...,jawOpen; song time). Drives the jaw in the silences more '
                                        'cleanly. Empty = the vocal envelope only.'),
            ],
            outputs=[T.LSDubTweaks.Output(display_name='tweaks', tooltip='The mouth controls.')],
        )

    @classmethod
    def execute(cls, lead, word_close, face_target, snap, auto, continuity, settings, jaw_csv=''):
        jaw_csv = (jaw_csv or '').strip()
        if jaw_csv and not os.path.exists(jaw_csv): raise FileNotFoundError(f'LSDub Mouth Tweaks: no jaw CSV {jaw_csv}')
        flags = dict(lead=lead, word_close=word_close, face_target=face_target, snap=snap, auto=auto, continuity=continuity,
                     jaw_csv=jaw_csv, jaw_csv_hash=file_hash(jaw_csv) if jaw_csv else '')
        return io.NodeOutput(T.Tweaks(flags=flags, settings=settings.strip(), hash=text_hash(json.dumps(flags, sort_keys=True) + settings)))


DEFAULT_TWEAKS = T.Tweaks(flags=dict(lead=0.06, word_close=0.06, face_target=512.0, snap=True, auto=True, continuity=True))
ON_FAIL = ('source', 'render')
AUDIO = ('source', 'song')
VOCALISATIONS = ('skip', 'rest', 'mouth')


def _log(msg):
    print(f'[LSDub] {msg}', file=sys.stderr)


def _write_frames(frames, dst, fps, crf=10):
    """BGR frames -> an H.264 file through ffmpeg; raises if ffmpeg fails (the file appears only when complete)."""
    import numpy as np
    from .lsdub_engine.io import ff
    h, w = frames[0].shape[:2]
    tmp = dst + '.part.mp4'
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                  '-c:v', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p', tmp], stdin=ff.PIPE)
    try:
        for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() != 0: raise RuntimeError(f'ffmpeg failed writing {dst}')
    os.replace(tmp, dst)
    return dst


def extract_shot(src, f0, f1, dst):
    """frames f0..f1 of src, frame-exact: decoded in order from frame 0 (no seeking, which is not frame-exact on
    long-GOP sources) -> dst. Raises when the source ends early."""
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    for _ in range(f0):
        if not cap.grab(): break
    F = []
    for _ in range(f1 - f0 + 1):
        ok, f = cap.read()
        if not ok: break
        F.append(f)
    cap.release()
    if len(F) != f1 - f0 + 1:
        raise RuntimeError(f'shot frames {f0}-{f1}: the source has only {f0 + len(F)} frames')
    return _write_frames(F, dst, fps)


def _audio_has_stream(path):
    from .lsdub_engine.io import ff
    r = ff.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', path],
               capture_output=True, text=True)
    return bool(r.stdout.strip())


def assemble(source, renders, out, fps, crf=18, audio='source', song=None, offset=0.0):
    """the cut: the source's frames, with each dubbed shot's frames replacing its range. Audio: 'source' = the source's
    own track (a music video whose picture already carries the song); 'song' = the song file, placed at song_offset
    (video time = song time + offset). Raises if ffmpeg fails; -> warnings (frame-count mismatches, no audio)."""
    import cv2
    import numpy as np
    from .lsdub_engine.core import medium as MED
    from .lsdub_engine.io import ff
    warn = []
    cap = cv2.VideoCapture(source)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reps = {}
    for (f0, f1), path in renders.items():
        R, _ = MED.read(path)
        if len(R) != f1 - f0 + 1: warn.append(f'shot {f0}-{f1}: render has {len(R)} frames, expected {f1 - f0 + 1}')
        for i, f in enumerate(R[:f1 - f0 + 1]): reps[f0 + i] = f
    if audio == 'source' and not _audio_has_stream(source):
        if song: audio = 'song'
        warn.append('the source has no audio' + ('; using the song at song_offset' if song else '; the cut is silent'))
    if audio == 'song' and song:
        a_in = ['-itsoffset', f'{offset:.3f}', '-i', song]
    else:
        a_in = ['-i', source]
    tmp = out + '.part.mp4'
    p = ff.popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                  *a_in, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-crf', str(int(crf)), '-pix_fmt', 'yuv420p',
                  '-c:a', 'aac', '-shortest', '-movflags', '+faststart', tmp], stdin=ff.PIPE)
    i = 0
    try:
        while True:
            ok, f = cap.read()
            if not ok: break
            g = reps.get(i, f)
            if g.shape[:2] != (H, W): g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
            p.stdin.write(np.ascontiguousarray(g).tobytes())
            i += 1
        p.stdin.close()
    except BrokenPipeError:
        pass
    finally:
        cap.release()
    if p.wait() != 0: raise RuntimeError('ffmpeg failed assembling the cut')
    os.replace(tmp, out)
    for w in warn: _log(w)
    return warn


def cut_render(project, source, shots, quality, audio, song, scope='LSDubDub/cut'):
    """assemble (or reuse) the cut of these per-shot results. The key holds every shot's RESULT (render hash, pass,
    which frames were used), so a redone shot always gives a new cut."""
    used = {k: v for k, v in shots.items() if v.get('used') == 'render' and v.get('render')}
    ckey = stage_key(scope, {'shots': {k: [v.get('result'), v.get('used')] for k, v in sorted(shots.items())}, 'crf': quality,
                             'audio': audio}, [project.cache_key(), source.hash, song.cache_key() if (song and audio == 'song') else ''])
    cst = project.stage('shots/_cut', ckey)
    cut = cst.path('cut.mp4')
    if not cst.is_done():
        warn = assemble(source.path, {(v['f0'], v['f1']): v['render'] for v in used.values()}, cut, source.fps, quality, audio,
                        (song.mix if song else None), project.song_offset)
        cst.mark_done(True, shots=sorted(used), warnings=warn)
    failed = sorted(int(k) for k, v in shots.items() if not v.get('pass'))
    return T.Render(shots=shots, cut=cut, hash=ckey), failed


class LSDubDub(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubDub', display_name='LSDub Dub', category='LSDub/mouth',
            description='Dub every sung shot the triage passed (close-ups and mediums), leave the rest untouched, and '
                        'assemble the cut. Resumable: each shot is saved as it finishes; re-runs only redo what changed.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project (mode test_learn skips the dub).'),
                T.LSDubSource.Input('source', tooltip='The video.'),
                T.LSDubShots.Input('shots', tooltip='The triaged shots (from LSDub Triage).'),
                T.LSDubSong.Input('song', tooltip='The song: its lead vocal drives the mouth.'),
                T.LSDubAlign.Input('align', tooltip='The word timings (from LSDub Align).'),
                T.LSDubKit.Input('kit', tooltip='A singer\'s lip kit.'),
                T.LSDubKit.Input('kit_2', optional=True, tooltip='A second singer\'s lip kit.'),
                T.LSDubKit.Input('kit_3', optional=True, tooltip='A third singer\'s lip kit.'),
                T.LSDubTweaks.Input('tweaks', optional=True, tooltip='Mouth controls (defaults: the proven recipe).'),
                T.LSDubMasks.Input('masks', optional=True, tooltip='Occluder masks (from LSDub Occluders): keep a mic in front.'),
                T.LSDubCast.Input('cast', optional=True, tooltip='The cast: finds each singer by identity (else by the triage box).'),
                io.String.Input('only_shots', default='', tooltip='Dub only these shots, e.g. "16,19" (empty = all that passed).'),
                io.Int.Input('quality', default=18, min=10, max=30, optional=True,
                             tooltip='Cut quality (x264 CRF): lower = better and bigger; 18 is visually clean, 14 near-lossless.'),
                io.Combo.Input('on_fail', options=list(ON_FAIL), default='source', optional=True,
                               tooltip='A shot that fails its sync gate: source = keep the original frames in the cut (the '
                                       'shot is listed for Fix-It); render = use the render anyway.'),
                io.Combo.Input('audio', options=list(AUDIO), default='source', optional=True,
                               tooltip='Sound of the cut: source = the video\'s own track (a music video that already '
                                       'carries the song); song = the song file at the project\'s song_offset.'),
                io.Combo.Input('vocalisations', options=list(VOCALISATIONS), default='skip', optional=True,
                               tooltip='Sung sounds that are not words (ooh, da da, scat): skip = keep the video\'s own mouth '
                                       'there; rest = close the mouth; mouth = sing them like words (best for a scat chorus '
                                       'or a generated performance).'),
            ],
            outputs=[T.LSDubRender.Output(display_name='render', tooltip='Per-shot results (pass / fail and why), gates and '
                                                                         'the assembled cut.'),
                     io.Video.Output(display_name='video', tooltip='The assembled cut.')],
        )

    @classmethod
    def _plan(cls, project, shots, kits, only):
        only = set(only)
        return [s for s in shots.shots if s.get('verdict') in DUB_VERDICTS and (not only or s['shot'] in only)
                and cls._kit_for(s, kits)]

    @staticmethod
    def _kit_for(s, kits):
        """the shot's singer's kit; with ONE kit and no cast names (triage without a cast), that kit for every shot."""
        if s.get('target') in kits: return kits[s['target']]
        return next(iter(kits.values())) if len(kits) == 1 else None

    @classmethod
    def _shot_key(cls, project, source, s, song, align, kit, tweaks, masks, cast, vocalisations='skip'):
        occ = (masks.shots or {}).get(str(s['shot'])) if masks else None
        extra = {} if vocalisations == 'skip' else {'vocalisations': vocalisations}      # skip = the keys before the option
        return stage_key('LSDubDub/shot', {'f': [s['f0'], s['f1']], 'start': s['start'], 'target': s.get('target'),
                                           'box': s.get('box'), 'occ': file_hash(occ) if occ and os.path.exists(occ) else '', **extra},
                         [project.cache_key(), source.hash, song.cache_key(), align.hash, kit.hash, tweaks.cache_key(),
                          cast.cache_key() if cast else ''])

    @classmethod
    def _keys(cls, project, source, shots, song, align, kit, kit_2, kit_3, tweaks, masks, cast, only_shots, vocalisations='skip'):
        from .project import parse_force_shots
        kits = {k.identity: k for k in (kit, kit_2, kit_3) if k is not None}
        tw = tweaks or DEFAULT_TWEAKS
        plan = cls._plan(project, shots, kits, parse_force_shots(only_shots))
        return {s['shot']: cls._shot_key(project, source, s, song, align, cls._kit_for(s, kits), tw, masks, cast, vocalisations)
                for s in plan}, plan, kits, tw

    @classmethod
    def fingerprint_inputs(cls, project, source, shots, song, align, kit, kit_2=None, kit_3=None, tweaks=None, masks=None,
                           cast=None, only_shots='', quality=18, on_fail='source', audio='source', vocalisations='skip'):
        if project.mode == 'test_learn': return stage_key('LSDubDub', {'mode': 'test_learn'}, [source.hash])
        keys = cls._keys(project, source, shots, song, align, kit, kit_2, kit_3, tweaks, masks, cast, only_shots, vocalisations)[0]
        return stage_key('LSDubDub', {'shots': keys, 'force': list(project.force_shots), 'crf': quality, 'fail': on_fail, 'audio': audio})

    @classmethod
    def _mask_for(cls, masks, s, tw, n):
        """the shot's occluder mask, if it was made on a crop that matches this dub's (face size, frame rate, length)."""
        occ = (masks.shots or {}).get(str(s['shot'])) if masks else None
        if not occ or not os.path.exists(occ): return None, None
        meta_p = occ[:-4] + '.json'
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        want = {'face_target': float(tw.flags.get('face_target', 512.0)), 'fps_mult': 2 if tw.flags.get('snap', True) else 1,
                'frames': n * (2 if tw.flags.get('snap', True) else 1)}
        bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        if bad:
            msg = f"shot {s['shot']}: occluder mask was made on a different crop {bad}; not used (re-run LSDub Occluders)"
            _log(msg)
            return None, msg
        return occ, None

    @classmethod
    def run_shot(cls, project, source, s, song, align, kit, tw, masks, cast, st, vocalisations='skip'):
        from .lsdub_engine.core import sing as SING, continuity as CONT
        work = st.path('work')
        os.makedirs(work, exist_ok=True)
        take = extract_shot(source.path, s['f0'], s['f1'], st.path('take.mp4'))
        occ, note = cls._mask_for(masks, s, tw, s['f1'] - s['f0'] + 1)
        if occ: shutil.copyfile(occ, os.path.join(work, 'occ.npz'))
        f = tw.flags
        pa = ['--lib', kit.dir, *_kit_flags(project, kit.identity), *SING_EXTRA, *tw.settings.split(),
              '--vocalisations', vocalisations]
        info = SING.run(take, align.align, song.lead or song.vocal or song.mix, s['start'] - project.song_offset,
                        st.path('sung.mp4'), work, pa, char=s['target'] if cast else None, db=cast.db if cast else None,
                        box=None if cast else s.get('box'), face_target=float(f.get('face_target', 512.0)),
                        snap=bool(f.get('snap', True)), report=st.path('gate.json'), lead=f.get('lead'),
                        word_close=f.get('word_close'), a2f=f.get('jaw_csv') or None, auto=bool(f.get('auto', True)),
                        preset=project.preset)
        render = st.path('render.mp4')
        if f.get('continuity', True):
            CONT.run(take, st.path('sung.mp4'), render, report=st.path('continuity.json'))
        else:
            shutil.copyfile(st.path('sung.mp4'), render)
        if note: info['mask_note'] = note
        return render, info

    @classmethod
    def execute(cls, project, source, shots, song, align, kit, kit_2=None, kit_3=None, tweaks=None, masks=None, cast=None,
                only_shots='', quality=18, on_fail='source', audio='source', vocalisations='skip'):
        from . import devices as _dev
        _dev.ensure_face_models()
        from .project import take_force
        if project.mode == 'test_learn':                              # analyse only: the cut is the source
            return io.NodeOutput(T.Render(shots={}, cut=source.path, hash=source.hash), cls._video(source.path))
        keys, plan, kits, tw = cls._keys(project, source, shots, song, align, kit, kit_2, kit_3, tweaks, masks, cast, only_shots,
                                         vocalisations)
        forced = take_force(project, 'dub')
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(max(1, len(plan)))
        except Exception:
            pbar = None
        res = {}
        for s in plan:
            sid, k = s['shot'], keys[s['shot']]
            st = project.shot(sid, k)
            if shots_todo(project, k, [sid], forced):
                try:
                    if os.path.exists(st.path('gate.json')): os.remove(st.path('gate.json'))   # never a stale gate
                    render, info = cls.run_shot(project, source, s, song, align, cls._kit_for(s, kits), tw, masks, cast, st,
                                                vocalisations)
                    reason = '' if info.get('pass') else (f"sync gate failed: measured corr {info.get('corr')} "
                                                           f"(open in silence {info.get('open_in_silence')})")
                    st.mark_done(bool(info.get('pass')), corr=info.get('corr'), auto=(info.get('auto') or {}).get('winner'),
                                 render=render, reason=reason or info.get('mask_note', ''))
                except Exception as e:                                  # a failed shot does not stop the cut
                    _log(f'shot {sid} failed: {type(e).__name__}: {e}')
                    st.mark_done(False, error=f'{type(e).__name__}: {e}', reason=f'dub error: {type(e).__name__}: {e}')
            d = st.done() or {}
            r = d.get('render') if d.get('render') and os.path.exists(d.get('render')) else ''
            ok = bool(d.get('pass'))
            res[str(sid)] = {'f0': s['f0'], 'f1': s['f1'], 'render': r, 'pass': ok, 'corr': d.get('corr'),
                             'reason': d.get('reason') or d.get('error') or '', 'gate': '' if d.get('error') or not os.path.exists(st.path('gate.json'))
                             else st.path('gate.json'), 'key': k,
                             'result': file_hash(r) if r else 'none',
                             'used': 'render' if r and (ok or on_fail == 'render') else 'source'}
            if pbar: pbar.update(1)
        rec, failed = cut_render(project, source, res, quality, audio, song)
        if failed: _log(f'{len(failed)} shot(s) failed: {failed} (see LSDub Fix It)')
        return io.NodeOutput(rec, cls._video(rec.cut))

    @staticmethod
    def _video(path):
        try:
            from comfy_api.latest import InputImpl
            return InputImpl.VideoFromFile(path)
        except Exception:
            return path


class LSDubOccluders(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubOccluders', display_name='LSDub Occluders', category='LSDub/mouth',
            description='Mask what is in front of the mouth (a microphone, a hand) on the dubbed shots\' face crops, so '
                        'the next dub keeps it in front. Uses the GPU (Florence-2 + SAM 2.1, weights in models/lsdub).',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                T.LSDubRender.Input('render', tooltip='A dub (its shots\' face crops are masked).'),
                io.String.Input('phrases', default='microphone',
                                tooltip='What to mask, comma-separated (e.g. "microphone, hand").'),
                io.String.Input('shot_ids', default='', tooltip='Only these shots, e.g. "19,51" (empty = every dubbed shot).'),
            ],
            outputs=[T.LSDubMasks.Output(display_name='masks', tooltip='Per-shot occluder masks; feed back into LSDub Dub.'),
                     io.Video.Output(display_name='review', tooltip='The first shot\'s mask overlay.')],
        )

    @classmethod
    def _crops(cls, render, shot_ids):
        from .project import parse_force_shots
        want = set(parse_force_shots(shot_ids))
        for sid, v in sorted(render.shots.items(), key=lambda kv: int(kv[0])):
            if want and int(sid) not in want: continue
            d = os.path.dirname(v.get('gate') or v.get('render') or '')
            crop = os.path.join(d, 'work', 'crop.mp4')
            if os.path.exists(crop): yield sid, d, crop

    @classmethod
    def fingerprint_inputs(cls, project, render, phrases, shot_ids=''):
        return stage_key('LSDubOccluders', {'p': phrases, 'crops': [file_hash(c) for _, _, c in cls._crops(render, shot_ids)]})

    @classmethod
    def execute(cls, project, render, phrases, shot_ids=''):
        from .lsdub_engine.core import occluders as OCC
        from . import devices
        out, review = {}, None
        for sid, d, crop in cls._crops(render, shot_ids):
            st = project.stage('mouth/masks', stage_key('LSDubOccluders', {'p': phrases, 'crop': file_hash(crop),
                                                                           'models': [OCC.FLORENCE, OCC.SAM2_ID]}))
            m, rv = st.path('occ.npz'), st.path('review.mp4')
            if not st.is_done():
                md = devices.models_dir('occluders')
                argv = ['--video', crop, '--out', m, '--review', rv, '--device', str(devices.torch_device()), '--model-dir', md]
                ck = os.path.join(md, OCC.SAM2_FILE)
                if os.path.exists(ck): argv += ['--sam2-ckpt', ck]
                for ph in phrases.split(','):
                    if ph.strip(): argv += ['--phrase', ph.strip()]
                try:
                    call(OCC.main, argv, name='occluders')
                finally:
                    devices.free()
                g = json.load(open(os.path.join(d, 'gate.json'))) if os.path.exists(os.path.join(d, 'gate.json')) else {}
                media_meta = {'face_target': float(g.get('face_target', 512.0)), 'fps_mult': int(g.get('fps_mult', 2)),
                              'frames': int(g.get('frames', 0)) * int(g.get('fps_mult', 2)), 'crop': file_hash(crop)}
                with open(m[:-4] + '.json', 'w') as f: json.dump(media_meta, f)
                st.mark_done(True)
            out[sid] = m
            review = review or rv
        return io.NodeOutput(T.Masks(dir=os.path.join(project.dir, 'mouth', 'masks'), shots=out,
                                     hash=text_hash(json.dumps(sorted((k, file_hash(p)) for k, p in out.items())))),
                             LSDubDub._video(review or media.placeholder_video(os.path.join(project.dir, 'mouth', 'masks', 'none.mp4'),
                                                                                'no dubbed shot crops to mask')))


class LSDubContinuity(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubContinuity', display_name='LSDub Continuity', category='LSDub/mouth',
            description='Re-run the continuity repair on chosen shots with stricter settings (single-frame pops, ghost '
                        'lips), then re-assemble the cut.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                T.LSDubSource.Input('source', tooltip='The video.'),
                T.LSDubRender.Input('render', tooltip='The dub to repair.'),
                io.String.Input('shot_ids', default='', tooltip='Shots to repair, e.g. "21,23" (empty = all dubbed shots).'),
                io.Float.Input('z', default=3.5, min=1.5, max=8.0, step=0.1,
                               tooltip='How unusual a frame must be to be repaired; lower repairs more.'),
                io.Int.Input('quality', default=18, min=10, max=30, optional=True, tooltip='Cut quality (x264 CRF).'),
            ],
            outputs=[T.LSDubRender.Output(display_name='render', tooltip='The dub with the repaired shots.'),
                     io.Video.Output(display_name='video', tooltip='The re-assembled cut.')],
        )

    @classmethod
    def fingerprint_inputs(cls, project, source, render, shot_ids='', z=3.5, quality=18):
        return stage_key('LSDubContinuity', {'ids': shot_ids, 'z': z, 'q': quality}, [render.hash])

    @classmethod
    def execute(cls, project, source, render, shot_ids='', z=3.5, quality=18):
        from .lsdub_engine.core import continuity as CONT
        from .project import parse_force_shots
        want = set(parse_force_shots(shot_ids))
        shots = {}
        for sid, v in render.shots.items():
            v = dict(v)
            if v.get('render') and (not want or int(sid) in want):
                d = os.path.dirname(v.get('gate') or v['render'])
                st = project.stage('shots/_continuity', stage_key('LSDubContinuity', {'z': z}, [v['key'], v['result']]))
                out = st.path('render.mp4')
                if not st.is_done():
                    CONT.run(os.path.join(d, 'take.mp4'), os.path.join(d, 'sung.mp4'), out, z=z, report=st.path('continuity.json'))
                    st.mark_done(True)
                v.update(render=out, result=file_hash(out))
            shots[sid] = v
        rec, _ = cut_render(project, source, shots, quality, 'source', None)
        return io.NodeOutput(rec, LSDubDub._video(rec.cut))


NODES = [LSDubBuildLipKit, LSDubTweaks, LSDubOccluders, LSDubDub, LSDubContinuity]
