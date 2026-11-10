# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""MOUTH lane: the lip kit, the mouth controls, occluder masks, the dub itself (per shot, resumable) and continuity.

LSDub Dub, per shot of the triage: DUB / DUB-MEDIUM -> `sing --auto` (crop to a 512 px face, warp at 2x fps, paste
the lower face back, measured gate; the native full-frame warp is also run and the higher measured sync wins), then
the continuity pass; WIDE / NO -> the source frames, untouched. Every shot is saved under shots/shot_###/<key>/ with a
done.json, so an interrupted dub resumes and a re-run only redoes changed, failed or forced shots. The shots are then
assembled into the cut with the source's audio; LSDub Save Cut writes the watermarked final file. With mouth_card on,
each dubbed shot also gets the engine's mouth card beside its render (the render is only read) and the card cut is
assembled from those, a separate, watermarked QA file."""
import json
import os
import shutil
import sys
import time

from comfy_api.v0_0_2 import io

from . import lsdub_types as T
from . import media
from .media import kit_folder
from .lsdub_engine.core import paint_api, presets, recipes
from .lsdub_engine.core.steps import call
from .project import file_hash, shots_todo, stage_key, text_hash

DUB_VERDICTS = ('DUB', 'DUB-MEDIUM')
WARP_SHAPES = ('asym', 'off')                               # the first is the engine's default; the last is 0.1.9's
TRAVEL_GUARDS = ('aligner', 'take')
# the vocal-driven mouth stages (engine core/nucleus_events, nucleus_hold, closure_plan, aperture, the realise stage): since 0.3.0 the first
# value is the engine's default (sing.SNAP turns them on); `classic` is the 0.2.x mouth, one choice away on each. STAGE_FLAGS maps a Combo
# to its paint flag. LEGACY: the values a 0.2.x workflow saved for the old default, read as `classic`.
WARP_EVENTS = ('vocal', 'classic')
WARP_HOLDS = ('nucleus-follow', 'nucleus', 'classic')
WARP_CLOSURES = ('aligned-press', 'aligned', 'classic')
WARP_APERTURES = ('p95', 'map', 'classic')
WARP_REALISES = ('events', 'classic')
STAGE_FLAGS = {'warp_events': ('--warp-events', WARP_EVENTS), 'warp_hold': ('--warp-hold', WARP_HOLDS),
               'warp_closures': ('--warp-closures', WARP_CLOSURES), 'warp_aperture': ('--warp-aperture', WARP_APERTURES),
               'warp_realise': ('--warp-realise', WARP_REALISES)}
LEGACY = {'off': 'classic', 'plan': 'classic'}
COHERENCE_CHARGES = ('remade', 'all')
SWITCH_TIPS = dict(
    warp_shape='Mouth shape on the snap warp: asym closes the lips 20 ms early and opens them on time; off is the 0.1.9 shape.',
    travel_guard='A re-timed mouth that barely moves is swapped for the un-retimed render: aligner only when it also moves less than '
                 'that render; take judges it against the take alone (0.1.9).',
    coherence_charge='What the continuity repair\'s budget counts: remade only the frames it re-makes; all every flagged run (0.1.9), '
                     'so less gets repaired.',
    warp_events='Reads the sung syllables off the lead vocal and writes them on the mouth plan (needed by the four below). classic = the '
                '0.2.x plan.',
    warp_hold='One mouth opening per sung syllable, held to the next one, with a dip before each syllable so every onset is a visible rise: '
              'nucleus-follow also lets the opening follow the voice\'s loudness inside the syllable; nucleus holds it flat; classic = the '
              '0.2.x envelope mouth. Needs warp_events vocal: without it the engine warns and it does nothing.',
    warp_closures='Where the lips shut: aligned-press only on the lyric\'s p/b/m, timed off the vocal, pressed to contact; aligned the same '
                  'without the press (a slit on an open take); classic = the 0.2.x closures. Needs warp_events vocal: '
                  'without it the engine warns and it does nothing.',
    warp_aperture='How far the vowels open: p95 scales the plan so the loud wide vowels reach the singer\'s own widest opening; map the same '
                  'over every frame; classic = as planned (0.2.x). p95 needs warp_events vocal: without it p95 is the same as map.',
    warp_realise='events lets each sung syllable\'s attack through the smoothing, so short syllables still show; classic = the 0.2.x '
                 'smoothing everywhere. Needs warp_events vocal.',
)
SING_EXTRA = ['--warp-track-mode', 'chain']                 # the standard sing recipe's track mode


def _paint_warnings(argv):
    """what these flags will not do on this run; observation only, argv goes to the paint unchanged."""
    from .lsdub_engine.core import mouth_paint
    try:
        out = mouth_paint.paint_warnings(['--mode', 'warp', *argv])                   # sing paints with warp
    except SystemExit:                                                   # a flag paint rejects: the render reports it
        return []
    for w in out: _log(f'paint: {w}')
    return out


def _stack(project, identity):
    """(recipe, per-face argv) of this face. mouth_stack picks the recipe; on `plain`, a face the preset lists keeps the preset's
    recipe."""
    p = presets.get(project.preset)
    listed = identity in p.face_overrides
    recipe = p.recipe if project.mouth_stack == 'plain' and listed else recipes.MOUTH_STACKS[project.mouth_stack]
    return recipe, p.face_overrides.get(identity, ())


def _advance(tw):
    """the Tweaks' sync_advance_ms as paint flags; settings come after it, so a flag typed there wins."""
    ms = (getattr(tw, 'flags', None) or {}).get('sync_advance_ms')
    return ['--coart-advance-ms', str(ms)] if ms else []


def _phrases(tw, vocal):
    """the Tweaks' sync_phrases as paint flags: the planner reads the phrase offsets off this vocal."""
    return ['--coart-sync-vocal', vocal] if vocal and (getattr(tw, 'flags', None) or {}).get('sync_phrases') else []


def _shape(tw):
    """the Tweaks' warp_shape as paint flags: nothing at asym (sing's own flag), --warp-shape off at off; settings come after it."""
    return ['--warp-shape', 'off'] if (getattr(tw, 'flags', None) or {}).get('warp_shape', WARP_SHAPES[0]) == 'off' else []


def _stages(tw):
    """the Tweaks' mouth-stage Combos as paint flags: nothing at a default (sing's own), the flag at any other value (`classic` too: the
    engine reads it as the 0.2.x value, and sing then drops its own); sing adds --warp-audio for --warp-events itself. settings come
    after them."""
    f = getattr(tw, 'flags', None) or {}
    out = []
    for name, (flag, options) in STAGE_FLAGS.items():
        v = f.get(name, options[0])
        if v != options[0]: out += [flag, v]
    if f.get('warp_events', WARP_EVENTS[0]) != WARP_EVENTS[0]:  # the phone network's folder under models/lsdub, as Align's
        from . import devices
        out += ['--warp-phone-models', devices.models_dir('wav2vec2-espeak')]
    return out


def _paint_flags(project, kit, tw, vocalisations, vocal=None):
    """the shot's flags for sing. --vocalisations is a user-level flag: settings can change it."""
    recipe, face = _stack(project, kit.identity)
    return paint_api.flags(recipe, lib=kit.dir, mode_mods=SING_EXTRA, face_overrides=face,
                           user_overrides=['--vocalisations', vocalisations, *_advance(tw), *_phrases(tw, vocal), *_shape(tw), *_stages(tw),
                                           *paint_api.split_settings(tw.settings)])


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
                h.append(f"{os.path.relpath(p, kit_dir).replace(os.sep, '/')}:{os.path.getsize(p)}:{file_hash(p)}")
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
                                tooltip='Advanced: extra mouth-warp flags on top of the mouth recipe (e.g. --warp-erase closures). Put a value '
                                        'holding a space or an apostrophe in double quotes; backslashes are kept as typed. With snap on, its three flags (--warp-sigma, --warp-temporal, --warp-close-depth) stay fixed.'),
                io.String.Input('jaw_csv', default='', optional=True,
                                tooltip='Optional: a jaw-curve CSV for the whole song (one row per frame, '
                                        'columns frame,time_s,...,jawOpen; song time). Drives the jaw in the silences more '
                                        'cleanly. Empty = the vocal envelope only.'),
                io.Float.Input('sync_advance_ms', default=0.0, min=0.0, max=200.0, step=10.0, optional=True,
                               tooltip='Moves the whole mouth plan earlier against the sound, ms, snapped to whole frames. It keeps the mouth\'s shapes and only '
                                       'changes when they land. 0 = off.'),
                io.Boolean.Input('sync_phrases', default=False, optional=True,
                                 tooltip='Lets each sung phrase land a little earlier or later, read from the vocal alone. It keeps the mouth\'s shapes '
                                         'and only changes when they land. Needs a layers mouth_stack.'),
                io.Combo.Input('warp_shape', options=['asym', 'off'], default='asym', optional=True, tooltip=SWITCH_TIPS['warp_shape']),
                io.Combo.Input('travel_guard', options=['aligner', 'take'], default='aligner', optional=True,
                               tooltip=SWITCH_TIPS['travel_guard']),
                io.Combo.Input('coherence_charge', options=['remade', 'all'], default='remade', optional=True,
                               tooltip=SWITCH_TIPS['coherence_charge']),
                io.Combo.Input('warp_events', options=['vocal', 'classic'], default='vocal', optional=True, tooltip=SWITCH_TIPS['warp_events']),
                io.Combo.Input('warp_hold', options=['nucleus-follow', 'nucleus', 'classic'], default='nucleus-follow', optional=True,
                               tooltip=SWITCH_TIPS['warp_hold']),
                io.Combo.Input('warp_closures', options=['aligned-press', 'aligned', 'classic'], default='aligned-press', optional=True,
                               tooltip=SWITCH_TIPS['warp_closures']),
                io.Combo.Input('warp_aperture', options=['p95', 'map', 'classic'], default='p95', optional=True,
                               tooltip=SWITCH_TIPS['warp_aperture']),
                io.Combo.Input('warp_realise', options=['events', 'classic'], default='events', optional=True,
                               tooltip=SWITCH_TIPS['warp_realise']),
            ],
            outputs=[T.LSDubTweaks.Output(display_name='tweaks', tooltip='The mouth controls.')],
        )

    @classmethod
    def validate_inputs(cls, warp_events='vocal', warp_hold='nucleus-follow', warp_closures='aligned-press', warp_aperture='p95',
                        warp_realise='events'):
        """the mouth-stage Combos take their options and the values a 0.2.x workflow saved (off / plan, read as classic)."""
        for name, v in (('warp_events', warp_events), ('warp_hold', warp_hold), ('warp_closures', warp_closures),
                        ('warp_aperture', warp_aperture), ('warp_realise', warp_realise)):
            options = STAGE_FLAGS[name][1]
            if v not in options and LEGACY.get(v) not in options:
                return f'{name} must be one of {", ".join(options)}, not {v!r}'
        return True

    @classmethod
    def execute(cls, lead, word_close, face_target, snap, auto, continuity, settings, jaw_csv='', sync_advance_ms=0.0, sync_phrases=False,
                warp_shape='asym', travel_guard='aligner', coherence_charge='remade', warp_events='vocal', warp_hold='nucleus-follow',
                warp_closures='aligned-press', warp_aperture='p95', warp_realise='events'):
        try:
            paint_api.split_settings(settings)
        except ValueError as e:
            raise ValueError(f'LSDub Mouth Tweaks: settings: {e}') from None
        jaw_csv = (jaw_csv or '').strip()
        if jaw_csv and not os.path.exists(jaw_csv): raise FileNotFoundError(f'LSDub Mouth Tweaks: no jaw CSV {jaw_csv}')
        flags = dict(lead=lead, word_close=word_close, face_target=face_target, snap=snap, auto=auto, continuity=continuity,
                     jaw_csv=jaw_csv, jaw_csv_hash=file_hash(jaw_csv) if jaw_csv else '')
        if sync_advance_ms:  # absent at 0: a tweaks hash, and every shot behind it, stays what it was
            flags['sync_advance_ms'] = float(sync_advance_ms)
        if sync_phrases:  # the same: absent when off
            flags['sync_phrases'] = True
        for name, value, options in (('warp_shape', warp_shape, WARP_SHAPES), ('travel_guard', travel_guard, TRAVEL_GUARDS),
                                     ('coherence_charge', coherence_charge, COHERENCE_CHARGES), ('warp_events', warp_events, WARP_EVENTS),
                                     ('warp_hold', warp_hold, WARP_HOLDS), ('warp_closures', warp_closures, WARP_CLOSURES),
                                     ('warp_aperture', warp_aperture, WARP_APERTURES), ('warp_realise', warp_realise, WARP_REALISES)):
            if name in STAGE_FLAGS and value not in options and LEGACY.get(value) in options:
                _log(f'Mouth Tweaks: {name} {value!r} is a 0.2.x value, read as classic (the 0.2.x mouth); pick {options[0]} for the 0.3.0 mouth')
                value = 'classic'
            if value not in options: raise ValueError(f'LSDub Mouth Tweaks: {name} must be one of {", ".join(options)}, not {value!r}')
            if value != options[0]:  # absent at the default, as above; a cached shot still re-dubs when the engine changes (its key
                flags[name] = value  # carries the whole engine's version)
        return io.NodeOutput(T.Tweaks(flags=flags, settings=settings.strip(), hash=text_hash(json.dumps(flags, sort_keys=True) + settings)))


DEFAULT_TWEAKS = T.Tweaks(flags=dict(lead=0.06, word_close=0.06, face_target=512.0, snap=True, auto=True, continuity=True))
ON_FAIL = ('source', 'render')
AUDIO = ('source', 'song')
VOCALISATIONS = ('skip', 'rest', 'mouth')


def _vocal_note(info):
    """a failed gate on a shot this much vocalisation says so: the sync levers are not the answer there."""
    from .lsdub_engine.audits.qa import VOCAL_NOTE
    f = info.get('vocal_frac') or 0
    return f'vocalisation spans cover {f:.0%} of the shot; ' if f >= VOCAL_NOTE else ''


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
    return T.Render(shots=shots, cut=cut, hash=ckey, audio=audio, song=song.to_json() if song and audio == 'song' else {}), failed


def card_render(project, source, shots, quality, audio, song):
    """the card cut: the clean cut with each dubbed shot's mouth-card video in its place (a card in a band below the
    picture is scaled to the frame). Keyed on every shot's result and card; the clean cut is a different file."""
    cards = {k: v['card'] for k, v in shots.items() if v.get('card') and os.path.exists(v['card'])}
    ckey = stage_key('LSDubDub/card', {'shots': {k: [v.get('result'), v.get('used'), file_hash(cards[k]) if k in cards else '']
                                                 for k, v in sorted(shots.items())}, 'crf': quality, 'audio': audio},
                     [project.cache_key(), source.hash, song.cache_key() if (song and audio == 'song') else ''])
    cst = project.stage('shots/_card', ckey)
    cut = cst.path('card_cut.mp4')
    if not cst.is_done():
        from .lsdub_engine import watermark as WM
        warn = assemble(source.path, {(shots[k]['f0'], shots[k]['f1']): p for k, p in cards.items()}, cut, source.fps, quality, audio,
                        (song.mix if song else None), project.song_offset)
        WM.apply_inplace(cut, crf=int(quality))                       # marked like the engine's paint --annotate card video
        cst.mark_done(True, shots=sorted(cards), warnings=warn)
    return cut


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
                                       'or a generated performance). The Dub finds these spans again from the word timings; '
                                       'the skip marks LSDub Align writes are not read here.'),
                io.Boolean.Input('mouth_card', default=False, optional=True,
                                 tooltip='Mouth card: a QA view of each dubbed shot with a per-frame card in a corner (never over '
                                         'a face): the sung word, its mouth shape, openness and the next second of mouth '
                                         'shapes; uncertain words marked with a ? and the reason. The clean cut and its '
                                         'renders never change; the card cut comes out of card_video.'),
            ],
            outputs=[T.LSDubRender.Output(display_name='render', tooltip='Per-shot results (pass / fail and why), gates and '
                                                                         'the assembled cut.'),
                     io.Video.Output(display_name='video', tooltip='The assembled cut.'),
                     io.Video.Output(display_name='card_video', tooltip='The cut with the mouth card on each dubbed shot (QA '
                                                                        'view, not saved by Save Cut). With mouth_card off: '
                                                                        'the assembled cut.')],
        )

    @classmethod
    def _plan(cls, project, shots, kits, only):
        return [s for s in cls._wanted(shots, only) if cls._kit_for(s, kits)]

    @staticmethod
    def _wanted(shots, only):
        only = set(only)
        return [s for s in shots.shots if s.get('verdict') in DUB_VERDICTS and (not only or s['shot'] in only)]

    @classmethod
    def _dropped(cls, shots, kits, only, cast=None):
        """the shots to dub that no kit serves, each with its warning."""
        hint = "give a kit this singer's cast name" if cast else 'connect LSDub Cast or use one kit'
        return {s['shot']: (s, f"shot {s['shot']}: target {s.get('target')!r} matches no kit (kits: {', '.join(kits)}); {hint}")
                for s in cls._wanted(shots, only) if not cls._kit_for(s, kits)}

    @staticmethod
    def _kit_for(s, kits):
        """the shot's singer's kit; with ONE kit and no cast names (triage without a cast), that kit for every shot."""
        if s.get('target') in kits: return kits[s['target']]
        return next(iter(kits.values())) if len(kits) == 1 else None

    @classmethod
    def _shot_key(cls, project, source, s, song, align, kit, tweaks, masks, cast, vocalisations='skip'):
        occ = (masks.shots or {}).get(str(s['shot'])) if masks else None
        extra = {} if vocalisations == 'skip' else {'vocalisations': vocalisations}      # skip = the keys before the option
        if project.mouth_stack != 'plain':                                               # plain = the keys before the option
            extra.update(mouth_stack=project.mouth_stack, identity=kit.identity,
                         recipe=recipes.MOUTH_STACKS[project.mouth_stack])
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
                           cast=None, only_shots='', quality=18, on_fail='source', audio='source', vocalisations='skip',
                           mouth_card=False):
        if project.mode == 'test_learn': return stage_key('LSDubDub', {'mode': 'test_learn'}, [source.hash])
        keys = cls._keys(project, source, shots, song, align, kit, kit_2, kit_3, tweaks, masks, cast, only_shots, vocalisations)[0]
        card = {'card': True} if mouth_card else {}                          # absent when off: the fingerprint stays what it was
        return stage_key('LSDubDub', {'shots': keys, 'force': list(project.force_shots), 'crf': quality, 'fail': on_fail, 'audio': audio,
                                      **card})

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
        from .lsdub_engine.core import sing as SING
        work = st.path('work')
        os.makedirs(work, exist_ok=True)
        take = extract_shot(source.path, s['f0'], s['f1'], st.path('take.mp4'))
        occ, note = cls._mask_for(masks, s, tw, s['f1'] - s['f0'] + 1)
        if occ: shutil.copyfile(occ, os.path.join(work, 'occ.npz'))
        f = tw.flags
        pa = _paint_flags(project, kit, tw, vocalisations, song.lead or song.mix)
        warned = _paint_warnings(pa)
        info = SING.run(take, align.align, song.lead or song.mix, s['start'] - project.song_offset,
                        st.path('sung.mp4'), work, pa, char=s['target'] if cast else None, db=cast.db if cast else None,
                        box=None if cast else s.get('box'), face_target=float(f.get('face_target', 512.0)),
                        snap=bool(f.get('snap', True)), report=st.path('gate.json'), lead=f.get('lead'),
                        word_close=f.get('word_close'), a2f=f.get('jaw_csv') or None, auto=bool(f.get('auto', True)),
                        preset=project.preset, travel_ref=f.get('travel_guard', TRAVEL_GUARDS[0]))
        render = st.path('render.mp4')
        if f.get('continuity', True):
            info.update(SING.repair(take, st.path('sung.mp4'), render, info, work, report=st.path('continuity.json'),
                                    charge=f.get('coherence_charge', COHERENCE_CHARGES[0])))
            json.dump(info, open(st.path('gate.json'), 'w'), indent=1)   # the gate's record carries what the cut delivers
        else:
            shutil.copyfile(st.path('sung.mp4'), render)
        if note: info['mask_note'] = note
        if warned: info['paint_warnings'] = warned
        return render, info

    @classmethod
    def execute(cls, project, source, shots, song, align, kit, kit_2=None, kit_3=None, tweaks=None, masks=None, cast=None,
                only_shots='', quality=18, on_fail='source', audio='source', vocalisations='skip', mouth_card=False):
        from . import devices as _dev
        _dev.ensure_face_models()
        from .project import take_force
        if project.mode == 'test_learn':                              # analyse only: the cut is the source
            return io.NodeOutput(T.Render(shots={}, cut=source.path, hash=source.hash), cls._video(source.path),
                                 cls._video(source.path))
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
                t0, c0 = time.perf_counter(), time.process_time()
                try:
                    if os.path.exists(st.path('gate.json')): os.remove(st.path('gate.json'))   # never a stale gate
                    render, info = cls.run_shot(project, source, s, song, align, cls._kit_for(s, kits), tw, masks, cast, st,
                                                vocalisations)
                    reason = '' if info.get('pass') else (f"{_vocal_note(info)}sync gate failed: audio sync {info.get('truth')} "
                                                           f"(mouth {info.get('truth_lag_ms')} ms vs the vocal), plan corr {info.get('corr')} "
                                                           f"(open in silence {info.get('open_in_silence')})")
                    st.mark_done(bool(info.get('pass')), time.perf_counter() - t0, time.process_time() - c0,
                                 corr=info.get('corr'), auto=(info.get('auto') or {}).get('winner'),
                                 render=render, reason=reason or info.get('mask_note', ''),
                                 paint_warnings=info.get('paint_warnings', []))
                except Exception as e:                                  # a failed shot does not stop the cut
                    _log(f'shot {sid} failed: {type(e).__name__}: {e}')
                    st.mark_done(False, time.perf_counter() - t0, time.process_time() - c0, error=f'{type(e).__name__}: {e}',
                                 reason=f'dub error: {type(e).__name__}: {e}')
            d = st.done() or {}
            r = d.get('render') if d.get('render') and os.path.exists(d.get('render')) else ''
            ok = bool(d.get('pass'))
            res[str(sid)] = {'f0': s['f0'], 'f1': s['f1'], 'render': r, 'pass': ok, 'corr': d.get('corr'),
                             'reason': d.get('reason') or d.get('error') or '', 'gate': '' if d.get('error') or not os.path.exists(st.path('gate.json'))
                             else st.path('gate.json'), 'key': k,
                             'result': file_hash(r) if r else 'none',
                             'used': 'render' if r and (ok or on_fail == 'render') else 'source'}
            if mouth_card and res[str(sid)]['used'] == 'render':
                card = cls.shot_card(project, source, s, st, song, align, cls._kit_for(s, kits), tw, vocalisations, r)
                if card: res[str(sid)]['card'] = card
            if pbar: pbar.update(1)
        from .project import parse_force_shots
        for sid, (s, msg) in cls._dropped(shots, kits, parse_force_shots(only_shots), cast).items():
            _log(msg)
            res[str(sid)] = {'f0': s['f0'], 'f1': s['f1'], 'render': '', 'pass': False, 'corr': None, 'reason': 'no kit for target',
                             'gate': '', 'key': '', 'result': 'none', 'used': 'source'}
        rec, failed = cut_render(project, source, res, quality, audio, song)
        rec.on_fail = on_fail
        if failed: _log(f'{len(failed)} shot(s) failed: {failed} (see LSDub Fix It)')
        card_cut = rec.cut
        if mouth_card:
            try:
                card_cut = card_render(project, source, res, quality, audio, song)
            except Exception as e:                                      # a QA view: never costs the dub
                _log(f'mouth card cut failed ({type(e).__name__}: {e}); card_video is the clean cut')
        return io.NodeOutput(rec, cls._video(rec.cut), cls._video(card_cut))

    @classmethod
    def shot_card(cls, project, source, s, st, song, align, kit, tw, vocalisations, render):
        """the mouth card beside a dubbed shot's render (the engine's paint --annotate path, which only READS the render):
        <render>_annotated.mp4 + its lyric check. Reused while newer than the render. A side product: a failure is logged,
        the shot and the clean cut are unaffected. -> the annotated path, or ''."""
        from types import SimpleNamespace
        from .lsdub_engine.core import mouth_card as MC
        ann = MC.annotated_path(render)
        if os.path.exists(ann) and os.path.getmtime(ann) >= os.path.getmtime(render): return ann
        gate = json.load(open(st.path('gate.json'))) if os.path.exists(st.path('gate.json')) else {}
        al = gate.get('align') if gate.get('align') and os.path.exists(gate['align']) else align.align   # what the shot was painted from
        f, n = tw.flags, s['f1'] - s['f0'] + 1
        trk = st.path('card_track.json')
        try:
            call('visemes', ['--align', al, '--t0', s['start'] - project.song_offset, '--frames', n, '--fps', source.fps,
                             '--audio', song.lead or song.mix, '--lead', f.get('lead', 0.06),
                             *(('--word-close', f['word_close']) if f.get('word_close') else ()), '--out', trk], quiet=True)
        except Exception as e:
            _log(f"shot {s['shot']}: mouth card skipped ({type(e).__name__}: {e}); the render is unaffected")
            return ''
        a = SimpleNamespace(out=render, track=trk, mode='warp', vocalisations=vocalisations, annotate_corner='auto',
                            vocal_align=al, lib=kit.dir if kit else None)
        return MC.after_paint(a).get('annotated', '')

    @staticmethod
    def _video(path):
        try:
            from comfy_api.latest import InputImpl
        except ImportError:
            raise RuntimeError('LSDub needs ComfyUI 0.3.48 or newer: comfy_api.latest is missing') from None
        return InputImpl.VideoFromFile(path)


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
                io.Combo.Input('coherence_charge', options=['remade', 'all'], default='remade', optional=True,
                               tooltip=SWITCH_TIPS['coherence_charge']),
            ],
            outputs=[T.LSDubRender.Output(display_name='render', tooltip='The dub with the repaired shots.'),
                     io.Video.Output(display_name='video', tooltip='The re-assembled cut.')],
        )

    @staticmethod
    def _charge(coherence_charge):
        """absent at the default: the keys before the option."""
        if coherence_charge not in COHERENCE_CHARGES:
            raise ValueError(f'LSDub Continuity: coherence_charge must be one of {", ".join(COHERENCE_CHARGES)}, not {coherence_charge!r}')
        return {} if coherence_charge == COHERENCE_CHARGES[0] else {'charge': coherence_charge}

    @classmethod
    def fingerprint_inputs(cls, project, source, render, shot_ids='', z=3.5, quality=18, coherence_charge='remade'):
        return stage_key('LSDubContinuity', {'ids': shot_ids, 'z': z, 'q': quality, **cls._charge(coherence_charge)}, [render.hash])

    @classmethod
    def execute(cls, project, source, render, shot_ids='', z=3.5, quality=18, coherence_charge='remade'):
        from .lsdub_engine.core import continuity as CONT
        from .project import parse_force_shots
        want = set(parse_force_shots(shot_ids))
        charge = cls._charge(coherence_charge)
        shots = {}
        for sid, v in render.shots.items():
            v = dict(v)
            if v.get('render') and (not want or int(sid) in want):
                d = os.path.dirname(v.get('gate') or v['render'])
                st = project.stage('shots/_continuity', stage_key('LSDubContinuity', {'z': z, **charge}, [v['key'], v['result']]))
                out = st.path('render.mp4')
                if not st.is_done():
                    CONT.run(os.path.join(d, 'take.mp4'), os.path.join(d, 'sung.mp4'), out, z=z, report=st.path('continuity.json'),
                             charge=coherence_charge)
                    st.mark_done(True)
                v.update(render=out, result=file_hash(out))
            shots[sid] = v
        song = T.Song.from_json(render.song) if render.audio == 'song' and render.song else None
        rec, _ = cut_render(project, source, shots, quality, render.audio if song else 'source', song)
        rec.on_fail = render.on_fail
        return io.NodeOutput(rec, LSDubDub._video(rec.cut))


NODES = [LSDubBuildLipKit, LSDubTweaks, LSDubOccluders, LSDubDub, LSDubContinuity]
