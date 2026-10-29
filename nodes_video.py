# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""VIDEO lane: the video slot, shot split, cast (who is in each shot) and triage (which shots can be dubbed)."""
import json
import os

from comfy_api.v0_0_2 import io

from . import generate, graphs
from . import lsdub_types as T
from . import media
from .lsdub_engine import presets
from .project import file_hash, stage_key

VERDICT_COLOUR = {'DUB': (60, 200, 60), 'DUB-MEDIUM': (0, 170, 255), 'NO': (60, 60, 210)}


def _fit(wh, long_side=1024, multiple=32):
    """a still's aspect at ~long_side px, both sides a multiple of 32 (LTX)."""
    w, h = wh
    k = long_side / max(w, h)
    return max(multiple, int(round(w * k / multiple)) * multiple), max(multiple, int(round(h * k / multiple)) * multiple)


class LSDubStillSlot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubStillSlot', display_name='LSDub Still', category='LSDub/video',
            description='A still for the video slot: bring an image, or edit one with Qwen-Image-Edit-2511 (open '
                        'weights) by an instruction. Local GPU or Comfy Cloud per the project.',
            hidden=[io.Hidden.api_key_comfy_org],
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                io.Image.Input('image', tooltip='The image to use or to edit (first image of the batch).'),
                io.String.Input('edit_prompt', multiline=True, default='',
                                tooltip='Optional edit instruction, e.g. "make her hold a chrome microphone". Empty = '
                                        'use the image as it is. (This model takes no negative prompt.)'),
                io.Image.Input('ref', optional=True, tooltip='Optional second reference image (a costume, a prop).'),
            ],
            outputs=[io.Image.Output(display_name='image', tooltip='The still (edited or as given).')],
        )

    @classmethod
    def _key(cls, project, image, edit_prompt, ref):
        h = media.audio_hash({'waveform': image[None], 'sample_rate': 0})
        r = media.audio_hash({'waveform': ref[None], 'sample_rate': 0}) if ref is not None else ''
        return stage_key('LSDubStillSlot', {'img': h, 'ref': r, 'p': edit_prompt.strip(), 'seed': project.seed,
                                            'backend': project.backend}, [project.cache_key()])

    @classmethod
    def fingerprint_inputs(cls, project, image, edit_prompt='', ref=None):
        return cls._key(project, image, edit_prompt, ref)

    @classmethod
    async def execute(cls, project, image, edit_prompt='', ref=None):
        if not (edit_prompt or '').strip():
            return io.NodeOutput(image[:1])
        st = project.stage('slots/still', cls._key(project, image, edit_prompt, ref))
        out = st.path('still.png')
        if not st.is_done():
            ups = {'img': media.image_to_png(image, st.path('in.png'))}
            if ref is not None: ups['ref'] = media.image_to_png(ref, st.path('ref.png'))
            g = graphs.still('@img', edit_prompt.strip(), project.seed, ref='@ref' if ref is not None else None,
                             prefix=f'lsdub/still_{os.path.basename(st.dir)}')
            paths = await generate.run(g, project.backend, uploads=ups, out_dir=st.path('gen'), api_key=media.hidden_api_key(cls))
            os.replace(paths[0], out)
            st.mark_done(True, backend=project.backend)
        return io.NodeOutput(media.load_image(out))


class LSDubVideoSlot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubVideoSlot', display_name='LSDub Video', category='LSDub/video',
            description='The picture to dub. Bring a finished video, or animate a still with LTX-2.5 (open weights) from '
                        'a prompt. Saved in the project, keyed by its bytes or its settings.',
            hidden=[io.Hidden.api_key_comfy_org],
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                io.Video.Input('video', optional=True, tooltip='The finished video (e.g. from Load Video). Takes priority.'),
                io.Image.Input('still', optional=True, tooltip='A still: dubbed as is, or animated when a prompt is given '
                                                              '(use LSDub Still to make or edit one).'),
                io.String.Input('prompt', multiline=True, default='',
                                tooltip='What happens in the clip (generated from the still). Ignored when video is connected.'),
                io.String.Input('negative_prompt', multiline=True, default='',
                                tooltip='What to avoid in the generated clip.'),
                io.Float.Input('seconds', default=5.0, min=1.0, max=20.0, step=0.5, tooltip='Length of a generated clip, s.'),
            ],
            outputs=[T.LSDubSource.Output(display_name='source', tooltip='The source record: path, hash, fps, frames, size.'),
                     io.Video.Output(display_name='video', tooltip='The video, for previews.')],
        )

    @classmethod
    def _gen_key(cls, project, still, prompt, negative_prompt, seconds):
        return stage_key('LSDubVideoSlot/gen', {'still': media.audio_hash({'waveform': still[None], 'sample_rate': 0}),
                                                'p': prompt.strip(), 'n': negative_prompt.strip(), 's': seconds,
                                                'seed': project.seed, 'backend': project.backend}, [project.cache_key()])

    @classmethod
    def fingerprint_inputs(cls, project, video=None, still=None, prompt='', negative_prompt='', seconds=5.0):
        if video is not None:
            p, data = media.video_source(video)
            import hashlib
            return stage_key('LSDubVideoSlot', {'video': file_hash(p) if p else hashlib.sha1(data).hexdigest()}, [project.cache_key()])
        if still is not None and (prompt or '').strip():
            return cls._gen_key(project, still, prompt, negative_prompt, seconds)
        return stage_key('LSDubVideoSlot', {'still': None if still is None else media.audio_hash({'waveform': still[None], 'sample_rate': 0}),
                                            'prompt': prompt}, [project.cache_key()])

    @classmethod
    async def execute(cls, project, video=None, still=None, prompt='', negative_prompt='', seconds=5.0):
        if video is not None:
            tmp = project.stage('slots/video', '_incoming').path('source.mp4')
            h = media.store_video(video, tmp)
            st = project.stage('slots/video', stage_key('LSDubVideoSlot', {'video': h}, [project.cache_key()]))
            dst = st.path('source.mp4')
            os.replace(tmp, dst)
            fps, n, wh = media.probe_video(dst)
            st.mark_done(True, fps=fps, frames=n)
            src_path = media.video_source(video)[0]
            return io.NodeOutput(T.Source(path=dst, hash=h, fps=fps, n_frames=n, wh=wh,
                                          name=os.path.basename(src_path) if src_path else ''), video)
        if still is not None and (prompt or '').strip():
            st = project.stage('slots/video', cls._gen_key(project, still, prompt, negative_prompt, seconds))
            dst = st.path('source.mp4')
            if not st.is_done():
                img = media.image_to_png(still, st.path('still.png'))
                w, h = _fit((int(still.shape[2]), int(still.shape[1])))
                g = graphs.video('@img', prompt.strip(), negative_prompt.strip(), seconds, project.seed, project.fps, w, h,
                                 prefix=f'lsdub/video_{os.path.basename(st.dir)}')
                paths = await generate.run(g, project.backend, uploads={'img': img}, out_dir=st.path('gen'),
                                           api_key=media.hidden_api_key(cls))
                os.replace(paths[0], dst)
                st.mark_done(True, backend=project.backend)
            fps, n, wh = media.probe_video(dst)
            from .nodes_mouth import LSDubDub
            return io.NodeOutput(T.Source(path=dst, hash=file_hash(dst), fps=fps, n_frames=n, wh=wh, name='generated.mp4'),
                                 LSDubDub._video(dst))
        if still is not None:
            st = project.stage('slots/still', '_incoming')
            p = media.image_to_png(still, st.path('still.png'))
            h = file_hash(p)
            st2 = project.stage('slots/still', stage_key('LSDubVideoSlot', {'still': h}, [project.cache_key()]))
            dst = st2.path('still.png')
            os.replace(p, dst)
            st2.mark_done(True)
            from .nodes_mouth import LSDubDub
            prev = media.placeholder_video(st2.path('still.mp4'), 'a still (dubbed as one frame)')
            return io.NodeOutput(T.Source(path=dst, hash=h, fps=project.fps, n_frames=1,
                                          wh=(int(still.shape[2]), int(still.shape[1])), still=True), LSDubDub._video(prev))
        if (prompt or '').strip():
            raise ValueError('LSDub Video: to generate, connect a still (LSDub Still makes or edits one) and a prompt.')
        raise ValueError('LSDub Video: connect a video (or a still) to dub.')


class LSDubPerformance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubPerformance', display_name='LSDub Performance', category='LSDub/video',
            description='A character image + a song -> a performance video as long as the song: LTX-2.5 image-to-video '
                        'shots of ~5 s, placed one after another on the song timeline (open weights; local GPU or Comfy '
                        'Cloud per the project). Dub it afterwards to make the mouth sing the words.',
            hidden=[io.Hidden.api_key_comfy_org],
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project (fps, seed, backend).'),
                io.Image.Input('character', tooltip='The character to animate (first image is used).'),
                T.LSDubSong.Input('song', tooltip='The song: the video is made as long as it.'),
                io.String.Input('prompt', multiline=True, default='close-up, the character sings to the camera, mouth clearly visible',
                                tooltip='The performance: what the character does, the camera, the light.'),
                io.String.Input('negative_prompt', multiline=True, default='blurry, distorted face, extra fingers, text',
                                tooltip='What to avoid.'),
                io.Float.Input('shot_seconds', default=5.0, min=2.0, max=10.0, step=0.5,
                               tooltip='Length of each generated shot, s.'),
                io.Float.Input('max_seconds', default=0.0, min=0.0, max=600.0, step=1.0, optional=True,
                               tooltip='Stop after this many seconds of the song (0 = the whole song).'),
                io.Combo.Input('chain', options=['same_still', 'last_frame'], default='same_still', optional=True,
                               tooltip='same_still: every shot starts from the character image (a new seed per shot) and '
                                       'keeps the identity best; last_frame: each shot continues from the previous one '
                                       '(smoother, but the face drifts over a long song).'),
            ],
            outputs=[T.LSDubSource.Output(display_name='source', tooltip='The performance video (with the song).'),
                     io.Video.Output(display_name='video', tooltip='The performance video.')],
        )

    @classmethod
    def _base(cls, project, character, song, prompt, negative_prompt, shot_seconds, max_seconds, chain):
        return {'img': media.audio_hash({'waveform': character[:1][None], 'sample_rate': 0}), 'song': song.cache_key(),
                'p': prompt.strip(), 'n': negative_prompt.strip(), 'shot': shot_seconds, 'max': max_seconds, 'chain': chain,
                'seed': project.seed, 'backend': project.backend, 'fps': project.fps}

    @classmethod
    def fingerprint_inputs(cls, project, character, song, prompt, negative_prompt, shot_seconds=5.0, max_seconds=0.0,
                           chain='same_still'):
        return stage_key('LSDubPerformance', cls._base(project, character, song, prompt, negative_prompt, shot_seconds,
                                                       max_seconds, chain), [project.cache_key(), project.dir])

    @classmethod
    async def execute(cls, project, character, song, prompt, negative_prompt, shot_seconds=5.0, max_seconds=0.0,
                      chain='same_still'):
        import math
        import cv2
        from .lsdub_engine.core import medium as MED
        from .nodes_mouth import LSDubDub, _write_frames
        if project.song_offset:
            raise ValueError('LSDub Performance: the project\'s song_offset must be 0 (the performance starts with the '
                             'song); set it on the project, or trim the song first.')
        base = cls._base(project, character, song, prompt, negative_prompt, shot_seconds, max_seconds, chain)
        st = project.stage('slots/video', stage_key('LSDubPerformance', base))
        dst = st.path('source.mp4')
        if not st.is_done():
            dur = media.duration(song.mix)
            if max_seconds and max_seconds > 0: dur = min(dur, float(max_seconds))
            need = int(round(dur * project.fps))
            n = max(1, math.ceil(dur / shot_seconds))
            still = media.image_to_png(character, st.path('character.png'))
            w, h = _fit((int(character.shape[2]), int(character.shape[1])))
            frames, key = [], media.audio_hash({'waveform': character[:1][None], 'sample_rate': 0})
            try:
                import comfy.utils
                pbar = comfy.utils.ProgressBar(n)
            except Exception:
                pbar = None
            i = 0
            while i < n or len(frames) < need:                        # a generated shot can be a few frames short of
                if i >= n + 8: raise RuntimeError('LSDub Performance: the video model returns too few frames per shot')
                if i >= n and pbar: pbar.total = i + 1                 # shot_seconds: keep going until the song is covered
                sk = stage_key('LSDubPerformance/shot', {**base, 'i': i, 'start': key})
                sst = project.stage('slots/video/shots', sk)
                clip = sst.path('shot.mp4')
                if not sst.is_done():
                    g = graphs.video('@img', prompt.strip(), negative_prompt.strip(), shot_seconds, project.seed + 1000 * i,
                                     project.fps, w, h, prefix=f'lsdub/perf_{sk}')
                    paths = await generate.run(g, project.backend, uploads={'img': still}, out_dir=sst.path('gen'),
                                               api_key=media.hidden_api_key(cls))
                    os.replace(paths[0], clip)
                    sst.mark_done(True)
                F, _ = MED.read(clip)
                frames += F
                if chain == 'last_frame' and F:                       # the next shot starts where this one ended
                    still = sst.path('last.png')
                    cv2.imwrite(still, F[-1])
                    key = file_hash(still)
                if pbar: pbar.update(1)
                i += 1
            frames = frames[:need]
            silent = _write_frames(frames, st.path('silent.mp4'), project.fps, crf=12)
            from .lsdub_engine.io import ff
            r = ff.run(['ffmpeg', '-v', 'error', '-y', '-i', silent, '-i', song.mix, '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
                        '-c:a', 'aac', '-shortest', dst + '.part.mp4'], capture_output=True)
            if r.returncode != 0: raise RuntimeError('ffmpeg failed muxing the performance')
            os.replace(dst + '.part.mp4', dst)
            st.mark_done(True, shots=i, seconds=round(dur, 3))
        fps, nf, wh = media.probe_video(dst)
        return io.NodeOutput(T.Source(path=dst, hash=file_hash(dst), fps=fps, n_frames=nf, wh=wh, name='performance.mp4'),
                             LSDubDub._video(dst))


def _shots_key(project, source):
    return stage_key('LSDubShotSplit', {}, [project.cache_key(), source.hash])


def load_detected(shots):
    """the stored detect_shots() result of a SHOTS record: (shots, fps, (W, H), (dh, dp))."""
    import numpy as np
    z = np.load(shots.features)
    return shots.shots, float(z['fps']), tuple(int(v) for v in z['size']), (z['dh'], z['dp'])


class LSDubShotSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubShotSplit', display_name='LSDub Shot Split', category='LSDub/video',
            description='Find the cuts: one shot per camera take. Fades and flashes are not cuts.',
            inputs=[T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                    T.LSDubSource.Input('source', tooltip='The video (from LSDub Video).')],
            outputs=[T.LSDubShots.Output(display_name='shots', tooltip='The shots: id, frames, start/end seconds.'),
                     io.Image.Output(display_name='sheet', tooltip='Contact sheet: the middle frame of each shot.')],
        )

    @classmethod
    def fingerprint_inputs(cls, project, source):
        return _shots_key(project, source)

    @classmethod
    def execute(cls, project, source):
        from .lsdub_engine.audits import qa
        import numpy as np
        st = project.stage('video/shots', _shots_key(project, source))
        js, feats, sheet = st.path('shots.json'), st.path('features.npz'), st.path('sheet.jpg')
        if not st.is_done():
            if source.still:
                shots, fps, size = [{'shot': 0, 'f0': 0, 'f1': 0, 'frames': 1, 'start': 0.0, 'end': 1 / project.fps}], project.fps, source.wh
                dh = dp = np.zeros(1)
            else:
                shots, fps, size, (dh, dp) = qa.detect_shots(source.path)
            np.savez(feats, dh=dh, dp=dp, fps=fps, size=np.array(size))
            media.dump(shots, js)
            if not source.still:
                qa.sheet(source.path, shots, sheet, lambda s: (200, 200, 200), lambda s: f"{s['frames']} f")
            st.mark_done(True, shots=len(shots))
        shots = json.load(open(js))
        rec = T.Shots(shots=shots, hash=file_hash(js), sheet=sheet if os.path.exists(sheet) else '', features=feats)
        return io.NodeOutput(rec, media.load_image(sheet))


class LSDubCast(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubCast', display_name='LSDub Cast', category='LSDub/video',
            description='Who is who: enroll the singers from reference pictures (or a saved cast file), then name the '
                        'faces in every shot.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project (its preset names the singers).'),
                T.LSDubSource.Input('source', tooltip='The video.'),
                T.LSDubShots.Input('shots', tooltip='The shots (from LSDub Shot Split).'),
                io.Image.Input('singer_refs', optional=True,
                               tooltip='Reference pictures of the singers, one face each; names gives their names in order.'),
                io.String.Input('names', default='',
                                tooltip='Comma-separated names for the singer_refs images, in order (e.g. "alice, bob").'),
                io.String.Input('cast_db', default='', optional=True,
                                tooltip='Optional: a saved cast file (.npz) instead of reference pictures.'),
                io.String.Input('refs_dir', default='', optional=True,
                                tooltip='Optional: a folder with one sub-folder of pictures per person (<dir>/<name>/*.png).'),
            ],
            outputs=[T.LSDubCast.Output(display_name='cast', tooltip='The cast file and who appears in each shot.'),
                     io.Image.Output(display_name='sheet', tooltip='Contact sheet labelled with the faces found.')],
        )

    @classmethod
    def _key(cls, project, source, shots, singer_refs, names, cast_db, refs_dir):
        refs = None if singer_refs is None else media.audio_hash({'waveform': singer_refs[None], 'sample_rate': 0})
        from .nodes_mouth import LSDubBuildLipKit
        listing = LSDubBuildLipKit._hash(refs_dir) if refs_dir and os.path.isdir(refs_dir) else ''
        return stage_key('LSDubCast', {'refs': refs, 'names': names, 'db': file_hash(cast_db) if cast_db else '',
                                       'dir': refs_dir, 'dir_files': listing}, [project.cache_key(), source.hash, shots.hash])

    @classmethod
    def fingerprint_inputs(cls, project, source, shots, singer_refs=None, names='', cast_db='', refs_dir=''):
        return cls._key(project, source, shots, singer_refs, names, cast_db, refs_dir)

    @classmethod
    def execute(cls, project, source, shots, singer_refs=None, names='', cast_db='', refs_dir=''):
        from . import devices as _dev
        _dev.ensure_face_models()
        import numpy as np
        from .lsdub_engine.audits import qa
        from .lsdub_engine.core import cast as C
        st = project.stage('video/cast', cls._key(project, source, shots, singer_refs, names, cast_db, refs_dir))
        db_path, per_shot, sheet = st.path('cast_db.npz'), st.path('shots_cast.json'), st.path('sheet.jpg')
        if not st.is_done():
            kv = []
            if singer_refs is not None:
                nm = [x.strip() for x in names.split(',') if x.strip()]
                if len(nm) != int(singer_refs.shape[0]):
                    raise ValueError(f'LSDub Cast: {int(singer_refs.shape[0])} reference images but {len(nm)} names')
                for i, n in enumerate(nm):
                    kv.append(f'{n}={media.image_to_png(singer_refs[i:i + 1], st.path(f"ref_{i:02d}_{n}.png"))}')
            from . import devices
            emb = devices.embedder()
            try:
                db = qa.build_db(cast_db or None, refs_dir or None, kv, emb=emb)
                if db is None: raise ValueError('LSDub Cast: give singer_refs + names, a cast_db file or a refs_dir')
                np.savez_compressed(db_path, **db)
                faces = C.Faces()
                mids = {s['shot']: (s['f0'] + s['f1']) // 2 for s in shots.shots}
                F = qa.read_frames(source.path, sorted(set(mids.values())))
                who = {}
                for sid, f in mids.items():
                    img = F.get(f)
                    fs = faces(img) if img is not None else []
                    who[str(sid)] = sorted({qa.identify(db, emb, img, x, project.preset)[0] for x in fs})
            finally:
                devices.release(emb)
            media.dump(who, per_shot)
            qa.sheet(source.path, shots.shots, sheet, lambda s: (200, 200, 200), lambda s: ' '.join(who.get(str(s['shot'])) or ['-']))
            st.mark_done(True, names=sorted(set(db['name'].tolist())))
        return io.NodeOutput(T.Cast(db=db_path, scan=per_shot, hash=file_hash(per_shot) + file_hash(db_path)[:8]),
                             media.load_image(sheet))


class LSDubTriage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubTriage', display_name='LSDub Triage', category='LSDub/video',
            description='Which shots can be dubbed: DUB (close-up), DUB-MEDIUM (a smaller face, dubbed through a crop) '
                        'or NO, each with its reason (face too small, profile, mouth hidden, no singing).',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project (song_offset, preset singers).'),
                T.LSDubSource.Input('source', tooltip='The video.'),
                T.LSDubShots.Input('shots', tooltip='The shots (from LSDub Shot Split).'),
                T.LSDubSong.Input('song', optional=True, tooltip='The song: its vocal marks which shots are sung.'),
                T.LSDubCast.Input('cast', optional=True, tooltip='The cast: judge only the singers\' faces. Without it, '
                                                                 'the largest face in each shot is judged.'),
                io.Int.Input('samples', default=6, min=1, max=50, tooltip='Frames sampled per shot.'),
            ],
            outputs=[T.LSDubShots.Output(display_name='shots', tooltip='The shots with verdict, reason, face size, singer.'),
                     io.Image.Output(display_name='sheet', tooltip='Contact sheet coloured by verdict.'),
                     io.String.Output(display_name='report', tooltip='Markdown: every shot, worst first, with its reason.')],
        )

    @classmethod
    def _key(cls, project, source, shots, song, cast, samples):
        return stage_key('LSDubTriage', {'samples': samples}, [project.cache_key(), source.hash, shots.hash,
                                                               song.cache_key() if song else '', cast.cache_key() if cast else ''])

    @classmethod
    def fingerprint_inputs(cls, project, source, shots, song=None, cast=None, samples=6):
        return cls._key(project, source, shots, song, cast, samples)

    @classmethod
    def execute(cls, project, source, shots, song=None, cast=None, samples=6):
        from . import devices as _dev
        _dev.ensure_face_models()
        from .lsdub_engine.audits import qa
        from .lsdub_engine.core import cast as C
        st = project.stage('video/triage', cls._key(project, source, shots, song, cast, samples))
        out = st.path('triage')
        if not st.is_done():
            db = C.load_db(cast.db) if cast else None
            sing = presets.get(project.preset).singers
            targets = tuple(sing) if (sing and db is not None) else ()
            vocal = (song.lead or song.vocal or song.mix) if song else None
            from . import devices
            emb = devices.embedder() if db is not None else None
            try:
                meta, rows = qa.triage(source.path, vocal, project.song_offset, db, targets, samples,
                                       detected=load_detected(shots), preset=project.preset, emb=emb)
            finally:
                if emb is not None: devices.release(emb)
            qa.write_triage(meta, rows, out)
            if source.name:                                              # the report names the user's file, not our copy
                md = open(out + '.md').read().split('\n', 1)
                with open(out + '.md', 'w') as f: f.write(f'# LSDub QA triage: {source.name}\n' + md[1])
            st.mark_done(True, **{v: sum(r['verdict'] == v for r in rows) for v in VERDICT_COLOUR})
        rows = json.load(open(out + '.json'))['shots']
        rec = T.Shots(shots=rows, hash=file_hash(out + '.json'), sheet=out + '_sheet.jpg', features=shots.features)
        return io.NodeOutput(rec, media.load_image(out + '_sheet.jpg'), open(out + '.md').read())


NODES = [LSDubStillSlot, LSDubVideoSlot, LSDubPerformance, LSDubShotSplit, LSDubCast, LSDubTriage]
