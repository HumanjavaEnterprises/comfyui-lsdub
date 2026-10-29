# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""VIDEO lane: the video slot, shot split, cast (who is in each shot) and triage (which shots can be dubbed)."""
import json
import os

from comfy_api.v0_0_2 import io

from . import lsdub_types as T
from . import media
from .lsdub_engine import presets
from .nodes_audio import GENERATION_LATER
from .project import file_hash, stage_key, text_hash

VERDICT_COLOUR = {'DUB': (60, 200, 60), 'DUB-MEDIUM': (0, 170, 255), 'NO': (60, 60, 210)}


class LSDubVideoSlot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubVideoSlot', display_name='LSDub Video', category='LSDub/video',
            description='The picture to dub. Bring a finished video (or a still); generation from a prompt comes in '
                        'a later release. Saved in the project, keyed by its bytes.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                io.Video.Input('video', optional=True, tooltip='The finished video (e.g. from Load Video). Takes priority.'),
                io.Image.Input('still', optional=True, tooltip='A still to dub or to animate (first image is used).'),
                io.String.Input('prompt', multiline=True, default='',
                                tooltip='What to generate (later release). Ignored when video is connected.'),
                io.String.Input('negative_prompt', multiline=True, default='',
                                tooltip='What to avoid when generating (later release).'),
                io.Float.Input('seconds', default=5.0, min=1.0, max=60.0, step=0.5,
                               tooltip='Length of a generated clip (later release).'),
            ],
            outputs=[T.LSDubSource.Output(display_name='source', tooltip='The source record: path, hash, fps, frames, size.'),
                     io.Video.Output(display_name='video', tooltip='The video, passed through for previews.')],
        )

    @classmethod
    def fingerprint_inputs(cls, project, video=None, still=None, prompt='', negative_prompt='', seconds=5.0):
        if video is not None:
            p, data = media.video_source(video)
            return stage_key('LSDubVideoSlot', {'video': file_hash(p) if p else text_hash(str(len(data)))}, [project.cache_key()])
        return stage_key('LSDubVideoSlot', {'still': None if still is None else float(still.sum()), 'prompt': prompt,
                                            'neg': negative_prompt, 'seconds': seconds}, [project.cache_key()])

    @classmethod
    def execute(cls, project, video=None, still=None, prompt='', negative_prompt='', seconds=5.0):
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
        if still is not None and not (prompt or '').strip():
            st = project.stage('slots/still', '_incoming')
            p = media.image_to_png(still, st.path('still.png'))
            h = file_hash(p)
            st2 = project.stage('slots/still', stage_key('LSDubVideoSlot', {'still': h}, [project.cache_key()]))
            dst = st2.path('still.png')
            os.replace(p, dst)
            st2.mark_done(True)
            return io.NodeOutput(T.Source(path=dst, hash=h, fps=project.fps, n_frames=1,
                                          wh=(int(still.shape[2]), int(still.shape[1])), still=True), None)
        if (prompt or '').strip():
            raise RuntimeError(GENERATION_LATER.format(node='LSDub Video', what='video', port='video'))
        raise ValueError('LSDub Video: connect a video (or a still) to dub.')


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
        return stage_key('LSDubCast', {'refs': refs, 'names': names, 'db': file_hash(cast_db) if cast_db else '',
                                       'dir': refs_dir}, [project.cache_key(), source.hash, shots.hash])

    @classmethod
    def fingerprint_inputs(cls, project, source, shots, singer_refs=None, names='', cast_db='', refs_dir=''):
        return cls._key(project, source, shots, singer_refs, names, cast_db, refs_dir)

    @classmethod
    def execute(cls, project, source, shots, singer_refs=None, names='', cast_db='', refs_dir=''):
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
            db = qa.build_db(cast_db or None, refs_dir or None, kv)
            if db is None: raise ValueError('LSDub Cast: give singer_refs + names, a cast_db file or a refs_dir')
            np.savez_compressed(db_path, **db)
            faces, emb = C.Faces(), C.Embedder(C.MODEL)
            mids = {s['shot']: (s['f0'] + s['f1']) // 2 for s in shots.shots}
            F = qa.read_frames(source.path, sorted(set(mids.values())))
            who = {}
            for sid, f in mids.items():
                img = F.get(f)
                fs = faces(img) if img is not None else []
                who[str(sid)] = sorted({qa.identify(db, emb, img, x, project.preset)[0] for x in fs})
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
        from .lsdub_engine.audits import qa
        from .lsdub_engine.core import cast as C
        st = project.stage('video/triage', cls._key(project, source, shots, song, cast, samples))
        out = st.path('triage')
        if not st.is_done():
            db = C.load_db(cast.db) if cast else None
            sing = presets.get(project.preset).singers
            targets = tuple(sing) if (sing and db is not None) else ()
            vocal = (song.lead or song.vocal or song.mix) if song else None
            meta, rows = qa.triage(source.path, vocal, project.song_offset, db, targets, samples,
                                   detected=load_detected(shots), preset=project.preset)
            qa.write_triage(meta, rows, out)
            if source.name:                                              # the report names the user's file, not our copy
                md = open(out + '.md').read().split('\n', 1)
                with open(out + '.md', 'w') as f: f.write(f'# LSDub QA triage: {source.name}\n' + md[1])
            st.mark_done(True, **{v: sum(r['verdict'] == v for r in rows) for v in VERDICT_COLOUR})
        rows = json.load(open(out + '.json'))['shots']
        rec = T.Shots(shots=rows, hash=file_hash(out + '.json'), sheet=out + '_sheet.jpg', features=shots.features)
        return io.NodeOutput(rec, media.load_image(out + '_sheet.jpg'), open(out + '.md').read())


NODES = [LSDubVideoSlot, LSDubShotSplit, LSDubCast, LSDubTriage]
