# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""QA: verify (likely success per dubbed shot), scorecard (every shot, 0-100), sync QA (measured mouth vs vocal, a
review card), fix-it (each problem -> the tweak to try + the force_shots to re-run) and save the cut."""
import json
import os
import shutil

from comfy_api.v0_0_2 import io

from . import lsdub_types as T
from . import media
from .project import file_hash, parse_force_shots, stage_key


def _vocal(song):
    return song.lead or song.vocal or song.mix


def _cuts(shots, only=()):
    return {'cut': [{'start': s['start'], 'dur': round(s['end'] - s['start'], 4), 'shot': s['shot']}
                    for s in shots if not only or s['shot'] in only]}


def _cut_video(project, source, render):
    """what to judge: the dub, or in test_learn the source."""
    return render.cut if render is not None and render.cut else source.path


class LSDubVerify(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubVerify', display_name='LSDub Verify', category='LSDub/qa',
            description='How likely each dubbed shot is to have worked: a 0-100 score and PASS / REVIEW / FAIL, compared '
                        'with the source (sync, mouth travel, silences, identity drift).',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                T.LSDubSource.Input('source', tooltip='The video before the dub.'),
                T.LSDubRender.Input('render', tooltip='The dub to verify.'),
                T.LSDubShots.Input('shots', tooltip='The triaged shots.'),
                T.LSDubSong.Input('song', tooltip='The song (its lead vocal).'),
                T.LSDubAlign.Input('align', tooltip='The word timings.'),
            ],
            outputs=[T.LSDubReport.Output(display_name='report', tooltip='Per-shot band and score, the shots to fix.'),
                     io.String.Output(display_name='md', tooltip='Markdown: worst first, with the reason.')],
        )

    @classmethod
    def _key(cls, project, source, render, shots, song, align):
        return stage_key('LSDubVerify', {}, [project.cache_key(), source.hash, render.hash, shots.hash, song.cache_key(), align.hash])

    @classmethod
    def fingerprint_inputs(cls, project, source, render, shots, song, align):
        return cls._key(project, source, render, shots, song, align)

    @classmethod
    def execute(cls, project, source, render, shots, song, align):
        from .lsdub_engine.audits import qa
        st = project.stage('qa/verify', cls._key(project, source, render, shots, song, align))
        out = st.path('verify')
        if not st.is_done():
            sj = media.dump({'fps': source.fps, 'offset': project.song_offset, 'shots': shots.shots}, st.path('shots.json'))
            meta, rows = qa.verify(_cut_video(project, source, render), _vocal(song), sj, before=source.path, align=align.align,
                                   work=st.path('work'))
            qa.write_verify(meta, rows, out)
            st.mark_done(True)
        J = json.load(open(out + '.json'))
        verdicts = {str(r['shot']): r['band'] for r in J['shots']}
        fix = sorted(int(k) for k, v in verdicts.items() if v in ('FAIL', 'REVIEW'))
        return io.NodeOutput(T.Report(json=out + '.json', md=out + '.md', verdicts=verdicts, fix=fix, hash=file_hash(out + '.json')),
                             open(out + '.md').read())


class LSDubScorecard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubScorecard', display_name='LSDub Scorecard', category='LSDub/qa',
            description='Score every shot of the cut 0-100 (sync, silences, snap, face size, mouth visible, motion, pops, '
                        'sharpness), worst first.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                T.LSDubSource.Input('source', tooltip='The video (judged in test_learn, when there is no dub).'),
                T.LSDubShots.Input('shots', tooltip='The shots.'),
                T.LSDubSong.Input('song', tooltip='The song (its lead vocal).'),
                T.LSDubAlign.Input('align', tooltip='The word timings.'),
                T.LSDubRender.Input('render', optional=True, tooltip='The dub to score (else the source).'),
            ],
            outputs=[T.LSDubReport.Output(display_name='report', tooltip='Per-shot class and score.'),
                     io.String.Output(display_name='md', tooltip='Markdown scorecard.')],
        )

    @classmethod
    def _key(cls, project, source, shots, song, align, render):
        return stage_key('LSDubScorecard', {}, [project.cache_key(), source.hash, shots.hash, song.cache_key(), align.hash,
                                                render.hash if render else ''])

    @classmethod
    def fingerprint_inputs(cls, project, source, shots, song, align, render=None):
        return cls._key(project, source, shots, song, align, render)

    @classmethod
    def execute(cls, project, source, shots, song, align, render=None):
        from .lsdub_engine.audits import scorecard as SC
        st = project.stage('qa/scorecard', cls._key(project, source, shots, song, align, render))
        out = st.path('scorecard')
        video = _cut_video(project, source, render)
        if not st.is_done():
            rows, thumbs = SC.score(video, _cuts(shots.shots), _vocal(song), project.song_offset, align.align,
                                    work=st.path('work'), preset=project.preset)
            SC.write(rows, thumbs, out, video)
            st.mark_done(True)
        J = json.load(open(out + '.json'))
        verdicts = {str(r.get('shot', r['cut'])): r.get('class') for r in J['cuts']}
        return io.NodeOutput(T.Report(json=out + '.json', md=out + '.md', verdicts=verdicts, hash=file_hash(out + '.json')),
                             open(out + '.md').read())


class LSDubSyncQA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubSyncQA', display_name='LSDub Sync QA', category='LSDub/qa',
            description='Measure the mouth against the vocal on every sung shot (correlation and lag), and make a '
                        'review card: the shot with the vocal waveform scrolling past a playhead.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                T.LSDubSource.Input('source', tooltip='The video (judged when there is no dub).'),
                T.LSDubShots.Input('shots', tooltip='The shots.'),
                T.LSDubSong.Input('song', tooltip='The song (its lead vocal).'),
                T.LSDubAlign.Input('align', tooltip='The word timings (the target mouth opening).'),
                T.LSDubRender.Input('render', optional=True, tooltip='The dub to measure (else the source).'),
                io.String.Input('shot_ids', default='', tooltip='Only these shots, e.g. "16,21" (empty = every sung shot). '
                                                                'The card shows the first.'),
            ],
            outputs=[T.LSDubReport.Output(display_name='report', tooltip='Per-shot corr / lag.'),
                     io.Video.Output(display_name='card', tooltip='Review card of the first shot: video + scrolling vocal.'),
                     io.String.Output(display_name='md', tooltip='Markdown: corr and lag per sung shot.')],
        )

    @classmethod
    def _key(cls, project, source, shots, song, align, render, shot_ids):
        return stage_key('LSDubSyncQA', {'only': list(parse_force_shots(shot_ids))},
                         [project.cache_key(), source.hash, shots.hash, song.cache_key(), align.hash, render.hash if render else ''])

    @classmethod
    def fingerprint_inputs(cls, project, source, shots, song, align, render=None, shot_ids=''):
        return cls._key(project, source, shots, song, align, render, shot_ids)

    @classmethod
    def execute(cls, project, source, shots, song, align, render=None, shot_ids=''):
        from .lsdub_engine.audits import sync_qa as Q, sync_strip as SS
        from .lsdub_engine._inproc import call
        from .nodes_mouth import LSDubDub, extract_shot
        only = parse_force_shots(shot_ids)
        st = project.stage('qa/syncqa', cls._key(project, source, shots, song, align, render, shot_ids))
        js, md, card = st.path('syncqa.json'), st.path('syncqa.md'), st.path('card.mp4')
        video = _cut_video(project, source, render)
        if not st.is_done():
            cuts = _cuts(shots.shots, only)
            rows = Q.score_film(video, cuts, _vocal(song), project.song_offset, align.align, work=st.path('work'),
                               preset=project.preset)
            media.dump(rows, js)
            with open(md, 'w') as f: f.write(Q.film_md(rows))
            first = next((r for r in rows if r.get('corr') is not None), rows[0] if rows else None)
            if first is not None:
                s = next(x for x in shots.shots if x['shot'] == first['shot'])
                clip = extract_shot(video, s['f0'], s['f1'], st.path('shot.mp4'))
                call(SS.main, ['--video', clip, '--out', card, '--audio', _vocal(song), '--t0', f"{s['start'] - project.song_offset:.3f}"],
                     quiet=True, name='strip')
            st.mark_done(True, shots=len(json.load(open(js))))
        rows = json.load(open(js))
        verdicts = {str(r.get('shot')): r.get('corr') for r in rows}
        return io.NodeOutput(T.Report(json=js, md=md, verdicts=verdicts, hash=file_hash(js)),
                             LSDubDub._video(card) if os.path.exists(card) else None, open(md).read())


# reason (substring) -> what to try
FIXES = [
    ('too small', 'Use a closer framing or a higher-resolution source; mediums dub through a crop (DUB-MEDIUM >= 120 px @1080).'),
    ('profile', 'The face is turned away: pick another take or accept the shot as is.'),
    ('mouth hidden', 'Mask what covers the mouth with LSDub Occluders (phrase "microphone" / "hand"), then dub again.'),
    ('mic', 'Mask the microphone with LSDub Occluders, then dub again.'),
    ('no singing', 'No vocal under this shot: leave it.'),
    ('too short', 'Shorter than the minimum shot: leave it.'),
    ('identity drift', 'The face changed: lower face_target or turn auto off in LSDub Mouth Tweaks, then force the shot.'),
    ('mouth off the vocal', 'Sync is off: try lead 0.03-0.10 in LSDub Mouth Tweaks (check the words in the lyric check first).'),
    ('open in the silences', 'Mouth open in the gaps: raise word_close (0.08-0.12) in LSDub Mouth Tweaks.'),
    ('soft attacks', 'Attacks are soft: keep snap on and lower lead a little.'),
    ('barely moves', 'Held gape or mumble: raise word_close, and check the vocal is the dry lead.'),
    ('temporal pops', 'Run LSDub Continuity on this shot with a lower z (3.0).'),
    ('off-model', 'The mouth does not match the face: rebuild or swap the lip kit for this singer.'),
]


def fix_for(reason):
    r = (reason or '').lower()
    return next((f for k, f in FIXES if k in r), 'Review the shot by eye (LSDub Sync QA card).')


class LSDubFixIt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubFixIt', display_name='LSDub Fix It', category='LSDub/qa',
            description='Turn the QA into a to-do list: each problem shot, why, the tweak to try, and the force_shots '
                        'string to paste into LSDub Project to redo just those shots.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                T.LSDubShots.Input('shots', tooltip='The triaged shots (their reasons).'),
                T.LSDubReport.Input('report', optional=True, tooltip='A Verify report (bands + reasons).'),
                T.LSDubReport.Input('sync', optional=True, tooltip='A Sync QA report (measured corr).'),
                io.Float.Input('min_corr', default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip='A sung shot measuring below this sync corr is listed.'),
            ],
            outputs=[io.String.Output(display_name='md', tooltip='Markdown to-do: shot, problem, what to try.'),
                     io.String.Output(display_name='force_shots', tooltip='Paste into LSDub Project > force_shots.')],
        )

    @classmethod
    def execute(cls, project, shots, report=None, sync=None, min_corr=0.5):
        items = {}
        for s in shots.shots:
            if s.get('verdict') == 'NO' and s.get('voiced') and s.get('target'):
                items.setdefault(s['shot'], []).append(('triage: ' + s['reason'], fix_for(s['reason']), False))
            elif s.get('verdict') in ('DUB', 'DUB-MEDIUM') and (s.get('mic') or 0) >= 0.10:   # the triage's mic line
                items.setdefault(s['shot'], []).append(('triage: mic near the mouth', fix_for('mic'), True))
        if report is not None and report.json and os.path.exists(report.json):
            for r in json.load(open(report.json)).get('shots', []):
                if r.get('band') in ('FAIL', 'REVIEW'):
                    items.setdefault(r['shot'], []).append((f"verify {r['band']} {r.get('score')}: {r.get('reason')}", fix_for(r.get('reason')), True))
        if sync is not None:
            for sid, c in (sync.verdicts or {}).items():
                if c is not None and sid not in ('None', '') and float(c) < min_corr:
                    items.setdefault(int(sid), []).append((f'sync corr {float(c):.2f} < {min_corr}', fix_for('mouth off the vocal'), True))
        redo = sorted(k for k, v in items.items() if any(x[2] for x in v))
        L = ['# LSDub fix-it', '', f"Mode {project.mode}. {len(items)} shot(s) to look at; redo: {','.join(map(str, redo)) or 'none'}.", '',
             '| shot | problem | what to try |', '|---|---|---|']
        for k in sorted(items):
            for prob, fix, _ in items[k]: L.append(f'| {k} | {prob} | {fix} |')
        if not items: L.append('| - | nothing flagged | - |')
        return io.NodeOutput('\n'.join(L) + '\n', ','.join(map(str, redo)))


class LSDubSaveCut(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubSaveCut', display_name='LSDub Save Cut', category='LSDub/qa', is_output_node=True,
            description='Save the finished cut to the project\'s final/ folder, with LSDub\'s invisible watermark.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                T.LSDubRender.Input('render', tooltip='The dub (its assembled cut).'),
                io.Boolean.Input('watermark', default=True, tooltip='Mark the file as AI-edited (invisible watermark + '
                                                                    'metadata tag). Keep it on when you publish.'),
            ],
            outputs=[io.Video.Output(display_name='video', tooltip='The saved cut.'),
                     io.String.Output(display_name='path', tooltip='Where it was saved.')],
        )

    @classmethod
    def execute(cls, project, render, watermark=True):
        from .lsdub_engine import watermark as WM
        from .nodes_mouth import LSDubDub
        os.makedirs(os.path.join(project.dir, 'final'), exist_ok=True)
        out = os.path.join(project.dir, 'final', f"{stage_key('LSDubSaveCut', {'wm': watermark}, [render.hash])}.mp4")
        if not os.path.exists(out):
            tmp = out + '.part.mp4'
            shutil.copyfile(render.cut, tmp)
            if watermark: WM.apply_inplace(tmp)
            os.replace(tmp, out)
        return io.NodeOutput(LSDubDub._video(out), out)


NODES = [LSDubVerify, LSDubScorecard, LSDubSyncQA, LSDubFixIt, LSDubSaveCut]
