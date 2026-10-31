# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""QA: verify (likely success per dubbed shot), scorecard (every shot, 0-100), sync QA (measured mouth vs vocal, a
review card), mouth ruler (take -> render, per dubbed shot), fix-it (each problem -> the tweak to try + the force_shots to re-run) and save the cut."""
import json
import os
import shutil

from comfy_api.v0_0_2 import io

from . import lsdub_types as T
from . import media
from .project import file_hash, parse_force_shots, stage_key


def _vocal(song):
    return song.lead or song.mix


def _cuts(shots, only=()):
    return {'cut': [{'start': s['start'], 'dur': round(s['end'] - s['start'], 4), 'shot': s['shot']}
                    for s in shots if not only or s['shot'] in only]}


def _cut_video(project, source, render):
    """what to judge: the dub, or in test_learn the source."""
    return render.cut if render is not None and render.cut else source.path


def _left_or_warp(v):
    """what Verify needs from a render shot: `used`, why it was left, the warp log of the render the cut carries."""
    if v.get('used') == 'source':
        return {'used': 'source', 'left_reason': v.get('reason') or ''}
    g = v.get('gate')
    return {'warp_json': json.load(open(g)).get('warp_json', '')} if g and os.path.exists(g) else {}


class LSDubVerify(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubVerify', display_name='LSDub Verify', category='LSDub/qa',
            description='How likely each dubbed shot is to have worked: a 0-100 score and PASS / REVIEW / FAIL, compared '
                        'with the source (sync, mouth travel, silences, identity drift). A shot the dub left as source reads '
                        'N/A. Frames in vocalisation spans are left out of the sync score.',
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
        from . import devices as _dev
        _dev.ensure_face_models()
        from .lsdub_engine.audits import qa
        st = project.stage('qa/verify', cls._key(project, source, render, shots, song, align))
        out = st.path('verify')
        if not st.is_done():
            rendered = render.shots or {}
            sj = media.dump({'fps': source.fps, 'offset': project.song_offset,
                             'shots': [{**s, **_left_or_warp(rendered.get(str(s['shot']), {}))} for s in shots.shots]}, st.path('shots.json'))
            from . import devices
            emb = devices.embedder()
            try:
                meta, rows = qa.verify(_cut_video(project, source, render), _vocal(song), sj, before=source.path, align=align.align,
                                       work=st.path('work'), emb=emb)
            finally:
                devices.release(emb)
            qa.write_verify(meta, rows, out)
            st.mark_done(True)
        J = json.load(open(out + '.json'))
        verdicts = {str(r['shot']): r['band'] for r in J['shots']}
        return io.NodeOutput(T.Report(json=out + '.json', md=out + '.md', verdicts=verdicts, hash=file_hash(out + '.json')),
                             open(out + '.md').read(), ui=media.ui_outputs(files=(out + '.md', out + '.json')))


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
                io.Combo.Input('sharpness_mode', options=['floor', 'plate'], default='floor', optional=True,
                               tooltip='floor = a shot scores for being at least as sharp as the cut\'s median; plate = for '
                                       'matching the source video, so over-sharpening costs as much as blur.'),
            ],
            outputs=[T.LSDubReport.Output(display_name='report', tooltip='Per-shot class and score.'),
                     io.String.Output(display_name='md', tooltip='Markdown scorecard.')],
        )

    @classmethod
    def _key(cls, project, source, shots, song, align, render, sharpness_mode='floor'):
        extra = {} if sharpness_mode == 'floor' else {'sharpness_mode': sharpness_mode}   # floor = the keys before the option
        return stage_key('LSDubScorecard', extra, [project.cache_key(), source.hash, shots.hash, song.cache_key(), align.hash,
                                                render.hash if render else ''])

    @classmethod
    def fingerprint_inputs(cls, project, source, shots, song, align, render=None, sharpness_mode='floor'):
        return cls._key(project, source, shots, song, align, render, sharpness_mode)

    @classmethod
    def execute(cls, project, source, shots, song, align, render=None, sharpness_mode='floor'):
        from . import devices as _dev
        _dev.ensure_face_models()
        from .lsdub_engine.audits import scorecard as SC
        st = project.stage('qa/scorecard', cls._key(project, source, shots, song, align, render, sharpness_mode))
        out = st.path('scorecard')
        video = _cut_video(project, source, render)
        if not st.is_done():
            plate = {} if sharpness_mode == 'floor' else {'sharpness_mode': sharpness_mode, 'plate': source.path}
            rows, thumbs = SC.score(video, _cuts(shots.shots), _vocal(song), project.song_offset, align.align,
                                    work=st.path('work'), preset=project.preset, **plate)
            SC.write(rows, thumbs, out, video)
            st.mark_done(True)
        J = json.load(open(out + '.json'))
        verdicts = {str(r.get('shot', r['cut'])): r.get('class') for r in J['cuts']}
        return io.NodeOutput(T.Report(json=out + '.json', md=out + '.md', verdicts=verdicts, hash=file_hash(out + '.json')),
                             open(out + '.md').read(), ui=media.ui_outputs(files=(out + '.md', out + '.json')))


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
        from . import devices as _dev
        _dev.ensure_face_models()
        from .lsdub_engine.audits import sync_qa as Q, sync_strip as SS
        from .lsdub_engine.core.steps import call
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
        shown = card if os.path.exists(card) else media.placeholder_video(st.path('none.mp4'), 'no sung shot to show')
        return io.NodeOutput(T.Report(json=js, md=md, verdicts=verdicts, hash=file_hash(js)), LSDubDub._video(shown), open(md).read(),
                             ui=media.ui_outputs(videos=(shown,), files=(md, js)))


class LSDubMouthRuler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubMouthRuler', display_name='LSDub Mouth Ruler', category='LSDub/qa',
            description='Measure the mouth itself, take against render, per dubbed shot: inner mouth colour and darkness, teeth '
                        'brightness, inner-lip edge, boil on held mouths, grain, sharpness by stage. Reported only: it sets no band.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                T.LSDubRender.Input('render', tooltip='The dub to measure.'),
                T.LSDubSource.Input('source', tooltip='The video before the dub (the take of each shot).'),
            ],
            outputs=[T.LSDubReport.Output(display_name='report', tooltip='Per-shot rulers and notes.'),
                     io.String.Output(display_name='md', tooltip='Markdown: take -> render per shot.')],
        )

    @classmethod
    def _key(cls, project, render, source):
        return stage_key('LSDubMouthRuler', {}, [project.cache_key(), source.hash, render.hash])

    @classmethod
    def fingerprint_inputs(cls, project, render, source):
        return cls._key(project, render, source)

    @classmethod
    def _row(cls, source, sid, v, st, R):
        from .nodes_mouth import extract_shot
        if v.get('used') != 'render': return {'shot': int(sid), 'note': 'left as source'}
        d = os.path.dirname(v.get('gate') or v['render'])
        take = os.path.join(d, 'take.mp4')
        if not os.path.exists(take): take = extract_shot(source.path, v['f0'], v['f1'], st.path(f'take_{sid}.mp4'))
        stage_items = [(n, p, 0, None, None) for n, p in (('take', take), ('crop', os.path.join(d, 'work', 'crop.mp4')),
                                                          ('sung', os.path.join(d, 'sung.mp4')), ('render', v['render']))]
        return {'shot': int(sid), **R.shot_row(take, v['render'], [i for i in stage_items if os.path.exists(i[1])], st.path('work'))}

    @classmethod
    def execute(cls, project, render, source):
        from . import devices as _dev
        _dev.ensure_face_models()
        from .lsdub_engine.audits import mouth_ruler as R
        st = project.stage('qa/ruler', cls._key(project, render, source))
        js, md = st.path('ruler.json'), st.path('ruler.md')
        if not st.is_done():
            rows = [cls._row(source, sid, v, st, R) for sid, v in sorted((render.shots or {}).items(), key=lambda kv: int(kv[0]))]
            media.dump({'shots': rows}, js)
            with open(md, 'w') as f: f.write(R.rows_md(rows))
            st.mark_done(True)
        rows = json.load(open(js))['shots']
        verdicts = {str(r['shot']): ', '.join(r.get('notes', [])) or r.get('note', 'measured') for r in rows}
        return io.NodeOutput(T.Report(json=js, md=md, verdicts=verdicts, hash=file_hash(js)), open(md).read(),
                             ui=media.ui_outputs(files=(md, js)))


# reason (substring) -> what to try
FIXES = [
    ('vocalisation', 'Much of this shot is sung sounds that are not words, so few frames were scored and the sync levers will not '
                     'help. Set vocalisations to mouth in LSDub Dub to sing them, or judge the shot by eye.'),
    ('no kit for target', 'This singer has no lip kit: add one to LSDub Dub (kit_2, kit_3) with the identity the triage named.'),
    ('wrong person', 'Another face leads this shot: check the cast references, or accept the shot as is.'),
    ('target on screen only', 'The singer is on screen for too little of the shot: leave it, or trim the shot.'),
    ('no face', 'No face to dub in this shot: leave it.'),
    ('no singer on screen', 'No singer on screen here: an editorial choice, nothing to dub.'),
    ('no voice', 'No vocal under this shot: nothing to judge.'),
    ('too small', 'Use a closer framing or a higher-resolution source; mediums dub through a crop (DUB-MEDIUM >= 120 px @1080).'),
    ('profile', 'The face is turned away: pick another take or accept the shot as is.'),
    ('mouth hidden', 'Mask what covers the mouth with LSDub Occluders (phrase "microphone" / "hand") and feed the masks to LSDub Dub.'),
    ('sync gate failed', 'The dub missed the vocal: check the lyric check for mis-timed words, then try lead 0.03-0.10 or word_close 0.08 in LSDub Mouth Tweaks.'),
    ('dub error', 'The dub stopped with an error (shown): fix the cause (kit, cast, missing file), then force the shot.'),
    ('mic', 'Mask the microphone with LSDub Occluders and connect its masks to LSDub Dub (the new masks re-dub the shot).'),
    ('no singing', 'No vocal under this shot: leave it.'),
    ('too short', 'Shorter than the minimum shot: leave it.'),
    ('identity drift', 'The face changed: lower face_target or turn auto off in LSDub Mouth Tweaks (a tweak change re-dubs the shot).'),
    ('mouth off the vocal', 'Sync is off: try lead 0.03-0.10 in LSDub Mouth Tweaks (check the words in the lyric check first).'),
    ('open in the silences', 'Mouth open in the gaps: raise word_close (0.08-0.12) in LSDub Mouth Tweaks.'),
    ('soft attacks', 'Attacks are soft: keep snap on and lower lead a little.'),
    ('barely moves', 'Held gape or mumble: raise word_close, and check the vocal is the dry lead.'),
    ('temporal pops', 'Run LSDub Continuity on this shot with a lower z (3.0).'),
    ('off-model', 'The mouth does not match the face: rebuild or swap the lip kit for this singer.'),
    ('near-still', 'The shot barely moves: a still playing as video. Use a take with motion.'),
    ('soft image', 'The shot is soft: use a sharper source, or raise face_target in LSDub Mouth Tweaks.'),
    ('boil', 'The mouth shimmers while held: keep the settings at the recipe, or set mouth_stack to layers in LSDub Project.'),
    ('cavity void', 'The inner mouth went black: try mouth_stack layers in LSDub Project, or rebuild the lip kit.'),
    ('dim teeth', 'The teeth read darker than the take: rebuild the lip kit from a frame with lit teeth.'),
    ('over-sharp', 'The lip edge is harder than the take: lower face_target in LSDub Mouth Tweaks.'),
    ('grain lost', 'The mouth is cleaner than the skin around it: check the cut at full size before anything else.'),
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
                T.LSDubRender.Input('render', optional=True, tooltip='The dub: its own per-shot pass / fail and why.'),
                io.Float.Input('min_corr', default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip='A sung shot measuring below this sync corr is listed.'),
                T.LSDubReport.Input('scorecard', optional=True, tooltip='A Scorecard report: its low-scoring shots are listed.'),
                T.LSDubReport.Input('ruler', optional=True, tooltip='A Mouth Ruler report: its notes are listed, with what to try.'),
            ],
            outputs=[io.String.Output(display_name='md', tooltip='Markdown to-do: shot, problem, what to try.'),
                     io.String.Output(display_name='force_shots', tooltip='Paste into LSDub Project > force_shots.')],
        )

    @classmethod
    def execute(cls, project, shots, report=None, sync=None, min_corr=0.5, render=None, scorecard=None, ruler=None):
        items = {}
        for s in shots.shots:
            if s.get('verdict') == 'NO' and s.get('voiced') and s.get('target'):
                items.setdefault(s['shot'], []).append(('triage: ' + s['reason'], fix_for(s['reason']), False))
            elif s.get('verdict') in ('DUB', 'DUB-MEDIUM') and (s.get('mic') or 0) >= 0.10:   # the triage's mic line
                items.setdefault(s['shot'], []).append(('triage: mic near the mouth', fix_for('mic'), False))
        if report is not None and report.json and os.path.exists(report.json):
            for r in json.load(open(report.json)).get('shots', []):
                if r.get('band') in ('FAIL', 'REVIEW'):
                    items.setdefault(r['shot'], []).append((f"verify {r['band']} {r.get('score')}: {r.get('reason')}", fix_for(r.get('reason')), True))
                elif r.get('band') == 'N/A':                                                  # informational: nothing to redo from here
                    items.setdefault(r['shot'], []).append((f"verify N/A: {r.get('reason')}", fix_for(r.get('reason')), False))
        if render is not None:
            for sid, v in (render.shots or {}).items():
                if not v.get('pass'):
                    why = v.get('reason') or 'failed'
                    items.setdefault(int(sid), []).append((f"dub: {why} (cut uses the {v.get('used', 'source')})", fix_for(why),
                                                          'dub error' in why))
        if sync is not None:
            for sid, c in (sync.verdicts or {}).items():
                if c is not None and str(sid).isdigit() and float(c) < min_corr:
                    items.setdefault(int(sid), []).append((f'sync corr {float(c):.2f} < {min_corr}', fix_for('mouth off the vocal'), True))
        if scorecard is not None and scorecard.json and os.path.exists(scorecard.json):
            from .lsdub_engine.audits.scorecard import AMBER
            for r in json.load(open(scorecard.json)).get('cuts', []):
                if r.get('score') is not None and r['score'] < AMBER and r.get('shot') is not None:
                    items.setdefault(r['shot'], []).append((f"scorecard {r['score']:.0f}: {r.get('reason')}", fix_for(r.get('reason')), False))
        if ruler is not None and ruler.json and os.path.exists(ruler.json):
            for r in json.load(open(ruler.json)).get('shots', []):
                for n in r.get('notes', []):                                                  # reported: no shot joins the redo list for it
                    items.setdefault(r['shot'], []).append((f'mouth ruler: {n}', fix_for(n), False))
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
                io.Int.Input('quality', default=18, min=10, max=30, optional=True,
                             tooltip='Quality of the watermarked file (x264 CRF): lower = better and bigger.'),
            ],
            outputs=[io.Video.Output(display_name='video', tooltip='The saved cut.'),
                     io.String.Output(display_name='path', tooltip='Where it was saved.')],
        )

    @classmethod
    def execute(cls, project, render, watermark=True, quality=18):
        from .lsdub_engine import watermark as WM
        from .nodes_mouth import LSDubDub
        os.makedirs(os.path.join(project.dir, 'final'), exist_ok=True)
        out = os.path.join(project.dir, 'final', f"{stage_key('LSDubSaveCut', {'wm': watermark, 'q': quality}, [render.hash])}.mp4")
        if not os.path.exists(out):
            tmp = out + '.part.mp4'
            shutil.copyfile(render.cut, tmp)
            if watermark: WM.apply_inplace(tmp, crf=int(quality))
            os.replace(tmp, out)
        return io.NodeOutput(LSDubDub._video(out), out, ui=media.ui_outputs(videos=(out,)))


NODES = [LSDubVerify, LSDubScorecard, LSDubSyncQA, LSDubMouthRuler, LSDubFixIt, LSDubSaveCut]
