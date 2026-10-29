# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""AUDIO lane: the song slot and the lyric alignment (word timings -> viseme track, vocalisations skipped)."""
import json
import os

from comfy_api.v0_0_2 import io

from . import generate, graphs
from . import lsdub_types as T
from . import media
from .lsdub_engine._inproc import call
from .project import file_hash, stage_key, text_hash

GENERATION_LATER = ('{node}: generating from a prompt arrives in a later LSDub release. For now connect a finished '
                    '{what} to the "{port}" input.')
WHISPER_MODELS = ['large-v3', 'medium', 'small', 'base']


class LSDubSongSlot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubSongSlot', display_name='LSDub Song', category='LSDub/audio',
            description='The song to lip-sync to. Bring a finished song (any audio node), or generate one from style '
                        'tags + your lyrics with ACE-Step 1.5 XL (open weights; local GPU or Comfy Cloud per the project). '
                        'Saved in the project, keyed by its samples or its settings.',
            hidden=[io.Hidden.api_key_comfy_org],
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                io.Audio.Input('audio', optional=True, tooltip='The finished song (the mix). Takes priority over tags.'),
                io.String.Input('tags', multiline=True, default='',
                                tooltip='Style tags to generate a song (genre, mood, instruments, voice). Ignored when '
                                        'audio is connected.'),
                T.LSDubLyrics.Input('lyrics', optional=True, tooltip='Lyrics to sing in a generated song (none = instrumental).'),
                io.Float.Input('seconds', default=60.0, min=5.0, max=600.0, step=0.5, tooltip='Length of a generated song, s.'),
                io.Int.Input('bpm', default=120, min=40, max=240, optional=True, tooltip='Tempo of a generated song.'),
                io.Combo.Input('keyscale', options=graphs.KEYS, default='C major', optional=True,
                               tooltip='Key of a generated song.'),
            ],
            outputs=[T.LSDubSong.Output(display_name='song', tooltip='The song record: mix path and hash.'),
                     io.Audio.Output(display_name='audio', tooltip='The song as ComfyUI audio, for previews.')],
        )

    @classmethod
    def _gen_key(cls, project, tags, lyrics, seconds, bpm, keyscale):
        return stage_key('LSDubSongSlot/gen', {'tags': tags.strip(), 'lyrics': lyrics.hash if lyrics else '', 'seconds': seconds,
                                               'bpm': bpm, 'key': keyscale, 'seed': project.seed, 'backend': project.backend},
                         [project.cache_key()])

    @classmethod
    def fingerprint_inputs(cls, project, audio=None, tags='', lyrics=None, seconds=60.0, bpm=120, keyscale='C major'):
        if audio is None: return cls._gen_key(project, tags, lyrics, seconds, bpm, keyscale)
        return stage_key('LSDubSongSlot', {'audio': media.audio_hash(audio)}, [project.cache_key()])

    @classmethod
    async def execute(cls, project, audio=None, tags='', lyrics=None, seconds=60.0, bpm=120, keyscale='C major'):
        if audio is None:
            if not (tags or '').strip():
                raise ValueError('LSDub Song: connect a song to the "audio" input, or write style tags to generate one.')
            st = project.stage('slots/song', cls._gen_key(project, tags, lyrics, seconds, bpm, keyscale))
            mix = st.path('mix.wav')
            if not st.is_done():
                text = open(lyrics.path, encoding='utf-8').read() if lyrics else '[Instrumental]'
                g = graphs.song(tags.strip(), text, seconds, project.seed, bpm, keyscale, prefix=f'lsdub/song_{os.path.basename(st.dir)}')
                paths = await generate.run(g, project.backend, out_dir=st.path('gen'), api_key=media.hidden_api_key(cls))
                media.to_wav(paths[0], mix)
                st.mark_done(True, generated=True, backend=project.backend)
            return io.NodeOutput(T.Song(mix=mix, hash=file_hash(mix), offset=project.song_offset), media.load_audio(mix))
        h = media.audio_hash(audio)
        st = project.stage('slots/song', stage_key('LSDubSongSlot', {'audio': h}, [project.cache_key()]))
        mix = st.path('mix.wav')
        if not st.is_done():
            media.write_audio(audio, mix)
            st.mark_done(True, sample_rate=int(audio['sample_rate']))
        return io.NodeOutput(T.Song(mix=mix, hash=h, offset=project.song_offset), audio)


def lyric_check_md(align, vis_skip=None):
    """the alignment as a reviewable sheet: match ratio, the words the aligner is least sure of, the skipped spans."""
    from .lsdub_engine.core import lyric_check as LC
    A = json.load(open(align))
    rows = LC.word_table(A)
    weak = sorted(((*LC.confidence(r), r) for r in rows), key=lambda x: x[0])
    L = ['# LSDub lyric check', '',
         f"{len(rows)} words aligned; lyric match {A.get('lyric_match_ratio')}; aligner {A.get('aligner')}.", '',
         '## Least certain words', '', '| t (s) | sung | heard | lyric | timing | confidence | why |', '|---|---|---|---|---|---|---|']
    for c, why, r in weak[:20]:
        if c >= LC.WEAK_BELOW: break
        L.append(f"| {float(r['start']):.2f} | {r.get('w')} | {r.get('heard') or ''} | {r.get('lyric') or ''} | {r.get('align') or ''} | {c:.2f} | {LC.short_reason(why)} |")
    if vis_skip and os.path.exists(vis_skip):
        sk = json.load(open(vis_skip)).get('skips') or []
        L += ['', f'## Vocalisations skipped ({len(sk)})', '', '| t (s) | words |', '|---|---|']
        L += [f"| {s['t0']:.2f}-{s['t1']:.2f} | {' '.join(s.get('words') or [])} |" for s in sk]
    return '\n'.join(L) + '\n'


class LSDubAlign(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubAlign', display_name='LSDub Align', category='LSDub/audio',
            description='Time every sung word: speech-to-text, reconciled with your lyrics, then forced alignment; '
                        'builds the song\'s viseme track with vocalisations (ooh, da da) marked to skip.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project (its fps sets the viseme track rate).'),
                T.LSDubSong.Input('song', tooltip='The song. Its lead vocal is used when present, else the vocal, else the mix.'),
                T.LSDubLyrics.Input('lyrics', tooltip='The lyrics as sung.'),
                io.Audio.Input('vocal', optional=True,
                               tooltip='The lead vocal (from LSDub Lead Vocal). Best results: a dry lead vocal.'),
                io.String.Input('pronunciations', multiline=True, default='',
                                tooltip='Words the dictionary lacks, one per line: word=PH ON ES (ARPAbet), e.g. '
                                        'zorbly=Z AO R B L IY.'),
                io.Combo.Input('stt_model', options=WHISPER_MODELS, default='large-v3',
                               tooltip='Speech-to-text model size. Downloaded on first use. large-v3 is the most '
                                       'accurate; smaller is faster.'),
                io.String.Input('transcript_json', default='', optional=True,
                                tooltip='Optional: a saved speech-to-text JSON for this vocal (skips that step).'),
            ],
            outputs=[T.LSDubAlign.Output(display_name='align', tooltip='Word timings + the song viseme track.'),
                     io.String.Output(display_name='lyric_check', tooltip='Markdown: least certain words and the '
                                                                           'skipped vocalisations.')],
        )

    @classmethod
    def _vocal(cls, song, vocal):
        return (song.lead or song.vocal or song.mix) if vocal is None else None

    @classmethod
    def _key(cls, project, song, lyrics, vocal, pronunciations, stt_model, transcript_json):
        vh = media.audio_hash(vocal) if vocal is not None else (song.hashes.get('lead') or song.hash or file_hash(cls._vocal(song, None)))
        tj = file_hash(transcript_json) if transcript_json else ''
        return stage_key('LSDubAlign', {'vocal': vh, 'lyrics': lyrics.hash, 'pron': text_hash(pronunciations.strip()),
                                        'stt': stt_model, 'transcript': tj, 'which': 'given' if vocal is not None else
                                        ('lead' if song.lead else 'vocal' if song.vocal else 'mix')}, [project.cache_key()])

    @classmethod
    def fingerprint_inputs(cls, project, song, lyrics, vocal=None, pronunciations='', stt_model='large-v3', transcript_json=''):
        return cls._key(project, song, lyrics, vocal, pronunciations, stt_model, transcript_json)

    @classmethod
    def execute(cls, project, song, lyrics, vocal=None, pronunciations='', stt_model='large-v3', transcript_json=''):
        from .lsdub_engine.core import lyric_align as LA, viseme_track as VT, vocalisations as VO
        st = project.stage('audio/align', cls._key(project, song, lyrics, vocal, pronunciations, stt_model, transcript_json))
        align, vis, vis_skip, md = st.path('align.json'), st.path('visemes.json'), st.path('visemes_skip.json'), st.path('lyric_check.md')
        if not st.is_done():
            voc = media.write_audio(vocal, st.path('vocal.wav')) if vocal is not None else cls._vocal(song, None)
            dur = media.duration(voc)
            stt = st.path('stt.json')
            if transcript_json and not os.path.exists(stt):
                with open(transcript_json) as f, open(stt, 'w') as g: g.write(f.read())
            from . import devices
            wd, wi = devices.whisper_device()
            argv = ['--audio', voc, '--lyrics', lyrics.path, '--t0', '0', '--t1', f'{dur:.3f}', '--out', align,
                    '--stt-json', stt, '--aligner', 'sphinx', '--whisper-python', '', '--whisper-model', stt_model,
                    '--device', wd, '--device-index', str(wi), '--whisper-dir', devices.models_dir('whisper')]
            for ln in pronunciations.splitlines():
                if '=' in ln: argv += ['--extra-dict', ln.strip()]
            call(LA.main, argv, quiet=True, name='align')
            devices.free()
            call(VT.main, ['--align', align, '--t0', '0', '--frames', str(int(dur * project.fps)), '--fps', str(project.fps),
                           '--audio', voc, '--out', vis], quiet=True, name='visemes')
            call(VO.main, ['skip', '--track', vis, '--out', vis_skip, '--lyrics', lyrics.path, '--align', align], quiet=True,
                 name='vocalisations')
            with open(md, 'w') as f: f.write(lyric_check_md(align, vis_skip))
            st.mark_done(True, vocal=voc, seconds=round(dur, 3))
        return io.NodeOutput(T.Align(align=align, hash=file_hash(align), visemes=vis_skip), open(md).read())


NODES = [LSDubSongSlot, LSDubAlign]
