# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Project nodes: the project folder every LSDub node writes into, and the lyrics slot."""
import os

import folder_paths
from comfy_api.v0_0_2 import io

from . import lsdub_types as T
from .lsdub_engine import presets
from .project import MODES, Project, parse_force_shots, stage_key, text_hash


def default_root():
    return os.path.join(folder_paths.get_output_directory(), 'lsdub')


class LSDubProjectNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubProject', display_name='LSDub Project', category='LSDub',
            description='The project folder for one cut. Every LSDub node saves its output here, keyed by its inputs, '
                        'so a re-run reuses what is unchanged and redoes only what changed.',
            inputs=[
                io.String.Input('name', default='my-cut', tooltip='Project name: a folder under root. One per cut.'),
                io.String.Input('root', default='', tooltip='Where projects live. Empty = ComfyUI output/lsdub.'),
                io.Combo.Input('mode', options=list(MODES), default='run',
                               tooltip='run = the full dub. test_learn = analyse the source and song, run sync QA and '
                                       'suggest fixes, without dubbing.'),
                io.Float.Input('fps', default=25.0, min=1.0, max=120.0, step=0.001,
                               tooltip='Frame rate of the delivered cut.'),
                io.Float.Input('song_offset', default=0.0, min=-600.0, max=600.0, step=0.001,
                               tooltip='Video time of the song\'s 0 s, in seconds (video = song + offset).'),
                io.Combo.Input('preset', options=sorted(presets.PRESETS), default='none',
                               tooltip='Cast preset: who sings, the hook lines and per-face kit settings. none = any '
                                       'named character can sing.'),
                io.Int.Input('seed', default=0, min=0, max=2 ** 31 - 1,
                             tooltip='Seed for the generator slots (video, still, song).'),
                io.String.Input('force_shots', default='',
                                tooltip='Shots to redo even if they passed, e.g. "21,23". Their earlier renders are kept.'),
            ],
            outputs=[T.LSDubProject.Output(display_name='project', tooltip='The project: pass it to every LSDub node.')],
        )

    @classmethod
    def build(cls, name, root, mode, fps, song_offset, preset, seed, force_shots):
        return Project(root=root or default_root(), name=name.strip(), mode=mode, fps=float(fps),
                       song_offset=float(song_offset), seed=int(seed), preset=preset,
                       force_shots=parse_force_shots(force_shots))

    @classmethod
    def fingerprint_inputs(cls, **kw):
        p = cls.build(**kw)
        return stage_key('LSDubProject', {'project': p.cache_key(), 'root': p.root, 'force': list(p.force_shots)})

    @classmethod
    def execute(cls, name, root, mode, fps, song_offset, preset, seed, force_shots):
        return io.NodeOutput(cls.build(name, root, mode, fps, song_offset, preset, seed, force_shots).ensure())


def norm_lyrics(text):
    """one form for the key and the file: LF line ends, stripped, one trailing newline."""
    return (text or '').replace('\r\n', '\n').strip() + '\n'


class LSDubLyricsSlot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubLyricsSlot', display_name='LSDub Lyrics', category='LSDub/audio',
            description='The sung words, one line per sung line. Saved in the project; alignment reads them.',
            inputs=[
                T.LSDubProject.Input('project', tooltip='The LSDub project.'),
                io.String.Input('text', multiline=True, default='',
                                tooltip='Lyrics as sung, one line per sung line. Leave vocalisations (ooh, ah) in; '
                                        'the project skips them.'),
            ],
            outputs=[T.LSDubLyrics.Output(display_name='lyrics', tooltip='The saved lyrics and their hash.')],
        )

    @classmethod
    def fingerprint_inputs(cls, project, text):
        return stage_key('LSDubLyricsSlot', {'text': norm_lyrics(text)}, [project.cache_key()])

    @classmethod
    def validate_inputs(cls, text=None, **kw):
        if text is None: return True                                    # linked: checked when it arrives
        return True if text.strip() else 'lyrics are empty'

    @classmethod
    def execute(cls, project, text):
        text = norm_lyrics(text)
        key = stage_key('LSDubLyricsSlot', {'text': text}, [project.cache_key()])
        st = project.stage('slots/lyrics', key)
        path = st.path('lyrics.txt')
        if not st.is_done():
            with open(path, 'w', encoding='utf-8') as f: f.write(text)
            st.mark_done(True, chars=len(text))
        return io.NodeOutput(T.Lyrics(path=path, hash=text_hash(text)))


NODES = [LSDubProjectNode, LSDubLyricsSlot]
