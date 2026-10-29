#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Build example_workflows/*.json (ComfyUI UI format) from the LSDub node schemas, plus same-name .jpg thumbnails.

  python scripts/make_workflows.py            (no ComfyUI needed: the schemas are read with the test stubs)

Every workflow uses preset=landing, backend=local and the demo assets from scripts/fetch_demo.py."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
import conftest  # noqa: E402  (installs the ComfyUI stubs, loads the pack)

PACK = conftest.load_pack()
WIDGET = {'STRING', 'INT', 'FLOAT', 'BOOLEAN', 'COMBO'}
LANE_TITLE = {'video': 'VIDEO', 'audio': 'AUDIO', 'mouth': 'MOUTH', 'qa': 'QA'}
GROUP_COLOUR = {'video': '#3f789e', 'audio': '#8A8', 'mouth': '#b06634', 'qa': '#88A'}
LANE_COLOUR = {'LSDub': '#335', 'LSDub/video': '#253', 'LSDub/audio': '#523', 'LSDub/mouth': '#532', 'LSDub/qa': '#255'}
CORE = {   # core nodes: (inputs [(name, type)], outputs [(name, type)], widgets builder)
    'LoadVideo': ([], [('VIDEO', 'VIDEO')], lambda w: [w['file'], 'image']),
    'LoadAudio': ([], [('AUDIO', 'AUDIO')], lambda w: [w['audio'], None, None]),
    'LoadImage': ([], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')], lambda w: [w['image'], 'image']),
    'PreviewAny': ([('source', '*')], [('STRING', 'STRING')], lambda w: [None, None, False]),
    'PreviewImage': ([('images', 'IMAGE')], [], lambda w: []),
    'PreviewAudio': ([('audio', 'AUDIO')], [], lambda w: []),
    'SaveVideo': ([('video', 'VIDEO')], [], lambda w: [w.get('filename_prefix', 'video/LSDub'), 'mp4', 'auto']),
}


def ui_order(inputs):
    """the frontend fills widgets_values in input_order: every required input, then every optional one."""
    return [p for p in inputs if not p.optional] + [p for p in inputs if p.optional]


def schema(node_id):
    for cls in (*PACK.nodes_project.NODES, *PACK.nodes_audio.NODES, *PACK.nodes_vocal.NODES, *PACK.nodes_video.NODES,
                *PACK.nodes_mouth.NODES, *PACK.nodes_qa.NODES):
        s = cls.define_schema()
        if s.node_id == node_id: return s
    raise KeyError(node_id)


class WF:
    def __init__(self, title):
        self.title, self.nodes, self.links, self.ids = title, [], [], {}

    def add(self, key, ntype, pos, **w):
        title, lane = w.pop('_title', None), w.pop('_lane', None)
        nid = len(self.nodes) + 1
        if ntype in CORE:
            ins, outs, wb = CORE[ntype]
            node = {'id': nid, 'type': ntype, 'pos': list(pos), 'size': [300, 120], 'flags': {}, 'order': nid, 'mode': 0,
                    'inputs': [{'name': n, 'type': t, 'link': None} for n, t in ins],
                    'outputs': [{'name': n, 'type': t, 'links': [], 'slot_index': i} for i, (n, t) in enumerate(outs)],
                    'properties': {'Node name for S&R': ntype}, 'widgets_values': wb(w)}
        else:
            s = schema(ntype)
            ins, wv = [], []
            for p in ui_order(s.inputs):
                t = p.io_type
                if t in WIDGET:
                    wv.append(w.get(p.id, p.kw.get('default', (p.kw.get('options') or [''])[0] if t == 'COMBO' else
                                                    {'STRING': '', 'INT': 0, 'FLOAT': 0.0, 'BOOLEAN': False}[t])))
                else:
                    ins.append({'name': p.id, 'type': t, 'link': None})
            node = {'id': nid, 'type': ntype, 'pos': list(pos), 'size': [340, 60 + 26 * (len(ins) + len(wv))], 'flags': {},
                    'order': nid, 'mode': 0, 'inputs': ins,
                    'outputs': [{'name': o.display_name, 'type': o.io_type, 'links': [], 'slot_index': i}
                                for i, o in enumerate(s.outputs)],
                    'properties': {'Node name for S&R': ntype}, 'widgets_values': wv,
                    'color': LANE_COLOUR.get(s.category, '#333'), 'title': s.display_name}
        node['_api'] = dict(w)
        if ntype not in CORE:
            names = [p.id for p in ui_order(schema(ntype).inputs) if p.io_type in WIDGET]
            node['_api'] = dict(zip(names, node['widgets_values']))
        elif ntype == 'SaveVideo':
            node['_api'] = {'filename_prefix': w.get('filename_prefix', 'video/LSDub'), 'format': 'mp4', 'format.codec': 'auto'}
        elif ntype == 'PreviewAny':
            node['_api'] = {}
        if title: node['title'] = title
        cat = schema(ntype).category if ntype not in CORE else ''
        node['_lane'] = lane or {'LSDub': 'project'}.get(cat, cat.split('/')[-1] if cat else
                                                         {'LoadVideo': 'video', 'LoadImage': 'video', 'LoadAudio': 'audio'}.get(ntype, 'qa'))
        self.nodes.append(node)
        self.ids[key] = node
        return self

    def api(self):
        """the same graph in API format (for a headless check run)."""
        g = {str(n['id']): {'class_type': n['type'], 'inputs': dict(n['_api'])} for n in self.nodes}
        for lid, a, o, b, i, t in self.links:
            dst = next(n for n in self.nodes if n['id'] == b)
            g[str(b)]['inputs'][dst['inputs'][i]['name']] = [str(a), o]
        return g

    def link(self, src, out, dst, inp):
        a, b = self.ids[src], self.ids[dst]
        o = out if isinstance(out, int) else next(i for i, x in enumerate(a['outputs']) if x['name'] == out)
        i = next(k for k, x in enumerate(b['inputs']) if x['name'] == inp)
        lid = len(self.links) + 1
        t = a['outputs'][o]['type']
        self.links.append([lid, a['id'], o, b['id'], i, t])
        a['outputs'][o]['links'].append(lid)
        b['inputs'][i]['link'] = lid
        return self

    def note(self, text):
        self.note_text = text
        return self

    def layout(self):
        """lanes as horizontal bands, left to right in the order the nodes were added; a group per lane."""
        x0, gap = 520, 40
        bands, y = {}, 0
        groups = []
        for lane in ('video', 'audio', 'mouth', 'qa'):
            row = [n for n in self.nodes if n['_lane'] == lane]
            if not row: continue
            h = max(n['size'][1] for n in row) + 80
            x = x0
            for n in row:
                n['pos'] = [x, y + 50]
                x += n['size'][0] + gap
            groups.append({'title': LANE_TITLE[lane], 'bounding': [x0 - 20, y, x - x0 + 20, h], 'color': GROUP_COLOUR[lane],
                           'font_size': 28, 'flags': {}})
            bands[lane] = y
            y += h + 60
        for n in self.nodes:
            if n['_lane'] == 'project': n['pos'] = [0, 420]
        return groups

    def json(self):
        groups = self.layout()
        nodes = [{k: v for k, v in n.items() if k not in ('_api', '_lane')} for n in self.nodes]
        if getattr(self, 'note_text', None):
            nodes.insert(0, {'id': len(self.nodes) + 1, 'type': 'MarkdownNote', 'pos': [0, 0], 'size': [480, 400], 'flags': {},
                             'order': 0, 'mode': 0, 'inputs': [], 'outputs': [], 'properties': {}, 'title': 'Start here',
                             'widgets_values': [self.note_text], 'color': '#432', 'bgcolor': '#653'})
        return {'last_node_id': len(self.nodes) + 1, 'last_link_id': len(self.links), 'nodes': nodes, 'links': self.links,
                'groups': groups, 'config': {}, 'extra': {'ds': {'scale': 0.6, 'offset': [0, 0]}, 'info': {'name': self.title}},
                'version': 0.4}


DEMO = 'lsdub_demo_'
LYRICS = 'Pleiadians bending space and time\nFour hundred light-years far away\nAlways so serious, never dance a day'


def project(wf, mode):
    return wf.add('project', 'LSDubProject', (0, 0), name=f'lsdub-demo-{mode}', mode=mode, preset='landing', song_offset=0.0,
                  backend='local')


def analysis(wf, demo_transcript=False):
    """shots, triage, lyrics, align (shared by every workflow)."""
    wf.add('shots', 'LSDubShotSplit', (800, 0)).link('project', 'project', 'shots', 'project').link('video', 'source', 'shots', 'source')
    wf.add('lyrics', 'LSDubLyricsSlot', (400, 700), text=LYRICS).link('project', 'project', 'lyrics', 'project')
    wf.add('align', 'LSDubAlign', (800, 700), **({'transcript_json': DEMO + 'landing_stt.json'} if demo_transcript else {}))
    for s, o, i in (('project', 'project', 'project'), ('song', 'song', 'song'), ('lyrics', 'lyrics', 'lyrics')): wf.link(s, o, 'align', i)
    wf.add('triage', 'LSDubTriage', (1200, 0))
    for s, o, i in (('project', 'project', 'project'), ('video', 'source', 'source'), ('shots', 'shots', 'shots'), ('song', 'song', 'song')):
        wf.link(s, o, 'triage', i)
    wf.add('triage_md', 'PreviewAny', (1600, 0), _title='Triage report', _lane='video').link('triage', 'report', 'triage_md', 'source')
    wf.add('triage_sheet', 'PreviewImage', (1600, 300), _title='Triage sheet', _lane='video').link('triage', 'sheet', 'triage_sheet', 'images')
    wf.add('lyric_md', 'PreviewAny', (1200, 700), _title='Lyric check', _lane='audio').link('align', 'lyric_check', 'lyric_md', 'source')


def byo_inputs(wf):
    wf.add('load_video', 'LoadVideo', (0, 250), file=DEMO + 'landing_clip.mp4')
    wf.add('video', 'LSDubVideoSlot', (400, 0)).link('project', 'project', 'video', 'project').link('load_video', 0, 'video', 'video')
    wf.add('load_song', 'LoadAudio', (0, 500), audio=DEMO + 'landing_song.wav')
    wf.add('load_lead', 'LoadAudio', (0, 650), audio=DEMO + 'landing_lead.wav')
    wf.add('song', 'LSDubSongSlot', (400, 450)).link('project', 'project', 'song', 'project').link('load_song', 0, 'song', 'audio')
    wf.link('load_lead', 0, 'song', 'vocal')


def dub_and_qa(wf, kit_kw, audio='source', vocalisations='skip'):
    wf.add('kit', 'LSDubBuildLipKit', (1200, 1000), **kit_kw).link('project', 'project', 'kit', 'project')
    wf.add('dub', 'LSDubDub', (1600, 700), audio=audio, vocalisations=vocalisations)
    for s, o, i in (('project', 'project', 'project'), ('video', 'source', 'source'), ('triage', 'shots', 'shots'), ('song', 'song', 'song'),
                    ('align', 'align', 'align'), ('kit', 'kit', 'kit')):
        wf.link(s, o, 'dub', i)
    wf.add('verify', 'LSDubVerify', (2000, 700))
    for s, o, i in (('project', 'project', 'project'), ('video', 'source', 'source'), ('dub', 'render', 'render'), ('triage', 'shots', 'shots'),
                    ('song', 'song', 'song'), ('align', 'align', 'align')):
        wf.link(s, o, 'verify', i)
    wf.add('verify_md', 'PreviewAny', (2400, 700), _title='Verify (sync score)', _lane='qa').link('verify', 'md', 'verify_md', 'source')
    wf.add('fixit', 'LSDubFixIt', (2000, 1000))
    for s, o, i in (('project', 'project', 'project'), ('triage', 'shots', 'shots'), ('verify', 'report', 'report'), ('dub', 'render', 'render')):
        wf.link(s, o, 'fixit', i)
    wf.add('fixit_md', 'PreviewAny', (2400, 1000), _title='Fix-it list', _lane='qa').link('fixit', 'md', 'fixit_md', 'source')
    wf.add('save', 'LSDubSaveCut', (2000, 1300)).link('project', 'project', 'save', 'project').link('dub', 'render', 'save', 'render')


def test_and_learn():
    wf = WF('LSDub 3: Test and Learn')
    wf.note('## LSDub 3: Test and Learn\n**Does:** analyses a music video with no dub: shots, which can be dubbed (Triage), word timing (Align), measured lip sync (Sync QA) and a fix-it list.\n\n**Needs:** the demo pack (`python custom_nodes/comfyui-lsdub/scripts/fetch_demo.py`), or your own clip + song. CPU is fine.\n\n**Time:** about 1-3 min on the demo (first run also downloads ~3 GB of speech-to-text weights).\n\n**Change for your video:** *Load Video* (your clip), *Load Audio* x2 (the song, and its dry lead vocal if you have it), *LSDub Lyrics*. Keep *LSDub Project* mode = test_learn.')
    project(wf, 'test_learn')
    byo_inputs(wf)
    analysis(wf, demo_transcript=True)
    wf.add('sync', 'LSDubSyncQA', (1600, 700))
    for s, o, i in (('project', 'project', 'project'), ('video', 'source', 'source'), ('triage', 'shots', 'shots'), ('song', 'song', 'song'),
                    ('align', 'align', 'align')):
        wf.link(s, o, 'sync', i)
    wf.add('sync_card', 'SaveVideo', (2000, 450), _title='Sync QA card', _lane='qa', filename_prefix='lsdub/sync_card').link('sync', 'card', 'sync_card', 'video')
    wf.add('sync_md', 'PreviewAny', (2000, 700), _title='Sync QA report', _lane='qa').link('sync', 'md', 'sync_md', 'source')
    wf.add('fixit', 'LSDubFixIt', (2000, 1000))
    for s, o, i in (('project', 'project', 'project'), ('triage', 'shots', 'shots'), ('sync', 'report', 'sync')): wf.link(s, o, 'fixit', i)
    wf.add('fixit_md', 'PreviewAny', (2400, 1000), _title='Fix-it list', _lane='qa').link('fixit', 'md', 'fixit_md', 'source')
    return wf


def byo():
    wf = WF('LSDub 2: BYO Clip and Song')
    wf.note("## LSDub 2: BYO Clip and Song\n**Does:** dubs the singer's mouth in your clip to your song, shot by shot, then grades it (Verify, Fix It) and saves the cut.\n\n**Needs:** the demo pack (`scripts/fetch_demo.py`): clip, song, lead vocal, lyrics and a lip kit. CPU is fine.\n\n**Time:** about 10-15 min on CPU for the 7.5 s demo (about 7-8 min per dubbed shot); re-runs are cache hits.\n\n**Change for your video:** *Load Video*, *Load Audio* x2, *LSDub Lyrics*, and *LSDub Lip Kit* kit_dir (or build a kit with workflow 1/4).")
    project(wf, 'run')
    byo_inputs(wf)
    analysis(wf, demo_transcript=True)
    dub_and_qa(wf, {'identity': 'red_nova', 'kit_dir': DEMO + 'kits/red_nova'})
    return wf


def prompt_everything():
    wf = WF('LSDub 1: Prompt Everything')
    wf.note('## LSDub 1: Prompt Everything\n**Does:** edits a still (Qwen-Image-Edit), animates it (LTX-2.5), generates a song (ACE-Step 1.5), builds a lip kit from the clip, dubs, grades and saves.\n\n**Needs:** a big GPU (LTX-2.5 needs ~70 GB, or set *LSDub Project* backend = comfy_cloud on a paid plan) and the generator models (README, Models).\n\n**Time:** about 10 min on a 96 GB card.\n\n**Change:** *Load Image* (your character), the *LSDub Still* and *LSDub Video* prompts, *LSDub Song* tags, *LSDub Lyrics*.')
    project(wf, 'run')
    wf.add('load_still', 'LoadImage', (0, 250), image=DEMO + 'landing_still.png')
    wf.add('still', 'LSDubStillSlot', (400, -300), edit_prompt='a close-up of the same singer, head and shoulders filling the frame, '
                                                                'on a small club stage, a deep red velvet curtain behind her, warm spotlight')
    wf.link('project', 'project', 'still', 'project').link('load_still', 0, 'still', 'image')
    wf.add('video', 'LSDubVideoSlot', (400, 0), prompt='close-up, she sings to the camera, her mouth clearly visible, gentle camera '
                                                       'push in, stage light',
           negative_prompt='blurry, distorted face, extra fingers', seconds=5.0)
    wf.link('project', 'project', 'video', 'project').link('still', 'image', 'video', 'still')
    wf.add('song', 'LSDubSongSlot', (400, 450), tags='dreamy synthpop, female vocal, 100 bpm, warm analog synths',
           seconds=6.0, bpm=100, keyscale='A minor')
    wf.link('project', 'project', 'song', 'project')
    analysis(wf)
    wf.link('lyrics', 'lyrics', 'song', 'lyrics')
    dub_and_qa(wf, {'identity': 'singer', 'kit_dir': '', 'shot_id': 0}, audio='song', vocalisations='skip')
    wf.link('video', 'source', 'kit', 'source').link('shots', 'shots', 'kit', 'shots')
    return wf


LYRICS_ALR = ("aged like rocks, still in the box\ntimeless on the shelf, winding my own clock\nit's, it's about that time\n"
              "hits, it's about that time\nthis is about that time")


def make_character_sing():
    wf = WF('LSDub 4: Make Your Character Sing')
    wf.note('## LSDub 4: Make Your Character Sing\n**Does:** a song + a character image -> a video of the character singing it: vocal split, lead vocal, alignment, an LTX-2.5 performance as long as the song, a lip kit from it, the dub, Verify (the sync score) and the saved video with the song.\n\n**Needs:** a big GPU (or backend = comfy_cloud), the generator models and the Mel-Band RoFormer vocal checkpoint in models/audio_separation (README).\n\n**Time:** about 30-40 min on a 96 GB card for the 15 s demo song (\"About That Time\", an ACE-Step song of ours).\n\n**Change:** *Load Audio* (your song), *Load Image* (your character), *LSDub Performance* prompt, *LSDub Lyrics*.')
    wf.add('project', 'LSDubProject', (0, 0), name='lsdub-character-sing', mode='run', preset='none', song_offset=0.0,
           backend='local')
    wf.add('load_song', 'LoadAudio', (0, 500), audio=DEMO + 'alr_song_15s.wav')
    wf.add('split', 'LSDubVocalSplit', (400, 500), model='MelBandRoformer_kim_vocals.ckpt').link('load_song', 0, 'split', 'audio')
    wf.add('lead', 'LSDubLeadVocal', (800, 500)).link('split', 'vocals', 'lead', 'vocals')
    wf.add('song', 'LSDubSongSlot', (1200, 500)).link('project', 'project', 'song', 'project').link('load_song', 0, 'song', 'audio')
    wf.link('lead', 'lead', 'song', 'vocal')
    wf.add('load_char', 'LoadImage', (0, 250), image=DEMO + 'landing_still.png')
    wf.add('video', 'LSDubPerformance', (400, 0), prompt='close-up, she sings to the camera on a small club stage, her mouth clearly '
                                                       'visible, warm spotlight, gentle camera push in',
           negative_prompt='blurry, distorted face, extra fingers, text, watermark', shot_seconds=5.0, chain='same_still')
    for s_, o, i in (('project', 'project', 'project'), ('load_char', 0, 'character'), ('song', 'song', 'song')): wf.link(s_, o, 'video', i)
    wf.add('shots', 'LSDubShotSplit', (800, 0)).link('project', 'project', 'shots', 'project').link('video', 'source', 'shots', 'source')
    wf.add('lyrics', 'LSDubLyricsSlot', (1200, 800), text=LYRICS_ALR).link('project', 'project', 'lyrics', 'project')
    wf.add('align', 'LSDubAlign', (1600, 800))
    for s_, o, i in (('project', 'project', 'project'), ('song', 'song', 'song'), ('lyrics', 'lyrics', 'lyrics')): wf.link(s_, o, 'align', i)
    wf.add('triage', 'LSDubTriage', (1200, 0))
    for s_, o, i in (('project', 'project', 'project'), ('video', 'source', 'source'), ('shots', 'shots', 'shots'), ('song', 'song', 'song')):
        wf.link(s_, o, 'triage', i)
    dub_and_qa(wf, {'identity': 'singer', 'kit_dir': '', 'shot_id': 0}, audio='song', vocalisations='mouth')
    wf.link('video', 'source', 'kit', 'source').link('shots', 'shots', 'kit', 'shots')
    return wf


def thumb(path, title, sub, still):
    import cv2
    import numpy as np
    W, H = 768, 432
    img = cv2.imread(still) if still and os.path.exists(still) else None
    img = cv2.resize(img, (W, H)) if img is not None else np.full((H, W, 3), 30, np.uint8)
    img = (img * 0.45).astype(np.uint8)
    cv2.rectangle(img, (0, H - 150), (W, H), (20, 20, 20), -1)
    cv2.putText(img, 'LSDub', (28, 70), cv2.FONT_HERSHEY_DUPLEX, 1.8, (90, 220, 255), 3, cv2.LINE_AA)
    cv2.putText(img, title, (28, H - 90), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (28, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    out = os.path.join(ROOT, 'example_workflows')
    os.makedirs(out, exist_ok=True)
    still = sys.argv[1] if len(sys.argv) > 1 else ''
    for name, wf, sub in (('1-LSDub-Prompt-Everything', prompt_everything(), 'still -> video, tags -> song, lip kit, dub, QA'),
                          ('2-LSDub-BYO-Clip-and-Song', byo(), 'your clip + song + lyrics + lip kit -> the dub'),
                          ('3-LSDub-Test-and-Learn', test_and_learn(), 'triage, sync QA and fix-it notes, no dub'),
                          ('4-LSDub-Make-Your-Character-Sing', make_character_sing(), 'a song + a character image -> a singing video')):
        json.dump(wf.json(), open(os.path.join(out, name + '.json'), 'w'), indent=1)
        if os.environ.get('LSDUB_API_OUT'):
            json.dump(wf.api(), open(os.path.join(os.environ['LSDUB_API_OUT'], name + '.api.json'), 'w'), indent=1)
        shot = os.environ.get('LSDUB_THUMB_' + name.split('-')[0])                 # a frame from a real run, if given
        thumb(os.path.join(out, name + '.jpg'), name.split('-', 1)[1].replace('LSDub-', '').replace('-', ' '), sub, shot or still)
        print(name, len(wf.nodes), 'nodes', len(wf.links), 'links')


if __name__ == '__main__':
    main()
