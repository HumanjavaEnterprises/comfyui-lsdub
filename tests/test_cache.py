# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Keep everything that renders: same inputs -> same key folder (a cache hit); per-shot invalidation; force_shots
re-opens just those shots; a failed shot is redone; nothing is deleted."""
import os


def _proj(pack, tmp_path, **kw):
    return pack.project.Project(root=str(tmp_path), name='cut', **kw).ensure()


def test_same_inputs_same_key(pack):
    K = pack.project.stage_key
    a = K('LSDubAlign', {'coart': True, 'x': 1.0}, ['h1', 'h2'])
    assert a == K('LSDubAlign', {'x': 1.0000000001, 'coart': True}, ['h1', 'h2'])        # order / float noise blind
    assert a != K('LSDubAlign', {'coart': False, 'x': 1.0}, ['h1', 'h2'])
    assert a != K('LSDubAlign', {'coart': True, 'x': 1.0}, ['h1', 'h3'])                  # an upstream change
    assert a != K('LSDubAlign', {'coart': True, 'x': 1.0}, ['h1', 'h2'], engine_version='9.9')
    assert len(a) == 16


def test_project_folder(pack, tmp_path):
    p = _proj(pack, tmp_path)
    assert os.path.exists(os.path.join(p.dir, 'project.json'))
    for s in ('slots', 'audio', 'video', 'mouth', 'shots', 'qa', 'final', 'logs'): assert os.path.isdir(os.path.join(p.dir, s))


def test_stage_done_roundtrip(pack, tmp_path):
    p = _proj(pack, tmp_path)
    st = p.stage('audio/align', 'k1')
    assert st.done() is None and not st.is_done()
    open(st.path('align.json'), 'w').write('{}')
    st.mark_done(True, corr=0.9)
    assert st.is_done() and st.done()['corr'] == 0.9
    assert p.stage('audio/align', 'k2').done() is None                                    # a new key = a new folder


def test_per_shot_skip_failed_and_force(pack, tmp_path):
    P = pack.project
    p = _proj(pack, tmp_path)
    for s, ok in ((1, True), (2, False), (3, True)):
        st = p.shot(s, 'kd')
        open(st.path('render.mp4'), 'w').write('x')
        st.mark_done(ok)
    assert P.shots_todo(p, 'kd', [1, 2, 3, 4]) == [2, 4]                                 # failed + never-run
    pf = P.Project(root=str(tmp_path), name='cut', force_shots=P.parse_force_shots('3'))
    assert P.shots_todo(pf, 'kd', [1, 2, 3, 4]) == [2, 3, 4]
    d3 = pf.shot(3, 'kd').dir
    assert os.path.exists(os.path.join(d3, 'render.mp4')) and os.path.exists(os.path.join(d3, 'done.1.json'))
    assert P.shots_todo(p, 'kd', [1]) == []                                              # shot 1 untouched


def test_parse_force_shots(pack):
    f = pack.project.parse_force_shots
    assert f('21,23') == (21, 23) and f(' 23 21 ') == (21, 23) and f('') == () and f([5, 5]) == (5,)


def test_lyrics_slot_cache(pack, tmp_path):
    N = pack.nodes_project
    p = _proj(pack, tmp_path)
    a = N.LSDubLyricsSlot.execute(p, 'la la\nfly away').args[0]
    b = N.LSDubLyricsSlot.execute(p, 'la la\r\nfly away\n').args[0]
    assert a.path == b.path and a.hash == b.hash and open(a.path).read() == 'la la\nfly away\n'
    assert N.LSDubLyricsSlot.fingerprint_inputs(p, 'x') != N.LSDubLyricsSlot.fingerprint_inputs(p, 'y')
    c = N.LSDubLyricsSlot.execute(p, 'la la\nfly home').args[0]
    assert c.path != a.path and os.path.exists(a.path)                                  # old take kept


def test_project_node(pack, tmp_path):
    N = pack.nodes_project
    kw = dict(name='cut', root=str(tmp_path), mode='test_learn', fps=25.0, song_offset=12.0, preset='landing', seed=1,
              force_shots='21, 23')
    p = N.LSDubProjectNode.execute(**kw).args[0]
    assert p.force_shots == (21, 23) and p.mode == 'test_learn' and os.path.isdir(p.dir)
    assert N.LSDubProjectNode.fingerprint_inputs(**kw) == N.LSDubProjectNode.fingerprint_inputs(**kw)
    assert N.LSDubProjectNode.fingerprint_inputs(**{**kw, 'force_shots': ''}) != N.LSDubProjectNode.fingerprint_inputs(**kw)
    d = N.LSDubProjectNode.execute(**{**kw, 'root': ''}).args[0]
    assert d.root.endswith(os.path.join('output', 'lsdub'))


def test_fingerprint_before_inputs_exist_is_quiet(pack):
    for cls in (pack.nodes_audio.LSDubAlign, pack.nodes_video.LSDubTriage, pack.nodes_mouth.LSDubDub, pack.nodes_qa.LSDubVerify):
        assert cls.fingerprint_inputs(project=None, song=None) == 'lsdub:pending'
