#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""mouth_card.py — v12: the mouth card (lsdub annotate's per-frame card) as a toggle on the paint path.

  lsdub paint ... --out shot.mp4 --annotate [--annotate-corner auto|lower-right|lower-left|band] [--annotate-faces heads.json]
  mouth_paint.paint(video, track, out, show_mouth_card=True, ...)      # the one call a ComfyUI node wraps (docs/node-contract.md)

After the normal render, and only then, the clean output is READ (never written) and next to it go:
  shot_annotated.mp4         the clean frames + the card (lsdub annotate: the word between its neighbours, the viseme
                             glyph, openness, the next 1 s of visemes, a struck 'skip' on vocalisations; v12: weak-alignment
                             marks and 'heard:' diffs); the card is never over a face (lower-right, else lower-left, else a
                             band below the picture), audio copied
  shot_lyric_check.json/.md  every heard / lyric mismatch, weak word and vocalisation span (lsdub/core/lyric_check.py)
  shot_lyrics_suggested.txt  the lyric sheet with '[heard: x]' suggestions (nothing applied)
`lsdub paint` (lsdub/cli.py) watermarks the annotated video like the clean one. The card describes the render: the track
and the vocalisation mode it was painted with (--vocalisations on the warp path; the other modes mouth everything).
"""
import os


def annotated_path(out):
    return os.path.splitext(out)[0] + '_annotated.mp4'


def side_paths(out):
    """(annotated video, lyric check JSON, lyric check .md, suggested lyrics) next to the clean output `out`."""
    b = os.path.splitext(out)[0]
    return annotated_path(out), b + '_lyric_check.json', b + '_lyric_check.md', b + '_lyrics_suggested.txt'


def annotate_args(a):
    """lsdub annotate's argv for a parsed `lsdub paint` namespace (the clean output is its --video)."""
    ann, cj, _, sg = side_paths(a.out)
    voc = getattr(a, 'vocalisations', 'skip') if a.mode == 'warp' else 'mouth'
    argv = ['--video', a.out, '--track', a.track, '--out', ann, '--corner', getattr(a, 'annotate_corner', 'auto') or 'auto',
            '--vocalisations', voc, '--lyric-check', cj, '--lyrics-suggested', sg]
    if getattr(a, 'skip_words', None): argv += ['--skip-words', a.skip_words]
    if getattr(a, 'skip_lyrics', None): argv += ['--lyrics', a.skip_lyrics]
    if getattr(a, 'vocal_align', None): argv += ['--align', a.vocal_align]
    if getattr(a, 'vocal_merge_gap', None) is not None: argv += ['--vocal-merge-gap', str(a.vocal_merge_gap)]
    if getattr(a, 'annotate_faces', None): argv += ['--faces', a.annotate_faces]
    lib = getattr(a, 'lib', None)                     # the face's own detailed mouth-shape cards (BuildLipKit)
    if lib and os.path.isdir(os.path.join(lib, 'cards')): argv += ['--cards', os.path.join(lib, 'cards')]
    return argv


def after_paint(a):
    """paint --annotate: the card next to the clean output (which is only read). Returns the side paths that exist. The
    card is a side product: last run's side files of this output are removed first (never mistaken for this run's), and a
    failing card is reported, not raised, so the clean output still gets its watermark (lsdub/cli.py runs after this)."""
    import sys
    side = dict(zip(('annotated', 'lyric_check', 'lyric_check_md', 'lyrics_suggested'), side_paths(a.out or '')))
    if not a.out or not os.path.exists(a.out):
        print(f'annotate: no clean output at {a.out}, no card', file=sys.stderr, flush=True)
        return {}
    for p in side.values():
        if os.path.exists(p): os.remove(p)
    from ..audits import viseme_overlay
    try:
        viseme_overlay.main(annotate_args(a))
    except (Exception, SystemExit) as e:         # (SystemExit too: annotate's own argparse errors, e.g. an unreadable video)
        for p in side.values():                  # no half-written card / check is left to be watermarked or returned
            if os.path.exists(p): os.remove(p)
        print(f'annotate: the card failed ({type(e).__name__}: {e}); the clean output {a.out} is unaffected', file=sys.stderr, flush=True)
    return {k: p for k, p in side.items() if os.path.exists(p)}


def strip_after_paint(a):
    """paint --sync-strip: the scrolling vocal waveform under the render (the annotated one when --annotate made it).
    Side product like the card: a failure is reported, never raised. Audio: --sync-strip-audio at the track's t0."""
    import json, sys
    from ..audits import sync_strip
    src = annotated_path(a.out) if getattr(a, 'annotate', False) and os.path.exists(annotated_path(a.out)) else a.out
    out = sync_strip.strip_path(a.out)
    try:
        t0 = getattr(a, 'sync_strip_t0', None)
        if t0 is None: t0 = float(json.load(open(a.track)).get('t0', 0.0)) if getattr(a, 'track', None) else 0.0
        aud = getattr(a, 'sync_strip_audio', None)
        sync_strip.render(src, out, aud, t0 if aud else 0.0)
        return out
    except Exception as e:
        if os.path.exists(out): os.remove(out)
        print(f'sync strip failed ({type(e).__name__}: {e}); {a.out} is unaffected', file=sys.stderr, flush=True)
