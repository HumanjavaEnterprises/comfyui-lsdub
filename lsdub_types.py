# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""lsdub_types.py: the LSDub socket types. Between LSDub nodes travel small records (a JSON manifest + paths in the
project folder), never tensors; ComfyUI's own VIDEO / AUDIO / IMAGE appear only at the edges (slot inputs, previews,
the final mux). Every record has a `hash` that downstream cache keys fold in, and round-trips through JSON."""
from dataclasses import asdict, dataclass, field, fields

from comfy_api.v0_0_2 import io

from .project import Project, canonical, text_hash


class _Record:
    """JSON round-trip + a content hash for the cache key."""

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def cache_key(self):
        return self.hash or text_hash(canonical(self.to_json()))


@dataclass
class Source(_Record):
    """LSDUB_SOURCE: the picture to dub (a video, or a still)."""
    path: str
    hash: str = ''
    fps: float = 25.0
    n_frames: int = 0
    wh: tuple = (0, 0)
    still: bool = False


@dataclass
class Song(_Record):
    """LSDUB_SONG: the mix and what the AUDIO lane derives from it."""
    mix: str
    hash: str = ''
    vocal: str = ''
    inst: str = ''
    lead: str = ''
    lead_activity: str = ''
    offset: float = 0.0
    hashes: dict = field(default_factory=dict)


@dataclass
class Lyrics(_Record):
    """LSDUB_LYRICS: the sung words (a text file in the project)."""
    path: str
    hash: str = ''


@dataclass
class Align(_Record):
    """LSDUB_ALIGN: word / phone timings and the viseme track."""
    align: str
    hash: str = ''
    visemes: str = ''


@dataclass
class Shots(_Record):
    """LSDUB_SHOTS: per shot {id, f0, f1, px_face, tier CU/MED/WIDE, verdict, reasons, singer} + a contact sheet."""
    shots: list = field(default_factory=list)
    hash: str = ''
    sheet: str = ''


@dataclass
class Cast(_Record):
    """LSDUB_CAST: who is in each shot (cast db, per-shot identity) and the dub plan."""
    db: str = ''
    scan: str = ''
    plan: str = ''
    hash: str = ''


@dataclass
class Kit(_Record):
    """LSDUB_KIT: one singer's lip kit (visemes, teeth, jaw, mouth cards / atlas) and its base frame."""
    dir: str
    identity: str = ''
    base_frame: str = ''
    hash: str = ''


@dataclass
class Masks(_Record):
    """LSDUB_MASKS: per-shot occluder masks (a mic, a hand) the dub must not paint over."""
    dir: str
    shots: list = field(default_factory=list)
    hash: str = ''


@dataclass
class Tweaks(_Record):
    """LSDUB_TWEAKS: the curated mouth controls + a raw settings string."""
    flags: dict = field(default_factory=dict)
    settings: str = ''
    hash: str = ''


@dataclass
class Render(_Record):
    """LSDUB_RENDER: per-shot renders + gate JSONs, and the assembled cut."""
    shots: dict = field(default_factory=dict)
    cut: str = ''
    hash: str = ''


@dataclass
class Report(_Record):
    """LSDUB_REPORT: verify / scorecard / sync-QA results, PASS / REVIEW / FAIL per shot, the shots to fix."""
    json: str = ''
    md: str = ''
    verdicts: dict = field(default_factory=dict)
    fix: list = field(default_factory=list)
    hash: str = ''


# socket types (io_type names are what the graph shows on a wire)
LSDubProject = io.Custom('LSDUB_PROJECT')
LSDubSource = io.Custom('LSDUB_SOURCE')
LSDubSong = io.Custom('LSDUB_SONG')
LSDubLyrics = io.Custom('LSDUB_LYRICS')
LSDubAlign = io.Custom('LSDUB_ALIGN')
LSDubShots = io.Custom('LSDUB_SHOTS')
LSDubCast = io.Custom('LSDUB_CAST')
LSDubKit = io.Custom('LSDUB_KIT')
LSDubMasks = io.Custom('LSDUB_MASKS')
LSDubTweaks = io.Custom('LSDUB_TWEAKS')
LSDubRender = io.Custom('LSDUB_RENDER')
LSDubReport = io.Custom('LSDUB_REPORT')

# io_type -> payload class
PAYLOADS = {'LSDUB_PROJECT': Project, 'LSDUB_SOURCE': Source, 'LSDUB_SONG': Song, 'LSDUB_LYRICS': Lyrics,
            'LSDUB_ALIGN': Align, 'LSDUB_SHOTS': Shots, 'LSDUB_CAST': Cast, 'LSDUB_KIT': Kit, 'LSDUB_MASKS': Masks,
            'LSDUB_TWEAKS': Tweaks, 'LSDUB_RENDER': Render, 'LSDUB_REPORT': Report}
