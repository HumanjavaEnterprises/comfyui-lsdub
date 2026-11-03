# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""spaces.py: the frames a paint sees, and an occluder mask that knows which frames it was made on.

  PaintSpace    how frame j of a paint relates to the take; scale() = paint px per take px
    NativeSpace the take itself: n frames, identity, scale 1
    CropSpace   medium's face crop: take frame k is a square of S take px around (cx_k, cy_k), zoomed by z to O x O px
                (medium.crop_matrix), and written m times, so paint frame j shows take frame j // m
  Occluder      one masks npz (`lsdub occluders`) and the space it was made in. for_space(space, grow) gives the mask
                and its grow in that space's pixels, mapped through the geometry; a mask is never resized to fit.

An occluder npz carries no record of the frames it was made on, so `Occluder.open` tells the space from the shape: `lsdub sing`
paints work/crop.mp4, and an LSDub Occluders mask made on it has one square frame per crop frame, (n * m, O, O). Any other
shape is refused before a paint runs. Once medium has run, `bind` checks the side against the crop's O and fixes the geometry.

A take whose frames are square, painted at m = 1 with a side equal to O, gives masks of the same shape in both spaces. The
shape reads as the crop: both mask makers that feed `sing` work on the crop. Only the side check can catch the other case.

A crop mask mapped back to the take is empty outside the crop square: the crop never saw those pixels.
"""

import cv2, numpy as np

from .medium import crop_matrix

GROW_DEFAULT = 3  # paint's --occluder-grow default


class PaintSpace:
    """n frames of (h, w) px."""

    def __init__(self, n, shape):
        self.n, self.shape = int(n), (int(shape[0]), int(shape[1]))

    def take_frame(self, j):
        raise NotImplementedError

    def scale(self):
        raise NotImplementedError


class NativeSpace(PaintSpace):
    """the take as it is."""

    def take_frame(self, j):
        return int(j)

    def scale(self):
        return 1.0


class CropSpace(PaintSpace):
    """centres: (cx, cy) per take frame, in take px; S: crop side in take px; O: crop side in crop px; m: crop frames per take frame.
    zoom: crop px per take px (medium's reported value; O / S when not given)."""

    def __init__(self, centres, S, O, m=1, zoom=None):
        self.centres = [(float(c[0]), float(c[1])) for c in centres]
        self.S, self.O, self.m = float(S), int(O), max(1, int(m))
        self.z = self.O / self.S
        self.zoom = float(zoom) if zoom else self.z
        super().__init__(len(self.centres) * self.m, (self.O, self.O))

    @classmethod
    def from_report(cls, R, m):
        """from medium.run's report: per_frame centre, crop_src_px, crop_px, zoom."""
        return cls([f["centre"] for f in R["per_frame"]], R["crop_src_px"], R["crop_px"], m, R.get("zoom"))

    def take_frame(self, j):
        return int(j) // self.m

    def scale(self):
        return self.zoom

    def matrix(self, k):
        """take px -> crop px for take frame k."""
        cx, cy = self.centres[k]
        return crop_matrix(cx, cy, self.S, self.z)

    def to_take(self, img, k, shape):
        """a crop-space image of take frame k -> take px (h, w); zero outside the crop square."""
        Ai = cv2.invertAffineTransform(self.matrix(k))
        return cv2.warpAffine(
            img, Ai, (int(shape[1]), int(shape[0])), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )


def occluder_grow(args):
    """--occluder-grow in these paint flags (the last one wins, as paint's parser reads it; `--x v` and `--x=v`); the default when absent."""
    a = [str(x) for x in args or ()]
    v = GROW_DEFAULT
    for i, x in enumerate(a):
        if x == "--occluder-grow" and i + 1 < len(a):
            v = int(a[i + 1])
        elif x.startswith("--occluder-grow="):
            v = int(x.split("=", 1)[1])
    return v


def scaled_grow(grow, frm, to):
    """a grow in frm's px -> the same reach in to's px, whole px, never below 0."""
    return max(0, int(round(float(grow) * to.scale() / frm.scale())))


class Occluder:
    """one occluder npz in one paint space."""

    def __init__(self, path, frames, side, space=None):
        self.path, self.frames, self.side, self.space = path, int(frames), int(side), space

    @staticmethod
    def shape(path):
        """the masks array's shape."""
        with np.load(path) as z:
            return tuple(int(v) for v in z["masks"].shape)

    @classmethod
    def open(cls, path, frames, how=""):
        """a mask made on `frames` square frames of the crop paint: (frames, X, X). Anything else raises ValueError; how: the
        line that makes a fitting one."""
        s = cls.shape(path)
        if len(s) != 3 or s[0] != int(frames) or s[1] != s[2]:
            raise ValueError(
                f"occluder mask {path} has shape {s}; it must hold {int(frames)} square frames of the face crop, "
                f"({int(frames)}, X, X), made on work/crop.mp4" + (f": {how}" if how else "")
            )
        return cls(path, frames, s[1])

    def bind(self, space):
        """this mask in the crop space medium built: the side must be the crop's O and the frames its n * m."""
        if self.side != space.O or self.frames != space.n:
            raise ValueError(
                f"occluder mask {self.path} is {self.frames} frames of {self.side} px; the face crop is {space.n} frames of {space.O} px. "
                "Make the mask again on this shot's work/crop.mp4"
            )
        return Occluder(self.path, self.frames, self.side, space)

    def for_space(self, space, grow=GROW_DEFAULT, out=None):
        """(path, grow) for a paint in `space`. The bound crop space: this file, unchanged. A NativeSpace of the same take: crop frame
        k * m of each take frame mapped back to take px, written to `out` (masks (n, h, w) uint8), grow in take px."""
        if space is self.space or (isinstance(space, CropSpace) and isinstance(self.space, CropSpace) and vars(space) == vars(self.space)):
            return self.path, int(grow)
        if not (isinstance(space, NativeSpace) and isinstance(self.space, CropSpace)):
            raise ValueError(f"occluder mask {self.path}: no mapping from {type(self.space).__name__} to {type(space).__name__}")
        c = self.space
        if space.n * c.m != self.frames:
            raise ValueError(f"occluder mask {self.path}: {self.frames} crop frames do not cover {space.n} take frames at x{c.m}")
        with np.load(self.path) as z:
            M = z["masks"]
            extra = {k: z[k] for k in z.files if k != "masks"}
        masks = np.stack([c.to_take(M[k * c.m], k, space.shape) for k in range(space.n)]).astype(np.uint8)
        if "coverage" in extra:
            cov = np.asarray(extra["coverage"])
            extra["coverage"] = cov[..., :: c.m][..., : space.n] if cov.ndim and cov.shape[-1] == self.frames else cov
        np.savez_compressed(out, masks=masks, **extra)
        return out, scaled_grow(grow, c, space)


def native_occluder(occ, n, shape, paint_args, out):
    """the native paint's operational occluder flags for a bound crop-space Occluder: the mask in take px and the user's grow scaled
    to take px. Operational flags come last, so this grow is the one paint reads."""
    path, g = occ.for_space(NativeSpace(n, shape), occluder_grow(paint_args), out)
    return ("--occluder-mask", path, "--occluder-grow", str(g))
