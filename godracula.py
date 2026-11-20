"""Recolor the Nord wallpapers to Dracula with ImageGoNord.

Running ImageGoNord straight on a Nord image with the 11 Dracula colors posterizes
it: Nord's four Polar Night shades all land on one Dracula grey, its three Snow
Storm whites on one white, and soft gradients turn into blotches. So each image
goes through two steps:

1. A smooth Nord -> Dracula color transfer in OKLab. It is exact on Nord's 16
   colors (each goes to a chosen Dracula color, keeping Nord's lightness steps)
   and interpolates in between, so JPEG noise and gradients follow along.
2. ImageGoNord snaps the result onto the Dracula palette (the transfer's target
   colors). Images with continuous tones (photos, soft gradients) skip the snap,
   because quantizing them bands visibly; they keep the transfer alone.

usage: python godracula.py FILE_OR_FOLDER... [--out DIR] [--jobs N]
       (without --out the files are replaced in place)
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
EXTS = ('.png', '.jpg', '.jpeg')

# ---------- palettes ----------
NORD = ['2E3440', '3B4252', '434C5E', '4C566A',             # Polar Night
        'D8DEE9', 'E5E9F0', 'ECEFF4',                       # Snow Storm
        '8FBCBB', '88C0D0', '81A1C1', '5E81AC',             # Frost
        'BF616A', 'D08770', 'EBCB8B', 'A3BE8C', 'B48EAD']   # Aurora
DRACULA = {'background': '282A36', 'current_line': '44475A', 'foreground': 'F8F8F2',
           'comment': '6272A4', 'cyan': '8BE9FD', 'green': '50FA7B', 'orange': 'FFB86C',
           'pink': 'FF79C6', 'purple': 'BD93F9', 'red': 'FF5555', 'yellow': 'F1FA8C'}


# ---------- color math (OKLab) ----------
def srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])
M1I, M2I = np.linalg.inv(M1), np.linalg.inv(M2)


def rgb2lab(rgb):
    """sRGB floats 0..1, shape (..., 3) -> OKLab."""
    return np.cbrt(srgb_to_lin(rgb) @ M1.T) @ M2.T


def lab2rgb(lab):
    return np.clip(lin_to_srgb(((lab @ M2I.T) ** 3) @ M1I.T), 0, 1)


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def rgb2hex(rgb):
    return '%02X%02X%02X' % tuple(int(round(v * 255)) for v in np.clip(rgb, 0, 1))


def mix(a, b, t):
    """Blend two hex colors in OKLab (t=0 gives a, t=1 gives b)."""
    return rgb2hex(lab2rgb(rgb2lab(hex2rgb(a)) * (1 - t) + rgb2lab(hex2rgb(b)) * t))


# ---------- the Nord -> Dracula mapping ----------
def anchors():
    """Nord color -> Dracula color, as hex strings."""
    m = {
        # Polar Night -> Dracula's background ramp (four distinct shades)
        '2E3440': '282A36', '3B4252': '343746', '434C5E': '44475A',
        '4C566A': mix('44475A', '6272A4', 0.5),
        # Snow Storm -> Foreground, the two dimmer whites tinted lavender
        'ECEFF4': 'F8F8F2', 'E5E9F0': mix('F8F8F2', 'BD93F9', 0.14),
        'D8DEE9': mix('F8F8F2', 'BD93F9', 0.28),
        # Frost is one ramp: Comment, Purple, a cyan-purple blend, Cyan. Teal and the light
        # blue are near-twins in Nord, so they must land close together here too
        '5E81AC': '6272A4', '81A1C1': 'BD93F9', '88C0D0': mix('8BE9FD', 'BD93F9', 0.35), '8FBCBB': '8BE9FD',
        # Aurora -> the Dracula accents
        'BF616A': 'FF5555', 'D08770': 'FFB86C', 'EBCB8B': 'F1FA8C', 'A3BE8C': '50FA7B',
        'B48EAD': 'FF79C6',
    }
    # Greys between Polar Night and Snow Storm stay greys (lavender-tinted)
    for t in (0.2, 0.4, 0.6, 0.8):
        m[mix('4C566A', 'D8DEE9', t)] = mix(m['4C566A'], m['D8DEE9'], t)
    m['000000'] = '191A21'   # Dracula's darkest background
    m['FFFFFF'] = 'FFFFFF'
    return m


class Transfer:
    """Inverse-distance-weighted color shift in OKLab, exact on every anchor."""

    # Chroma differences count this many times more than lightness differences when
    # weighing anchors, so a grey follows the grey anchors and not a nearby accent
    # whose shift is large (Nord purple -> Dracula pink would tint fog pink).
    CHROMA_WEIGHT = 3.0

    def __init__(self, power=5.0):
        a = anchors()
        self.src = rgb2lab(np.array([hex2rgb(k) for k in a]))
        self.dst = rgb2lab(np.array([hex2rgb(v) for v in a.values()]))
        self.disp = self.dst - self.src
        self.power = power
        self.scale = np.array([1.0, self.CHROMA_WEIGHT, self.CHROMA_WEIGHT])
        self.targets = sorted(set(a.values()))

    def lab(self, lab):
        d = np.linalg.norm((lab[:, None, :] - self.src[None]) * self.scale, axis=-1)
        w = 1.0 / np.maximum(d, 1e-9) ** self.power
        w /= w.sum(1, keepdims=True)
        return lab + w @ self.disp

    def rgb8(self, arr):
        """uint8 array (..., 3) -> recolored uint8 array; works on unique colors."""
        flat = arr.reshape(-1, 3)
        key = (flat[:, 0].astype(np.int64) << 16) | (flat[:, 1].astype(np.int64) << 8) | flat[:, 2]
        u, inv = np.unique(key, return_inverse=True)
        rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1) / 255.0
        out = np.empty_like(rgb)
        for i in range(0, len(rgb), 65536):
            out[i:i + 65536] = lab2rgb(self.lab(rgb2lab(rgb[i:i + 65536])))
        return np.round(out * 255).astype(np.uint8)[inv.reshape(-1)].reshape(arr.shape)


# ---------- deciding whether ImageGoNord should snap ----------
NEAR = 0.04          # OKLab distance that counts as "a Nord color plus compression noise"
PALETTE_SHARE = 0.85  # share of opaque pixels that must be near a Nord color to snap


def nord_share(rgba):
    """Share of opaque pixels within NEAR of a Nord color (or black/white)."""
    ref = rgb2lab(np.array([hex2rgb(c) for c in NORD + ['000000', 'FFFFFF']]))
    rgb = rgba[..., :3].reshape(-1, 3)
    opaque = rgba[..., 3].reshape(-1) >= 128
    key = (rgb[:, 0].astype(np.int64) << 16) | (rgb[:, 1].astype(np.int64) << 8) | rgb[:, 2]
    u, inv = np.unique(key[opaque], return_inverse=True)
    if len(u) == 0:
        return 1.0
    cnt = np.bincount(inv.reshape(-1), minlength=len(u))
    lab = rgb2lab(np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1) / 255.0)
    near = np.zeros(len(u), bool)
    for i in range(0, len(u), 65536):
        d = np.linalg.norm(lab[i:i + 65536, None] - ref[None], axis=-1).min(1)
        near[i:i + 65536] = d < NEAR
    return cnt[near].sum() / cnt.sum()


def ign_snap(img, colors):
    """ImageGoNord pixel replacement onto the given hex colors."""
    from ImageGoNord import GoNord
    go = GoNord()
    go.reset_palette()
    for c in colors:
        go.add_color_to_palette('#' + c)
    return go.convert_image(img)


# ---------- one image ----------
def recolor(img, transfer=None, snap=None):
    """Recolor a PIL image. snap=None decides from the image. Returns (image, snapped)."""
    transfer = transfer or Transfer()
    mode = img.mode
    has_alpha = mode in ('RGBA', 'LA', 'PA') or (mode == 'P' and 'transparency' in img.info)
    rgba = np.array(img.convert('RGBA'))
    if snap is None:
        snap = nord_share(rgba) >= PALETTE_SHARE
    rgba[..., :3] = transfer.rgb8(rgba[..., :3])
    if snap:
        snapped = ign_snap(Image.fromarray(rgba[..., :3].copy(), 'RGB'), transfer.targets)
        rgb = np.array(snapped)
        # ImageGoNord treats every pixel as opaque here; keep the transfer's color for
        # the fully transparent ones, which never show anyway
        rgba[..., :3] = np.where(rgba[..., 3:4] > 0, rgb, rgba[..., :3])
    out = Image.fromarray(rgba, 'RGBA')
    return (out if has_alpha else out.convert('RGB')), snap


def save(img, path, like=None):
    """Save by extension. A JPEG reuses the source JPEG's quantization tables and
    chroma subsampling (like=(qtables, subsampling)), so quality and size match it."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        opts = {'quality': 92, 'subsampling': 0}
        if like:
            opts = {'qtables': like[0], 'subsampling': like[1]}
        img.convert('RGB').save(path, 'JPEG', optimize=True, **opts)
    else:
        img.save(path, 'PNG', optimize=True)


def convert_file(src, dst):
    with Image.open(src) as im:
        im.load()
        like = None
        if im.format == 'JPEG':
            from PIL import JpegImagePlugin
            like = (im.quantization, JpegImagePlugin.get_sampling(im))
        out, snapped = recolor(im)
    tmp = dst + '.tmp' + os.path.splitext(dst)[1]
    save(out, tmp, like)
    os.replace(tmp, dst)
    return src, snapped


def gather(paths):
    for p in paths:
        if os.path.isdir(p):
            for d, _, fs in os.walk(p):
                for f in sorted(fs):
                    if f.lower().endswith(EXTS):
                        yield os.path.join(d, f), p
        elif p.lower().endswith(EXTS):
            yield p, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--out', help='write here (keeping folder layout) instead of in place')
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    jobs = []
    for src, root in gather(a.paths):
        if a.out:
            rel = os.path.relpath(src, root) if root else os.path.basename(src)
            dst = os.path.join(a.out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        else:
            dst = src
        jobs.append((src, dst))
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(convert_file, s, d) for s, d in jobs]
        for f in futs:
            src, snapped = f.result()
            print(('ign  ' if snapped else 'xfer ') + src, flush=True)


if __name__ == '__main__':
    sys.exit(main())
