import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import godracula as gd  # noqa: E402


@pytest.fixture(scope='module')
def t():
    return gd.Transfer()


def px(hexes):
    return np.array([[[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in hexes]], np.uint8)


def hexes(arr):
    return {'%02X%02X%02X' % tuple(p) for p in arr.reshape(-1, 3)}


def lab_of(h):
    return gd.rgb2lab(gd.hex2rgb(h))


# ---------- the mapping ----------
def test_every_nord_color_maps_exactly_to_its_target(t):
    a = gd.anchors()
    out = t.rgb8(px(gd.NORD))[0]
    for src, got in zip(gd.NORD, out):
        assert '%02X%02X%02X' % tuple(got) == a[src], src


def test_every_core_dracula_color_is_in_the_palette(t):
    for name, h in gd.DRACULA.items():
        assert h in t.targets, name


def test_no_nord_color_survives(t):
    assert not set(gd.NORD) & set(t.targets)


@pytest.mark.parametrize('ramp', [['2E3440', '3B4252', '434C5E', '4C566A'],
                                  ['D8DEE9', 'E5E9F0', 'ECEFF4']])
def test_nord_ramps_stay_distinct_and_ordered(ramp):
    a = gd.anchors()
    L = [lab_of(a[c])[0] for c in ramp]
    assert all(b - a_ > 0.01 for a_, b in zip(L, L[1:])), L


def test_compression_noise_lands_near_the_target(t):
    rng = np.random.default_rng(0)
    a = gd.anchors()
    for src in gd.NORD:
        base = px([src]).astype(int)
        noisy = np.clip(base + rng.integers(-4, 5, (1, 64, 3)), 0, 255).astype(np.uint8)
        out = t.rgb8(noisy).reshape(-1, 3) / 255.0
        d = np.linalg.norm(gd.rgb2lab(out) - lab_of(a[src]), axis=-1)
        assert d.max() < 0.05, (src, d.max())


def test_neutral_greys_stay_low_chroma(t):
    # fog and mist between Polar Night and Snow Storm must not turn pink or neon
    greys = [gd.mix('4C566A', 'D8DEE9', s) for s in np.linspace(0, 1, 21)]
    lab = gd.rgb2lab(t.rgb8(px(greys)).reshape(-1, 3) / 255.0)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    assert chroma.max() < 0.08, chroma.max()


def test_grey_axis_keeps_its_lightness_order(t):
    greys = [gd.mix('2E3440', 'ECEFF4', s) for s in np.linspace(0, 1, 64)]
    L = gd.rgb2lab(t.rgb8(px(greys)).reshape(-1, 3) / 255.0)[:, 0]
    assert (np.diff(L) > -0.002).all(), np.diff(L).min()


def test_transfer_is_smooth_along_a_gradient(t):
    ramp = [gd.mix('2E3440', 'ECEFF4', s) for s in np.linspace(0, 1, 256)]
    lab = gd.rgb2lab(t.rgb8(px(ramp)).reshape(-1, 3) / 255.0)
    steps = np.linalg.norm(np.diff(lab, axis=0), axis=-1)
    assert steps.max() < 0.03, steps.max()


# ---------- whole images ----------
def nord_blocks(size=(96, 64), noise=0, seed=1):
    rng = np.random.default_rng(seed)
    h, w = size[1], size[0]
    idx = (np.arange(h)[:, None] // 8 + np.arange(w)[None] // 8) % len(gd.NORD)
    arr = px(gd.NORD)[0][idx].astype(int)
    if noise:
        arr = np.clip(arr + rng.integers(-noise, noise + 1, arr.shape), 0, 255)
    return arr.astype(np.uint8)


def test_quantized_image_is_snapped_onto_the_palette(t):
    out, snapped = gd.recolor(Image.fromarray(nord_blocks(noise=4)), t)
    assert snapped
    assert hexes(np.array(out)) <= set(t.targets)


def test_smooth_image_is_not_snapped_and_keeps_its_tones(t):
    g = np.linspace(0, 1, 256)
    row = np.array([gd.lab2rgb(gd.rgb2lab(gd.hex2rgb('3B4252')) * (1 - s) + gd.rgb2lab(gd.hex2rgb('A0B0D0')) * s)
                    for s in g])
    arr = (np.repeat(row[None], 32, 0) * 255).round().astype(np.uint8)
    out, snapped = gd.recolor(Image.fromarray(arr), t)
    assert not snapped
    assert len(hexes(np.array(out))) > 100


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'P'])
def test_size_mode_and_alpha_are_kept(t, mode):
    arr = nord_blocks()
    if mode == 'RGBA':
        alpha = np.tile(np.linspace(0, 255, arr.shape[1]).astype(np.uint8), (arr.shape[0], 1))
        img = Image.fromarray(np.dstack([arr, alpha]), 'RGBA')
    elif mode == 'P':
        img = Image.fromarray(arr).quantize(16)
    else:
        img = Image.fromarray(arr)
    out, _ = gd.recolor(img, t)
    assert out.size == img.size
    assert out.mode == ('RGBA' if mode == 'RGBA' else 'RGB')
    if mode == 'RGBA':
        assert np.array_equal(np.array(out)[..., 3], alpha)


@pytest.mark.parametrize('ext', ['.png', '.jpg'])
def test_convert_file_in_place(tmp_path, ext):
    p = str(tmp_path / ('w' + ext))
    Image.fromarray(nord_blocks()).save(p)
    gd.convert_file(p, p)
    assert sorted(os.listdir(tmp_path)) == ['w' + ext]   # no temp file left behind
    with Image.open(p) as im:
        got = np.array(im.convert('RGB')).reshape(-1, 3)
    # mostly Dracula: the background shade is present, Nord's is gone
    assert (got == px(['282A36'])[0, 0]).all(1).mean() > 0.03 or ext == '.jpg'
    assert not (got == px(['2E3440'])[0, 0]).all(1).any()


def test_jpeg_keeps_the_source_quality_and_subsampling(tmp_path):
    from PIL import JpegImagePlugin
    p = str(tmp_path / 'w.jpg')
    Image.fromarray(nord_blocks(noise=3)).save(p, quality=75, subsampling=2)
    with Image.open(p) as im:
        before = (im.quantization, JpegImagePlugin.get_sampling(im))
    gd.convert_file(p, p)
    with Image.open(p) as im:
        assert (im.quantization, JpegImagePlugin.get_sampling(im)) == before


def test_gather_walks_folders_and_skips_other_files(tmp_path):
    (tmp_path / 'a').mkdir()
    for n in ['a/x.png', 'a/y.JPG', 'z.jpeg', 'notes.md']:
        (tmp_path / n).write_bytes(b'')
    found = sorted(os.path.relpath(p, tmp_path) for p, _ in gd.gather([str(tmp_path)]))
    assert found == ['a/x.png', 'a/y.JPG', 'z.jpeg']
