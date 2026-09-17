#!/usr/bin/env python3
"""Generate the embedded Moon albedo tier from the LROC WAC global mosaic.

Like Mars, this is a **one-endmember** tier pending real terrain units (#65 —
lunar mare and highland endmembers from RELAB/LSCC). Brightness varies across
the disk; colour does not. The mare/highland dichotomy is the single largest
albedo feature on the Moon, so getting colour out of it is worth doing, but it
needs measured lunar soil spectra rather than the terrestrial basalt analogue
this tier is expressed against.

The source is 109164 x 54582 single-band uint8, ~6 GB uncompressed, so it is
read through a memory map in row blocks and never held whole.

WAC's global morphology mosaic is normalised for illumination and is not
calibrated reflectance, so DN is rescaled to reproduce the Moon's published
geometric albedo as its disk average — the same treatment as the Mars tier, and
the same reason both product entries are `calibrated: false`.

Usage:
    uv run --with tifffile --with numpy \\
        python build_moon_tier.py ../data/moon_albedo_0p1deg.bin.gz
"""

import gzip
import math
import os
import struct
import sys

import numpy as np
import tifffile

SOURCE_TIF = os.path.expanduser("~/.cache/starfield/planet-maps/moon.tif")
PRODUCT = "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"

OUT_W, OUT_H = 3600, 1800

MAGIC = b"SFEMv3\n"
SCRIPT_VERSION = "4"

# USGS equirectangular, east-positive planetocentric, north at row 0. The Moon's
# flattening is ~1.2e-3 and USGS lunar products are defined on a sphere, so
# planetocentric and planetographic coincide.
#
# LON0 IS -180, NOT 0. The WAC mosaic is centred on the prime meridian, unlike
# the USGS Mars mosaics, which start at it. Same agency, same projection,
# different origin — verified against the source by landmark: at lon0 = -180 the
# maria read 27-37 DN and the highlands 75-76 with Tycho brightest at 121, which
# is the real lunar albedo pattern; at lon0 = 0 that inverts and the maria come
# out brighter than the highlands.
LON0_DEG = -180.0
LAT0_DEG = 90.0
LON_SENSE = 0
LAT_KIND = 0
REGISTRATION = 0
BODY_NAIF_ID = 301

# Lunar geometric albedo (Mallama et al. 2017).
MOON_GEOMETRIC_ALBEDO = 0.12

# Terrestrial analogue standing in until #65 ships real lunar soils. Fresh
# basalt is the right spectral family for mare and the wrong one for highlands,
# which is precisely what #65 fixes.
ENDMEMBER = "FreshBasalt"
ENDMEMBER_BAND_MEAN = 0.102  # solar-weighted 400-2400 nm, from the library

# WAC's global mosaic is the 643 nm band.
BAND_NM = (633.0, 653.0)

# DN below this is treated as no-data. See the comment at its use.
NODATA_DN_FLOOR = 5

# An output texel needs at least this fraction of its source pixels to be valid
# before it counts as a measurement. Without it, a texel averaged from a handful
# of pixels along a swath edge produces a very low albedo that is an artefact of
# partial coverage rather than dark ground -- ~0.1% of the tier within +/-70 deg
# latitude, but they read as implausibly dark terrain.
MIN_COVERAGE = 0.5


def main(out_path):
    if not os.path.exists(SOURCE_TIF):
        raise SystemExit(f"{SOURCE_TIF} missing; download {PRODUCT} first")

    with tifffile.TiffFile(SOURCE_TIF) as tif:
        page = tif.pages[0]
        print(f"source {page.shape}, dtype {page.dtype}, "
              f"compression {page.compression}", file=sys.stderr)
        in_h, in_w = page.shape[-2], page.shape[-1]

    # Memory-map rather than read: the array is ~6 GB.
    data = tifffile.memmap(SOURCE_TIF)
    if data.ndim == 3:
        data = data[0]
    assert data.shape == (in_h, in_w), (data.shape, in_h, in_w)

    acc = np.zeros((OUT_H, OUT_W), dtype=np.float64)
    cnt = np.zeros((OUT_H, OUT_W), dtype=np.float64)
    col_bin = np.minimum((np.arange(in_w) * OUT_W // in_w), OUT_W - 1)

    for r0 in range(0, in_h, 256):
        r1 = min(r0 + 256, in_h)
        chunk = np.asarray(data[r0:r1, :], dtype=np.float32)
        # WAC gaps are not all exact zero: about 6% of the mosaic sits at DN
        # 1-3, spread across every latitude, which is swath seams rather than
        # terrain. The darkest real lunar surface is mare at DN ~33 (albedo
        # 0.06), and even permanently shadowed polar craters are a fraction of
        # a percent of area, so a DN floor of 5 is an order of magnitude below
        # anything genuine and cannot erase real ground.
        valid = chunk >= NODATA_DN_FLOOR
        rows = np.minimum((np.arange(r0, r1) * OUT_H // in_h), OUT_H - 1)
        # bincount rather than np.add.at: the latter is an order of magnitude
        # slower and this loop runs over 6e9 source pixels.
        for i, orow in enumerate(rows):
            v = valid[i]
            acc[orow] += np.bincount(col_bin, weights=chunk[i] * v, minlength=OUT_W)
            cnt[orow] += np.bincount(col_bin, weights=v.astype(np.float64), minlength=OUT_W)
        if r0 % 8192 == 0:
            print(f"  rows {r0}/{in_h}", file=sys.stderr)

    # Source pixels per output texel, so coverage is a fraction rather than a
    # raw count that varies with latitude.
    per_texel = (in_h / OUT_H) * (in_w / OUT_W)
    coverage = cnt / per_texel
    nodata = coverage < MIN_COVERAGE
    lum = acc / np.maximum(cnt, 1.0)

    lat = np.linspace(90 - 0.05, -90 + 0.05, OUT_H)
    area = np.broadcast_to(np.cos(np.radians(lat))[:, None], (OUT_H, OUT_W))
    nodata_frac = (area * nodata).sum() / area.sum()
    print(f"\nno-data: {nodata_frac * 100:.3f}% of area", file=sys.stderr)

    valid_area = area * ~nodata
    mean_dn = (lum * valid_area).sum() / valid_area.sum()
    albedo = lum * (MOON_GEOMETRIC_ALBEDO / mean_dn)
    print(f"mean DN {mean_dn:.2f} -> geometric albedo {MOON_GEOMETRIC_ALBEDO}",
          file=sys.stderr)
    print(f"albedo range {albedo.min():.4f} - {albedo.max():.4f}", file=sys.stderr)

    abundance = albedo / ENDMEMBER_BAND_MEAN
    abundance[nodata] = 0.0
    scale = float(math.ceil(abundance.max() * 10.0) / 10.0)
    print(f"abundance max {abundance.max():.3f}, scale {scale}", file=sys.stderr)

    body = np.clip(abundance / scale * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes()
    names_blob = ENDMEMBER.encode()
    provenance = (
        f"USGS {PRODUCT} ({BAND_NM[0]:.0f}-{BAND_NM[1]:.0f} nm); "
        f"morphology mosaic, DN rescaled so the area-weighted mean is the "
        f"published geometric albedo {MOON_GEOMETRIC_ALBEDO} "
        f"(Mallama et al. 2017); one endmember ({ENDMEMBER}, band mean "
        f"{ENDMEMBER_BAND_MEAN}) pending RELAB/LSCC lunar soils; "
        f"no-data (DN < {NODATA_DN_FLOOR} or coverage < {MIN_COVERAGE}) "
        f"{nodata_frac * 100:.2f}% of area, encoded as abundance 0; "
        f"build_moon_tier.py v{SCRIPT_VERSION}"
    ).encode()

    header = MAGIC + struct.pack(
        "<HHBBBBiddQ",
        OUT_W, OUT_H, 1,
        LON_SENSE, LAT_KIND, REGISTRATION,
        BODY_NAIF_ID,
        LON0_DEG, LAT0_DEG,
        len(body),
    )
    header += struct.pack("<f", scale)
    header += struct.pack("<H", len(names_blob)) + names_blob
    header += struct.pack("<H", len(provenance)) + provenance

    assert len(body) == OUT_W * OUT_H

    with gzip.open(out_path, "wb", compresslevel=9) as fh:
        fh.write(header)
        fh.write(body)
    print(f"\nwrote {out_path}: {os.path.getsize(out_path) / 1e6:.2f} MB "
          f"({OUT_W}x{OUT_H}, 1 endmember, scale {scale})", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "moon_albedo_0p1deg.bin.gz")
