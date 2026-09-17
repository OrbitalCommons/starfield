#!/usr/bin/env python3
"""Generate the embedded Mars albedo tier from the USGS Viking colour mosaic.

Mars has no composition data yet (#66 — CRISM/OMEGA bright and dark terrain
endmembers), so this is deliberately a **one-endmember** tier: brightness varies
across the disk and colour does not. That is an honest limitation rather than a
hidden one, and it is exactly what the scalar-map path already does — this just
makes it work offline.

The Viking colour mosaic is an 8-bit RGB visualisation product. Its DN values
are not calibrated reflectance, so they are rescaled so that the disk-averaged
albedo matches Mars' published geometric albedo. That rescaling is the reason
this tier is usable for *appearance* and not for photometry; see `calibrated`
on the product entry.

Usage:
    uv run --with tifffile --with numpy \\
        python build_mars_tier.py ../data/mars_albedo_0p1deg.bin.gz
"""

import gzip
import math
import os
import struct
import sys

import numpy as np
import tifffile

SOURCE_TIF = os.path.expanduser("~/.cache/starfield/planet-maps/mars.tif")
PRODUCT = "Mars_Viking_ClrMosaic_global_925m.tif"

# Output grid: 0.1 deg/px, the tier resolution agreed with the consumer.
OUT_W, OUT_H = 3600, 1800

MAGIC = b"SFEMv3\n"
SCRIPT_VERSION = "2"

# USGS equirectangular: east-positive planetocentric, cell edges on the bounds,
# north at row 0, column 0 at longitude 0.
LON0_DEG = 0.0
LAT0_DEG = 90.0
LON_SENSE = 0
LAT_KIND = 0
REGISTRATION = 0
BODY_NAIF_ID = 499

# Mars' geometric albedo (Mallama et al. 2017). The mosaic is a visualisation
# product with no absolute calibration, so the DN field is scaled to reproduce
# this as its disk average rather than trusted as reflectance.
MARS_GEOMETRIC_ALBEDO = 0.17

# The endmember this tier's albedo is expressed against, matching the product
# entry. WeatheredBasalt is a terrestrial analogue standing in until #66.
ENDMEMBER = "WeatheredBasalt"
ENDMEMBER_BAND_MEAN = 0.147  # solar-weighted 400-2400 nm, from the library

# Luminance weights for collapsing RGB to a single brightness. Rec. 601, which
# is what a broadband panchromatic response most resembles.
RGB_WEIGHTS = (0.299, 0.587, 0.114)


def main(out_path):
    if not os.path.exists(SOURCE_TIF):
        raise SystemExit(f"{SOURCE_TIF} missing; download {PRODUCT} first")

    with tifffile.TiffFile(SOURCE_TIF) as tif:
        page = tif.pages[0]
        bands, in_h, in_w = page.shape
        assert bands == 3, page.shape
        print(f"source {in_w}x{in_h}, {bands} bands", file=sys.stderr)

        # Accumulate luminance into the output grid by area-averaging. The input
        # is not an integer multiple of the output, so bin by index rather than
        # reshaping.
        acc = np.zeros((OUT_H, OUT_W), dtype=np.float64)
        cnt = np.zeros((OUT_H, OUT_W), dtype=np.float64)
        col_bin = np.minimum((np.arange(in_w) * OUT_W // in_w), OUT_W - 1)

        data = page.asarray()  # (3, h, w) uint8, planar
        for r0 in range(0, in_h, 512):
            r1 = min(r0 + 512, in_h)
            chunk = data[:, r0:r1, :].astype(np.float32)
            lum = (
                RGB_WEIGHTS[0] * chunk[0]
                + RGB_WEIGHTS[1] * chunk[1]
                + RGB_WEIGHTS[2] * chunk[2]
            )
            # Viking never imaged the poles; the mosaic fills them with exact
            # black. Averaging that in would encode a dark polar cap, which is
            # the opposite of the truth, so no-data is excluded from both the
            # bin means and the calibration.
            valid = (chunk[0] > 0) | (chunk[1] > 0) | (chunk[2] > 0)
            rows = np.minimum((np.arange(r0, r1) * OUT_H // in_h), OUT_H - 1)
            for i, orow in enumerate(rows):
                np.add.at(acc[orow], col_bin, lum[i] * valid[i])
                np.add.at(cnt[orow], col_bin, valid[i].astype(np.float64))
            print(f"  rows {r0}-{r1}", file=sys.stderr)

    nodata = cnt == 0
    lum = acc / np.maximum(cnt, 1.0)

    lat = np.linspace(90 - 0.05, -90 + 0.05, OUT_H)
    area = np.cos(np.radians(lat))[:, None]
    area_full = np.broadcast_to(area, (OUT_H, OUT_W))
    nodata_frac = (area_full * nodata).sum() / area_full.sum()
    band = np.where(nodata.any(axis=1))[0]
    if band.size:
        print(f"NO-DATA: {nodata_frac * 100:.2f}% of area, rows "
              f"{band.min()}-{band.max()} "
              f"(lat {lat[band.min()]:.1f} to {lat[band.max()]:.1f})",
              file=sys.stderr)

    # Rescale so the area-weighted mean over VALID texels reproduces Mars'
    # geometric albedo.
    valid_area = area_full * ~nodata
    mean_dn = (lum * valid_area).sum() / valid_area.sum()
    albedo = lum * (MARS_GEOMETRIC_ALBEDO / mean_dn)
    print(f"\nmean DN {mean_dn:.2f} -> scaled to geometric albedo "
          f"{MARS_GEOMETRIC_ALBEDO}", file=sys.stderr)
    print(f"albedo range {albedo.min():.4f} - {albedo.max():.4f}", file=sys.stderr)

    # One-endmember abundance: albedo / the endmember's own band mean. This
    # exceeds 1, which is why the format carries a scale.
    abundance = albedo / ENDMEMBER_BAND_MEAN
    # No-data encodes as raw 0, which for a one-endmember tier yields an empty
    # mix with total_weight() == 0 — distinguishable from any real surface,
    # since a physical albedo is never zero.
    abundance[nodata] = 0.0
    scale = float(math.ceil(abundance.max() * 10.0) / 10.0)
    print(f"abundance range {abundance.min():.3f} - {abundance.max():.3f}, "
          f"scale {scale}", file=sys.stderr)

    body = np.clip(abundance / scale * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes()
    names_blob = ENDMEMBER.encode()
    provenance = (
        f"USGS {PRODUCT}; visualisation product, DN rescaled so the "
        f"area-weighted mean is the published geometric albedo "
        f"{MARS_GEOMETRIC_ALBEDO} (Mallama et al. 2017); "
        f"one endmember ({ENDMEMBER}, band mean {ENDMEMBER_BAND_MEAN}) pending "
        f"CRISM/OMEGA terrain units; "
        f"NO POLAR COVERAGE: {nodata_frac * 100:.2f}% of area is no-data, "
        f"encoded as abundance 0; "
        f"build_mars_tier.py v{SCRIPT_VERSION}"
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
    main(sys.argv[1] if len(sys.argv) > 1 else "mars_albedo_0p1deg.bin.gz")
