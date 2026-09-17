#!/usr/bin/env python3
"""Generate the embedded Earth endmember-abundance tier from MODIS MCD12C1.

Reads the sub-pixel IGBP land-cover percentages from MCD12C1 (0.05 deg CMG,
3600x7200x17), collapses the 17 classes onto the endmembers shipped by
starfield-reflectance-library, splits open ocean from coastal water with a
distance-to-land transform, and writes a downsampled abundance tier.

The credential is needed only here. The output is derived, redistributable
NASA open data, so every downstream user and CI stays credential-free.

Requires an Earthdata login in ~/.netrc for the download step; the file is
cached, so a second run needs no credential at all.

Usage:
    uv run --with pyhdf --with numpy --with scipy \\
        python build_earth_tier.py ../data/earth_endmembers_0p25deg.bin.gz
"""

import gzip
import os
import struct
import sys

import numpy as np
from pyhdf.SD import SD, SDC

SOURCE_HDF = os.path.expanduser("~/.cache/starfield/modis/mcd12c1.hdf")
GRANULE = "MCD12C1.A2024001.061.2025216131527"

# 0.05 deg -> 0.25 deg. 7200/5 = 1440, 3600/5 = 720, exact.
BLOCK = 5

# A water texel within this many 0.25 deg cells of land is CoastalWater.
# Build-time parameter, recorded in the header: it is exactly the sort of knob
# that hides bias, so it is stated rather than buried.
COASTAL_RADIUS_CELLS = 2

MAGIC = b"SFEMv3\n"
SCRIPT_VERSION = "3"

# MCD12C1 is a Climate Modeling Grid product: plate carree on WGS84, spanning
# -180..180 west-to-east and 90..-90 north-to-south, with cell EDGES on the
# bounds. Geodetic latitude, since the datum is WGS84.
#
# These go in the header rather than living only in the product catalogue, so a
# loader that has never heard of MCD12C1 still cannot misregister the grid by
# half a cell or mirror it.
LON0_DEG = -180.0        # longitude of column 0's LEFT edge
LAT0_DEG = 90.0          # latitude of row 0's TOP edge
LON_SENSE = 0            # 0 = east-positive, 1 = west-positive
LAT_KIND = 1             # 0 = planetocentric, 1 = planetographic/geodetic
REGISTRATION = 0         # 0 = cell edges on bounds, 1 = cell centres
BODY_NAIF_ID = 399

# Bump when any weight in IGBP changes: the fitted numbers are only defensible
# if you can tell which fit produced a given artefact.
CLASS_TABLE_VERSION = "igbp-fitted-1"

# Endmember weights per IGBP class, fitted against published MODIS class-mean
# shortwave albedos (Gao et al. 2005; MCD43 climatology, snow-free) and signed
# off by the focalplane session. Each row sums to 1.0.
#
# "Shade" is the standard spectral-mixture-analysis shade endmember: identically
# zero reflectance, accounting for canopy shadowing, gaps and multiple
# scattering at the texel scale. Without it the vegetated rows cannot reach
# published albedos at all -- leaf-level spectra are far brighter than canopies.
IGBP = [
    # (index, name, {endmember: weight}, fitted SW albedo or None)
    (0,  "Water Bodies",              {"WATER": 1.00},                                                        0.024),
    (1,  "Evergreen Needleleaf",      {"GreenVegetation": 0.55, "Shade": 0.45},                               0.120),
    (2,  "Evergreen Broadleaf",       {"GreenVegetation": 0.55, "Shade": 0.45},                               0.120),
    (3,  "Deciduous Needleleaf",      {"GreenVegetation": 0.55, "Shade": 0.45},                               0.120),
    (4,  "Deciduous Broadleaf",       {"GreenVegetation": 0.55, "Shade": 0.45},                               0.120),
    (5,  "Mixed Forests",             {"GreenVegetation": 0.55, "Shade": 0.45},                               0.120),
    (6,  "Closed Shrublands",         {"GreenVegetation": 0.50, "DryVegetation": 0.20, "Shade": 0.30},        0.158),
    (7,  "Open Shrublands",           {"GreenVegetation": 0.20, "DryVegetation": 0.25,
                                       "AridSoil": 0.30, "Shade": 0.25},                                     0.241),
    (8,  "Woody Savannas",            {"GreenVegetation": 0.50, "DryVegetation": 0.25, "Shade": 0.25},        0.170),
    (9,  "Savannas",                  {"GreenVegetation": 0.30, "DryVegetation": 0.50, "Shade": 0.20},        0.187),
    (10, "Grasslands",                {"GreenVegetation": 0.40, "DryVegetation": 0.45, "Shade": 0.15},        0.197),
    (11, "Permanent Wetlands",        {"GreenVegetation": 0.50, "CoastalWater": 0.50},                        None),
    (12, "Croplands",                 {"GreenVegetation": 0.65, "DryVegetation": 0.20, "Shade": 0.15},        0.191),
    (13, "Urban and Built-up",        {"Asphalt": 0.55, "Concrete": 0.25,
                                       "GreenVegetation": 0.15, "Shade": 0.05},                              0.174),
    (14, "Cropland/Natural Mosaic",   {"GreenVegetation": 0.60, "DryVegetation": 0.25, "Shade": 0.15},        0.192),
    (15, "Permanent Snow and Ice",    {"Snow": 1.00},                                                        0.629),
    (16, "Barren",                    {"AridSoil": 0.85, "Shade": 0.15},                                      0.387),
]

# "WATER" in class 0 resolves to one of these by the distance transform.
WATER_SPLIT = ("OpenOcean", "CoastalWater")

# Solar-weighted 400-2400 nm albedo per endmember, for the sanity number.
# Computed from the shipped reflectance library against TSIS-1 HSRS v2.
ENDMEMBER_SW_ALBEDO = {
    "OpenOcean": 0.024, "CoastalWater": 0.024, "GreenVegetation": 0.218,
    "DryVegetation": 0.244, "AridSoil": 0.455, "Sand": 0.293, "Snow": 0.629,
    "Asphalt": 0.121, "Concrete": 0.297, "Shade": 0.0,
}


def endmember_names():
    names = set(WATER_SPLIT)
    for _, _, weights, _ in IGBP:
        names.update(k for k in weights if k != "WATER")
    return sorted(names)


def main(out_path):
    if not os.path.exists(SOURCE_HDF):
        raise SystemExit(f"{SOURCE_HDF} missing; see the module docstring")

    names = endmember_names()
    index = {n: i for i, n in enumerate(names)}
    print(f"endmembers ({len(names)}): {', '.join(names)}", file=sys.stderr)

    f = SD(SOURCE_HDF, SDC.READ)
    sds = f.select("Land_Cover_Type_1_Percent")
    rows, cols, classes = sds.info()[2]
    assert (rows, cols, classes) == (3600, 7200, 17), (rows, cols, classes)

    out_h, out_w = rows // BLOCK, cols // BLOCK
    acc = np.zeros((len(names), out_h, out_w), dtype=np.float32)
    water = np.zeros((out_h, out_w), dtype=np.float32)
    classified = np.zeros((out_h, out_w), dtype=np.float32)

    # Stream in row blocks; the full array is 441 MB and there is no need to
    # hold it.
    for r0 in range(0, rows, BLOCK * 40):
        r1 = min(r0 + BLOCK * 40, rows)
        chunk = sds[r0:r1, :, :].astype(np.float32) / 100.0
        # (dr, out_h_chunk, BLOCK, out_w, BLOCK, classes) -> mean over the 5x5
        n = (r1 - r0) // BLOCK
        chunk = chunk.reshape(n, BLOCK, out_w, BLOCK, classes).mean(axis=(1, 3))
        o0 = r0 // BLOCK
        for idx, _, weights, _ in IGBP:
            frac = chunk[:, :, idx]
            for member, w in weights.items():
                if member == "WATER":
                    water[o0:o0 + n] += frac * w
                else:
                    acc[index[member], o0:o0 + n] += frac * w
        classified[o0:o0 + n] += chunk.sum(axis=2)
        print(f"  rows {r0}-{r1}", file=sys.stderr)

    # Coastal split: a water cell near land is CoastalWater, else OpenOcean.
    from scipy.ndimage import binary_dilation
    land = classified - water > 0.5
    near_land = binary_dilation(land, iterations=COASTAL_RADIUS_CELLS)
    acc[index["CoastalWater"]] += np.where(near_land, water, 0.0)
    acc[index["OpenOcean"]] += np.where(near_land, 0.0, water)

    # --- sanity numbers -------------------------------------------------
    lat = np.linspace(90 - 0.125, -90 + 0.125, out_h)
    cos_lat = np.cos(np.radians(lat))[:, None]
    area = np.broadcast_to(cos_lat, (out_h, out_w))
    total_area = area.sum()

    unclassified = np.clip(1.0 - classified, 0.0, 1.0)
    print(f"\nunclassified area fraction: "
          f"{(unclassified * area).sum() / total_area * 100:.3f}%", file=sys.stderr)

    albedo = np.zeros((out_h, out_w), dtype=np.float32)
    for name in names:
        albedo += acc[index[name]] * ENDMEMBER_SW_ALBEDO[name]
    land_mask = land & (classified > 0.5)
    land_area = (area * land_mask).sum()
    land_mean = (albedo * area * land_mask).sum() / land_area
    global_mean = (albedo * area).sum() / total_area

    # The published 0.20-0.25 target is explicitly SNOW-FREE, so compare like
    # with like: Antarctica and Greenland at 0.629 would otherwise inflate the
    # land mean and make an agreeing number look like a disagreeing one.
    snow_frac = acc[index["Snow"]]
    snow_free = land_mask & (snow_frac < 0.5)
    snow_free_area = (area * snow_free).sum()
    snow_free_mean = (albedo * area * snow_free).sum() / snow_free_area

    print(f"area-weighted land mean SW albedo (all):       {land_mean:.4f}",
          file=sys.stderr)
    print(f"area-weighted land mean SW albedo (snow-free): {snow_free_mean:.4f}  "
          f"<- compare with published ~0.20-0.25", file=sys.stderr)
    print(f"permanent snow/ice share of land area:         "
          f"{(area * (snow_frac >= 0.5)).sum() / land_area * 100:.1f}%",
          file=sys.stderr)
    print(f"area-weighted global mean SW albedo:           {global_mean:.4f}",
          file=sys.stderr)
    print("  (global figure is the DIFFUSE SURFACE term only: no specular"
          " glint, no clouds.\n"
          "   Earth's ~0.29 Bond albedo is mostly cloud, which is the"
          " consumer's layer.)", file=sys.stderr)

    # --- write ----------------------------------------------------------
    body = np.clip(acc * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes()
    names_blob = "\n".join(names).encode()
    provenance = (
        f"MCD12C1 collection 061, granule {GRANULE}; "
        f"class table {CLASS_TABLE_VERSION}; "
        f"build_earth_tier.py v{SCRIPT_VERSION}; "
        f"coastal radius {COASTAL_RADIUS_CELLS} cells; "
        f"weights fitted to MODIS class-mean SW albedos "
        f"(Gao et al. 2005; MCD43 climatology, snow-free)"
    ).encode()

    # All multi-byte fields little-endian.
    header = MAGIC + struct.pack(
        "<HHBBBBiddQ",
        out_w, out_h, len(names),
        LON_SENSE, LAT_KIND, REGISTRATION,
        BODY_NAIF_ID,
        LON0_DEG, LAT0_DEG,
        len(body),
    )
    # Composition tier: abundances are a partition, so full scale is 1.0.
    header += struct.pack("<f", 1.0)
    header += struct.pack("<H", len(names_blob)) + names_blob
    header += struct.pack("<H", len(provenance)) + provenance

    assert len(body) == len(names) * out_w * out_h, "payload length mismatch"

    with gzip.open(out_path, "wb", compresslevel=9) as fh:
        fh.write(header)
        fh.write(body)
    print(f"\nwrote {out_path}: {os.path.getsize(out_path) / 1e6:.2f} MB "
          f"({out_w}x{out_h}, {len(names)} endmembers)", file=sys.stderr)
    print(f"granule {GRANULE}, coastal radius {COASTAL_RADIUS_CELLS} cells",
          file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "earth_endmembers_0p25deg.bin.gz")
