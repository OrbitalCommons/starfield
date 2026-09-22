#!/usr/bin/env python3
"""Check consumer dependency isolation without workspace feature unification."""
import json
import subprocess


GROUPS = {
    "catalogs": {"gaia", "gaia-extended", "bright-galaxies", "hipparcos", "mast", "nsa"},
    "jpl": {"horizons", "sbdb", "mpc", "rubin"},
    "surfaces": {"planet-maps", "reflectance-library", "planet-spectra", "solar-spectrum"},
}


def main():
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--format-version", "1", "--no-deps"
    ], text=True))
    facade = next(p for p in metadata["packages"] if p["name"] == "starfield")
    features = sorted(set(facade["features"]) - {"python-tests"})
    for feature in [""] + features:
        # cargo tree with one selected package excludes dev/build dependencies and
        # avoids feature unification with the tools package (which enables Gaia).
        output = subprocess.check_output([
            "cargo", "tree", "-p", "starfield", "--no-default-features",
            "--features", feature, "--edges", "normal", "--prefix", "none",
            "--format", "{p}",
        ], text=True)
        packages = {line.split()[0] for line in output.splitlines()}
        expected_groups = {g for g, sources in GROUPS.items()
                           if feature in sources | {g, "all-data"}}
        if feature in {"gaia-all", "radial-profiles"}:
            expected_groups.add("catalogs")
        actual_groups = {g for g in GROUPS if f"starfield-{g}" in packages}
        assert actual_groups == expected_groups, (feature, actual_groups, expected_groups)
        gaia = feature in {"gaia", "gaia-all", "gaia-extended", "bright-galaxies",
                           "mast", "catalogs", "all-data"}
        fits = feature in {"mast", "nsa", "radial-profiles", "catalogs", "all-data"}
        assert ("arrow" in packages) == gaia, (feature, "unexpected Arrow dependency")
        assert ("fitsio-pure" in packages) == fits, (feature, "unexpected FITS dependency")
        print(f"{feature or '(no features)'}: groups={','.join(sorted(actual_groups)) or 'none'}, "
              f"Arrow={gaia}, FITS={fits}")


if __name__ == "__main__":
    main()
