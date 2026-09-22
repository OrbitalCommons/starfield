#!/usr/bin/env python3
"""Enforce lockstep versions, local exact internal edges, and package budgets."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", action="store_true", help="check generated .crate sizes")
    args = parser.parse_args()
    metadata = json.loads(subprocess.check_output([
        "cargo", "metadata", "--format-version", "1", "--no-deps"
    ], text=True))
    members = set(metadata["workspace_members"])
    packages = {p["name"]: p for p in metadata["packages"] if p["id"] in members}
    version = packages["starfield"]["version"]
    errors = []
    expected_packages = {"starfield", "starfield-core", "starfield-catalogs",
                         "starfield-jpl", "starfield-surfaces", "starfield-tools"}
    if set(packages) != expected_packages:
        errors.append(f"expected six grouped packages, found {sorted(packages)}")
    kernels = subprocess.check_output(["git", "ls-files", "*.bsp"], text=True).splitlines()
    if kernels != ["test_data/de421.bsp"]:
        errors.append(f"only test_data/de421.bsp may be checked in; found {kernels}")
    for name, package in packages.items():
        if package["version"] != version:
            errors.append(f"{name}: expected version {version}, got {package['version']}")
        for dependency in package["dependencies"]:
            target = dependency["name"]
            if target not in packages:
                continue
            expected_path = Path(packages[target]["manifest_path"]).parent.resolve()
            actual_path = Path(dependency.get("path") or ".").resolve()
            if dependency["req"] != f"={version}" or actual_path != expected_path or dependency["source"]:
                errors.append(f"{name} -> {target}: must use local path and ={version}")
        if args.packages and package["publish"] != []:
            package_dir = Path(metadata["target_directory"]) / "package"
            filename = f"{name}-{version}.crate"
            archive = package_dir / filename
            # Multi-package dry runs stage archives here while verifying against
            # their temporary registry; Cargo versions differ on the final move.
            if not archive.is_file():
                archive = package_dir / "tmp-crate" / filename
            # Preserve existing offline map tiers while leaving registry headroom.
            budget = 9_500_000 if name == "starfield-surfaces" else 9_000_000
            if not archive.is_file():
                errors.append(f"missing archive: {archive}")
            else:
                size = archive.stat().st_size
                print(f"{name}: {size:,} / {budget:,} bytes")
                if size > budget:
                    errors.append(f"{name}: archive exceeds the {budget:,} byte budget")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(packages)} workspace packages at {version}")


if __name__ == "__main__":
    main()
