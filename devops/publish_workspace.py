#!/usr/bin/env python3
"""Resume a workspace publication after crates.io's explicit rate-limit delay.

Only CI calls this helper. Other failures stay fatal; successful uploads are
removed from the next attempt instead of trying to overwrite registry versions.
"""

import email.utils
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


def retry_delay(output, now):
    """Return the server's requested delay, or None for a non-retryable error."""
    if "status 429 Too Many Requests" not in output:
        return None
    match = re.search(r"Please try again after (.*?) GMT", output)
    if not match:
        return None
    deadline = email.utils.parsedate_to_datetime(match.group(1) + " GMT").timestamp()
    return max(1, deadline - now + 5)


def remaining(packages, versions):
    """Check exact versions; unexpected registry errors must not become uploads."""
    pending = []
    for name in packages:
        request = urllib.request.Request(
            f"https://crates.io/api/v1/crates/{name}/{versions[name]}",
            headers={"User-Agent": "starfield-publish-ci (https://github.com/OrbitalCommons/starfield)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60):
                pass
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            pending.append(name)
        time.sleep(1)
    return pending


def upload(packages):
    """Stream Cargo diagnostics while retaining the server's retry instruction."""
    args = ["cargo", "publish"]
    for package in packages:
        args.extend(["-p", package])
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output.append(line)
    return process.wait(), "".join(output)


def publish(packages, versions):
    """Allow at most one retry per initial package, respecting registry pacing."""
    max_attempts = len(packages) + 1
    for attempt in range(max_attempts):
        code, output = upload(packages)
        if code == 0:
            return 0
        delay = retry_delay(output, time.time())
        if delay is None or attempt == max_attempts - 1:
            return code
        print(f"Registry rate limit: retrying after {delay:.0f}s", flush=True)
        time.sleep(delay)
        packages = remaining(packages, versions)
        if not packages:
            return 0
    return 1


if __name__ == "__main__":
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"], text=True
    ))
    versions = {p["name"]: p["version"] for p in metadata["packages"]}
    sys.exit(publish(sys.argv[1:], versions))
