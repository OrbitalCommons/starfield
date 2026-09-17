#!/usr/bin/env python
import sys

try:
    import skyfield
    print(f"Python Version: {sys.version}")
    print(f"Skyfield version: {skyfield.__version__}")
except ImportError as e:
    print(f"Error importing skyfield: {e}")
    sys.exit(1)