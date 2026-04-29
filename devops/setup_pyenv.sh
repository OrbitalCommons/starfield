#!/bin/bash
set -e

# Sets up a pyenv-managed Python interpreter and installs the reference
# packages used by `cargo test --features python-tests`.
#
# Note on venvs: previous versions of this script created a pyenv-virtualenv
# named "starfield" and installed packages into it. PyO3's embedded
# interpreter (used by `src/pybridge`) does not honor pyenv-virtualenv at
# runtime — it links against the *base* `libpython` and looks up packages
# in that base interpreter's `site-packages`. So packages must live in the
# base interpreter, not a venv. The venv is removed here.

# Configuration variables
PYENV_ROOT="$HOME/.pyenv"
PYTHON_VERSION=$(cat "$(dirname "$0")/../.python-version")
SKYFIELD_VERSION=$(cat "$(dirname "$0")/../.skyfield-version")
ASTROPY_VERSION=$(cat "$(dirname "$0")/../.astropy-version")

echo "=== Setting up pyenv environment for Starfield ==="

# Add pyenv to PATH if it's installed but not yet visible in this shell
# (running this script via `bash devops/setup_pyenv.sh` from a fresh
# session won't have sourced ~/.bashrc, so command -v pyenv would
# falsely report missing and try to reinstall over a working install).
if [ -d "$PYENV_ROOT" ] && ! command -v pyenv &> /dev/null; then
    export PATH="$PYENV_ROOT/bin:$PATH"
fi

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Pyenv not found. Installing pyenv..."
    curl https://pyenv.run | bash

    # Add pyenv to PATH and initialize for current session
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    # Add pyenv to .bashrc if not already there
    if ! grep -q "pyenv init" "$HOME/.bashrc"; then
        echo "Adding pyenv to .bashrc..."
        echo '' >> "$HOME/.bashrc"
        echo '# pyenv setup' >> "$HOME/.bashrc"
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$HOME/.bashrc"
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$HOME/.bashrc"
        echo 'eval "$(pyenv init --path)"' >> "$HOME/.bashrc"
        echo 'eval "$(pyenv init -)"' >> "$HOME/.bashrc"
        echo 'eval "$(pyenv virtualenv-init -)"' >> "$HOME/.bashrc"
    fi
else
    echo "Pyenv already installed."
    # Add to PATH for current session if not already there
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

# Install Python version via pyenv if not already installed.
# `pyenv versions --bare` prints one version-name per line so we can match
# exactly (the previous `pyenv versions | grep -q` would match substrings,
# including the "(set by .../starfield/...)" annotation, which silently
# masked missing installs).
if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
else
    echo "Python $PYTHON_VERSION already installed."
fi

# Pin the project to the base interpreter — packages get installed there.
echo "Setting Python $PYTHON_VERSION as local interpreter..."
pyenv local "$PYTHON_VERSION"

# Path to the base interpreter (not a shim — pip installs need the real exe).
PYTHON_EXEC="$PYENV_ROOT/versions/$PYTHON_VERSION/bin/python"

# Install required packages into the base interpreter.
echo "Installing required Python packages..."
"$PYTHON_EXEC" -m pip install --upgrade pip
"$PYTHON_EXEC" -m pip install \
    "skyfield==${SKYFIELD_VERSION}" \
    "astropy==${ASTROPY_VERSION}" \
    scipy \
    pytest

# Record the interpreter path for CI / dotenv loaders.
echo "$PYTHON_EXEC" > .python_path

# Create environment variables file for CI. LD_LIBRARY_PATH points at the
# pyenv install's lib dir so the test binary (linked against
# libpython3.10.so.1.0) can find it at runtime.
echo "Creating environment variables file for CI..."
{
    echo "PYO3_PYTHON=$PYTHON_EXEC"
    echo "PYTHON_SYS_EXECUTABLE=$PYTHON_EXEC"
    echo "PYTHON_COMMAND=$PYTHON_EXEC"
    echo "LD_LIBRARY_PATH=$PYENV_ROOT/versions/$PYTHON_VERSION/lib"
    echo "PYTHONPATH=$(pwd)"
} > .env.python

echo "Environment variables stored in .env.python"

echo ""
echo "=== Python environment setup complete ==="
echo ""
echo "Python $PYTHON_VERSION is now the local interpreter for this directory,"
echo "with skyfield $SKYFIELD_VERSION, astropy $ASTROPY_VERSION, and scipy installed."
