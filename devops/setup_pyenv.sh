#!/bin/bash
set -e

# Configuration variables
PYENV_ROOT="$HOME/.pyenv"
PYTHON_VERSION=$(cat "$(dirname "$0")/../.python-version")
PYENV_ENV_NAME="starfield"

echo "=== Setting up pyenv environment for Starfield ==="

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

# Install Python version via pyenv if not already installed
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION..."
    pyenv install $PYTHON_VERSION
else
    echo "Python $PYTHON_VERSION already installed."
fi

# Create or update pyenv environment
if ! pyenv versions | grep -q "$PYENV_ENV_NAME"; then
    echo "Creating pyenv environment '$PYENV_ENV_NAME'..."
    pyenv virtualenv $PYTHON_VERSION $PYENV_ENV_NAME
else
    echo "Pyenv environment '$PYENV_ENV_NAME' already exists."
fi

# Set local Python version to our environment
echo "Setting '$PYENV_ENV_NAME' as local Python environment..."
pyenv local $PYENV_ENV_NAME

# Install required packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install skyfield pytest

# Write the Python path to a file for CI
PYTHON_EXEC=$(which python)
echo "$PYTHON_EXEC" > .python_path

# Create environment variables file for CI
echo "Creating environment variables file for CI..."
{
    echo "PYO3_PYTHON=$PYTHON_EXEC"
    echo "PYTHONPATH=$(pwd)"
    echo "PYTHON_SYS_EXECUTABLE=$PYTHON_EXEC"
    echo "PYTHON_COMMAND=$PYTHON_EXEC"
} > .env.python

echo "Environment variables stored in .env.python"

echo ""
echo "=== Python environment setup complete ==="
echo ""
echo "The pyenv environment '$PYENV_ENV_NAME' is now configured with Python $PYTHON_VERSION"
echo "It is set as the local Python version for this directory."
echo ""
echo "To activate manually (should be automatic in this directory):"
echo "  pyenv activate $PYENV_ENV_NAME"