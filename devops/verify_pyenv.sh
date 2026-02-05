#!/bin/bash

# Verification script to check that the starfield pyenv is active

# Configuration
EXPECTED_ENV_NAME=$(cat "$(dirname "$0")/../.python-version")
EXPECTED_PYTHON_VERSION=$EXPECTED_ENV_NAME

# Text formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color
CHECK="✓"
CROSS="✗"

echo "=== Verifying Starfield Python Environment ==="

# Check if pyenv is installed and in PATH
if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}${CROSS} Pyenv is not installed or not in PATH${NC}"
    echo "Run devops/setup_pyenv.sh to set up the environment"
    exit 1
else
    echo -e "${GREEN}${CHECK} Pyenv installed${NC}"
fi

# Check current pyenv version
CURRENT_VERSION=$(pyenv version-name)
if [[ "$CURRENT_VERSION" == "$EXPECTED_ENV_NAME" ]]; then
    echo -e "${GREEN}${CHECK} Correct pyenv environment active: $CURRENT_VERSION${NC}"
else
    echo -e "${RED}${CROSS} Wrong pyenv environment active: $CURRENT_VERSION (expected: $EXPECTED_ENV_NAME)${NC}"
    echo "Run 'pyenv activate $EXPECTED_ENV_NAME' or cd into the project directory"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2)
if [[ "$PYTHON_VERSION" == "$EXPECTED_PYTHON_VERSION"* ]]; then
    echo -e "${GREEN}${CHECK} Correct Python version: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}${CROSS} Wrong Python version: $PYTHON_VERSION (expected: $EXPECTED_PYTHON_VERSION)${NC}"
    exit 1
fi

# Check if skyfield is installed
if ! python -c "import skyfield" &> /dev/null; then
    echo -e "${RED}${CROSS} Skyfield is not installed${NC}"
    exit 1
fi

# Check skyfield version
SKYFIELD_VERSION=$(python -c "import skyfield; print(skyfield.__version__)")
EXPECTED_SKYFIELD_VERSION=$(cat "$(dirname "$0")/../.skyfield-version")

echo -e "${GREEN}${CHECK} Skyfield installed: v$SKYFIELD_VERSION${NC}"

if [[ "$SKYFIELD_VERSION" != "$EXPECTED_SKYFIELD_VERSION" ]]; then
    echo -e "${RED}${CROSS} Skyfield version mismatch: found v$SKYFIELD_VERSION, expected v$EXPECTED_SKYFIELD_VERSION${NC}"
    exit 1
fi

# Check if pytest is installed
if python -c "import pytest" &> /dev/null; then
    PYTEST_VERSION=$(python -c "import pytest; print(pytest.__version__)")
    echo -e "${GREEN}${CHECK} Pytest installed: v$PYTEST_VERSION${NC}"
else
    echo -e "${RED}${CROSS} Pytest is not installed${NC}"
    echo "Run 'pip install skyfield pytest' to install required packages"
    exit 1
fi

# Check for .python-version file
if [ -f .python-version ]; then
    PYENV_FILE=$(cat .python-version)
    if [[ "$PYENV_FILE" == "$EXPECTED_ENV_NAME" ]]; then
        echo -e "${GREEN}${CHECK} Correct .python-version file: $PYENV_FILE${NC}"
    else
        echo -e "${YELLOW}Warning: .python-version file exists but contains: $PYENV_FILE (expected: $EXPECTED_ENV_NAME)${NC}"
    fi
else
    echo -e "${RED}${CROSS} No .python-version file found${NC}"
    echo "Run devops/setup_pyenv.sh to set up the environment"
    exit 1
fi

# Display current Python executable path
PYTHON_PATH=$(which python)
echo -e "${GREEN}${CHECK} Using Python at: $PYTHON_PATH${NC}"

echo ""
echo -e "${GREEN}Environment verification complete. All checks passed!${NC}"
echo "Starfield Python environment is correctly set up and activated."