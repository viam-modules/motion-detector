#!/usr/bin/env bash
# setup.sh -- environment bootstrapper for python virtualenv

# x: print out commands as they are run
# e: exit on any failure
# u: using a nonexistent environment variable is an error
# o pipefail: in a pipeline, any intermediate step exiting with failure counts as an overall fail
set -xeuo pipefail

SUDO=sudo
if ! command -v $SUDO; then
    echo no sudo on this system, proceeding as current user
    SUDO=""
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

if command -v apt-get; then
    $SUDO apt-get update
    $SUDO apt-get -y install python3-venv
    if dpkg -l python3-venv; then
        echo "python3-venv is installed, skipping setup"
    else
        if ! apt info python3-venv; then
            echo "python3-venv package info not found, trying apt update"
            $SUDO apt-get -qq update
        fi
        $SUDO apt-get install -qqy python3-venv
    fi
else
    echo "Skipping tool installation because your platform is missing apt-get."
    echo "If you see failures below, install the equivalent of python3-venv for your system."
fi

source .env
echo creating virtualenv at $VIRTUAL_ENV
python3 -m venv $VIRTUAL_ENV
echo installing dependencies from requirements.txt
$VIRTUAL_ENV/bin/pip install -r requirements.txt -U

# When running as a local module, we need meta.json to be in the same directory as the module.
mkdir -p dist
ln -sf ../meta.json dist
