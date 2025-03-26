# These targets do not construct files named after the target itself. Rerun them every time.
.PHONY: setup build test dist/main lint

# This first target is the one that runs by default.
dist/archive.tar.gz: dist/main
	tar -czvf dist/archive.tar.gz dist/main first_run.sh

setup:
	./setup.sh

build: dist/main

test:
	. .venv/bin/activate && PYTHONPATH=./src pytest

dist/main:
	. .venv/bin/activate && python -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data="./src:src" src/main.py

# We disable the following checks:
# C0114: module doesn't have a docstring
# C0116: functions don't have docstrings
# E1101: the linter can't figure out that the cv2 module has functions inside of it
# W0201: attributes are defined outside of __init__
lint:
	. .venv/bin/activate && pylint --disable=C0114,C0116,E1101,W0201 src/
