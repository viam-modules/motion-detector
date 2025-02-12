# These targets do not construct files named after the target itself. Rerun them every time.
.PHONY: setup build test dist/main lint

setup:
	./build.sh

build: dist/main

test:
	. .venv/bin/activate && PYTHONPATH=./src pytest

dist/main:
	. .venv/bin/activate && python -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data="./src:src" src/main.py

dist/archive.tar.gz: dist/main
	tar -czvf dist/archive.tar.gz dist/main

lint:
	. .venv/bin/activate && pylint --disable=C0114,E0401,E1101,C0116,W0613,R0913,C0116,R0914,C0103,W0201,W0719 src/

