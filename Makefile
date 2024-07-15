setup:
	./build.sh
	
test:
	PYTHONPATH=./src pytest
	
dist/archive.tar.gz:
	tar -czvf dist/archive.tar.gz dist/main

lint:
	pylint --disable=C0114,E0401,E1101,C0116,W0613,R0913,C0116,R0914,C0103,W0201,W0719 src/

