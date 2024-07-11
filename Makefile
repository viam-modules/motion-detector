build:
	./build.sh
	
test:
	PYTHONPATH=./src pytest
	
dist/archive.tar.gz:
	tar -czvf dist/archive.tar.gz dist/main
