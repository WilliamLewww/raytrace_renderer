CC=g++

IMAGENAME=image.ppm
OUTPUTNAME=raytrace_renderer_cpu.out

all: clean compile run

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	$(CC) ./src/main.cpp -o ./bin/$(OUTPUTNAME)

run:
	mkdir -p dump
	cd dump; ../bin/$(OUTPUTNAME) $(IMAGENAME)

open:
	cd dump; xdg-open $(IMAGENAME)