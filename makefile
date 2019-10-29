CC=g++

IMAGENAME=image.ppm

all: clean compile run

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	$(CC) ./src/main.cpp -o ./bin/raytrace_renderer_cpu.out

run:
	mkdir -p dump
	cd dump; ../bin/raytrace_renderer_cpu.out $(IMAGENAME)

open:
	cd dump; xdg-open $(IMAGENAME)