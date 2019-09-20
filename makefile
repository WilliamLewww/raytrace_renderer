CC=g++
NVCC=/usr/local/cuda-10.1/bin/nvcc
NVPROF=/usr/local/cuda-10.1/bin/nvprof
MEMCHECK=/usr/local/cuda-10.1/bin/cuda-memcheck
CXXFLAGS=
CUDAFLAGS=-m64 -gencode arch=compute_30,code=compute_30 -rdc=true
LIBS=
LIBDIRS=
INCDIRS=

all: clean compile run

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	$(NVCC) ./src_GPU/main.cu ./src_GPU/src/*.cu -o ./bin/raytrace_renderer_gpu.out $(CUDAFLAGS)

run:
	mkdir -p dump
	cd dump; \
	../bin/raytrace_renderer_gpu.out image.ppm runtime.log

memory-check:
	mkdir -p dump
	cd dump; \
	$(MEMCHECK) ../bin/raytrace_renderer_gpu.out image.ppm runtime.log

profile:
	mkdir -p dump
	cd dump; \
	sudo $(NVPROF) ../bin/raytrace_renderer_gpu.out image.ppm runtime.log

all-cpu:
	mkdir -p bin
	$(CC) ./src_CPU/main.cpp -o ./bin/raytrace_renderer_cpu.out
	mkdir -p dump
	cd dump; \
	../bin/raytrace_renderer_cpu.out image.ppm