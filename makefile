CC=g++
NVCC=/usr/local/cuda-10.1/bin/nvcc
NVPROF=/usr/local/cuda-10.1/bin/nvprof
MEMCHECK=/usr/local/cuda-10.1/bin/cuda-memcheck
CXXFLAGS=
CUDAFLAGS=--gpu-architecture=sm_50
LIBS=
LIBDIRS=
INCDIRS=

all: clean compile run

all-cpu:
	mkdir -p bin
	$(CC) ./src_CPU/main.cpp -o ./bin/raytrace_renderer_cpu.out
	mkdir -p dump
	cd dump; \
	../bin/raytrace_renderer_cpu.out image.ppm

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	cd bin; \
	$(NVCC) $(CUDAFLAGS) --device-c ../src_GPU/*.cu
	cd bin; \
	$(NVCC) $(CUDAFLAGS) *.o -o raytrace_renderer_gpu.out

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