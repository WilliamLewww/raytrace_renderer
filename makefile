CC=g++
NVCC=/usr/local/cuda-10.1/bin/nvcc
NVPROF=/usr/local/cuda-10.1/bin/nvprof
MEMCHECK=/usr/local/cuda-10.1/bin/cuda-memcheck
NSIGHTCLI=/usr/local/cuda-10.1/bin/nv-nsight-cu-cli
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
	cd dump; ../bin/raytrace_renderer_cpu.out image.ppm

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	cd bin; $(NVCC) $(CUDAFLAGS) --device-c ../src_GPU/*.cu
	cd bin; $(NVCC) $(CUDAFLAGS) *.o -o raytrace_renderer_gpu.out

run:
	mkdir -p dump
	cd dump; ../bin/raytrace_renderer_gpu.out image.ppm

memory-check:
	mkdir -p dump
	cd dump; $(MEMCHECK) ../bin/raytrace_renderer_gpu.out image.ppm

profile:
	mkdir -p dump
	cd dump; sudo $(NVPROF) ../bin/raytrace_renderer_gpu.out image.ppm 2>profile.log; cat profile.log;

profile-metrics:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --metrics all ../bin/raytrace_renderer_gpu.out image.ppm 2>profile-metrics.log; cat profile-metrics.log;

profile-events:
	mkdir -p dump
	cd dump; sudo $(NVPROF) --events all ../bin/raytrace_renderer_gpu.out image.ppm 2>profile-events.log; cat profile-events.log;

nsight-cli:
	mkdir -p dump
	cd dump; sudo $(NSIGHTCLI) ../bin/raytrace_renderer_gpu.out image.ppm