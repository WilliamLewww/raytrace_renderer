CC=g++
NVCC=/usr/local/cuda-10.1/bin/nvcc
NVPROF=/usr/local/cuda-10.1/bin/nvprof
MEMCHECK=/usr/local/cuda-10.1/bin/cuda-memcheck
CXXFLAGS=
CUDAFLAGS=-m64 -gencode arch=compute_30,code=compute_30
LIBS=
LIBDIRS=
INCDIRS=

all: clean compile

clean:
	rm -rf bin
	rm -rf dump

compile:
	mkdir -p bin
	$(NVCC) ./src_GPU/main.cu ./src_GPU/src/*.cu -o ./bin/raytrace_renderer_gpu.out $(CUDAFLAGS)

memory-check:
	mkdir -p dump
	cd dump; \
	$(MEMCHECK) ../bin/raytrace_renderer_gpu.out image.ppm runtime.log

profile:
	mkdir -p dump
	cd dump; \
	sudo $(NVPROF) ../bin/raytrace_renderer_gpu.out image.ppm runtime.log