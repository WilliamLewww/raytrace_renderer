if [ ! -d bin ]
then
	mkdir ./bin
fi

nvcc ./src_GPU/main.cu -o ./bin/raytrace_renderer_gpu.out