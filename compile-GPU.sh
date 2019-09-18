export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

if [ ! -d bin ]
then
	mkdir ./bin
fi

nvcc ./src_GPU/main.cu ./src_GPU/tuple.cu -o ./bin/raytrace_renderer_gpu.out -m64 -gencode arch=compute_30,code=compute_30 -lcublas