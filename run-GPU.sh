export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

if [ ! -d dump ]
then
	mkdir ./dump
fi

(cd dump && ../bin/raytrace_renderer_gpu.out image.ppm runtime.log)