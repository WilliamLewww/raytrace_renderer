if [ ! -d dump ]
then
	mkdir ./dump
fi

(cd dump && nvprof ../bin/raytrace_renderer_gpu.out image.ppm runtime.log)