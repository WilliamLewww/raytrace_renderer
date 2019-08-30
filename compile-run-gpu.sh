nvcc ./src_GPU/main.cu -o ./bin/raytrace_renderer_gpu
(cd dump && ../bin/raytrace_renderer_gpu image.ppm)