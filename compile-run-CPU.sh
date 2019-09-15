if [ ! -d bin ]
then
	mkdir ./bin
fi

if [ ! -d dump ]
then
	mkdir ./dump
fi

g++ ./src_CPU/main.cpp -o ./bin/raytrace_renderer_cpu.out
(cd dump && ../bin/raytrace_renderer_cpu.out image.ppm)