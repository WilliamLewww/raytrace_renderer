#pragma once
#include <cmath>
#include "ray.h"
#include "transform.h"

struct Camera {
	float viewWidth;
	float viewHeight;
	float fieldOfView;

	float halfWidth;
	float halfHeight;
	float pixelSize;

	Matrix viewMatrix;
	Matrix inverseViewMatrix;
};

Camera createCamera(float viewWidth, float viewHeight, float fieldOfView);
Matrix createViewMatrix(Tuple from, Tuple to, Tuple up);
void rayForPixel(Ray* rayOut, Camera camera);

__global__ void rayForPixelKernel(Ray* rayBuffer, float* inverseViewMatrixBuffer, Camera camera);