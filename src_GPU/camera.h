#pragma once
#include <cmath>
#include "ray.h"
#include "matrix.h"

struct Camera {
	float viewWidth;
	float viewHeight;
	float fieldOfView;

	Matrix viewMatrix;

	float halfWidth;
	float halfHeight;
	float pixelSize;
};

Camera createCamera(float viewWidth, float viewHeight, float fieldOfView) {
	Camera camera = { viewWidth, viewHeight, fieldOfView, createIdentityMatrix(4) };

	float halfView = tan(camera.fieldOfView / 2);
	float aspect = camera.viewWidth / camera.viewHeight;

	if (aspect >= 1) {
		camera.halfWidth = halfView;
		camera.halfHeight = halfView / aspect;
	}
	else {
		camera.halfWidth = halfView * aspect;
		camera.halfHeight = halfView;
	}

	camera.pixelSize = (camera.halfWidth * 2) / camera.viewWidth;

	return camera;
}