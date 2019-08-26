#pragma once
#include "tuple.h"

struct PointLight {
	Tuple position;
	Tuple intensity;
};

struct Material {
	Tuple color;
	float ambient;
	float diffuse;
	float specular;
	float shininess;
};

PointLight createPointLight(Tuple position, Tuple intensity) {
	return { position, intensity };
}

Material createMaterial() {
	return { createColor(1, 1, 1), 0.1, 0.9, 0.9, 200.0 };
}