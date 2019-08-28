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

Tuple lighting(Material material, PointLight light, Tuple point, Tuple eyeV, Tuple normalV, bool inShadow) {
	Tuple ambient;
	Tuple diffuse;
	Tuple specular;

	Tuple effectiveColor = hadamardProduct(material.color, light.intensity);
	Tuple lightV = normalize(light.position - point);

	ambient = effectiveColor * material.ambient;

	float lightDotNormal = dot(lightV, normalV);
	if (lightDotNormal < 0 || inShadow) {
		diffuse = createColor(0.0, 0.0, 0.0);
		specular = createColor(0.0, 0.0, 0.0);
	}
	else {
		diffuse = effectiveColor * material.diffuse * lightDotNormal;

		Tuple reflectV = reflect(negate(lightV), normalV);
		float reflectDotEye = dot(reflectV, eyeV);

		if (reflectDotEye <= 0) {
			specular = createColor(0.0, 0.0, 0.0);
		}
		else {
			float factor = pow(reflectDotEye, material.shininess);
			specular = light.intensity * material.specular * factor;
		}
	}

	return ambient + diffuse + specular;
}