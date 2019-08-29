#pragma once
#include "tuple.h"

enum PATTERNS { MATERIAL_PATTERNS_NONE, MATERIAL_PATTERNS_STRIPE };

struct Pattern {
	PATTERNS pattern;
	Matrix transform;

	Tuple* colors;
};

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

	Pattern pattern;
};

PointLight createPointLight(Tuple position, Tuple intensity) {
	return { position, intensity };
}

Material createMaterial(PATTERNS pattern = MATERIAL_PATTERNS_NONE) {
	if (pattern == MATERIAL_PATTERNS_STRIPE) {
		Material material = { createColor(1, 1, 1), 0.1, 0.9, 0.9, 200.0, MATERIAL_PATTERNS_STRIPE };

		material.pattern = { MATERIAL_PATTERNS_STRIPE, createIdentityMatrix(4) };
		material.pattern.colors = new Tuple[2];
		material.pattern.colors[0] = createColor(0, 0, 0);
		material.pattern.colors[1] = createColor(1, 1, 1);

		return material;
	}

	return { createColor(1, 1, 1), 0.1, 0.9, 0.9, 200.0, MATERIAL_PATTERNS_NONE };
}

Tuple stripeAt(Tuple* patternColors, Tuple point) {
	if (int(point.x) % 2 == 0) {
		return patternColors[0];
	}

	return patternColors[1];
}

Tuple lighting(Material material, PointLight light, Tuple point, Tuple eyeV, Tuple normalV, bool inShadow) {
	Tuple ambient;
	Tuple diffuse;
	Tuple specular;

	Tuple materialColor;
	if (material.pattern.pattern == MATERIAL_PATTERNS_NONE) {
		materialColor = material.color;
	}
	if (material.pattern.pattern == MATERIAL_PATTERNS_STRIPE) {
		materialColor = stripeAt(material.pattern.colors, point);
	}

	Tuple effectiveColor = hadamardProduct(materialColor, light.intensity);
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