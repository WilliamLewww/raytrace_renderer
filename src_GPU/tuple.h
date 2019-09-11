#pragma once

struct Tuple {
	float x;
	float y;
	float z;
	float w;
};

__host__ __device__
Tuple operator*(Tuple tuple, float scalar) {
	return { tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar };
}