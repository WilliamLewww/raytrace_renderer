#pragma once

struct Tuple {
	float x;
	float y;
	float z;
	float w;
};

Tuple operator*(Tuple lhs, float rhs);
Tuple cross(Tuple tupleA, Tuple tupleB);

__host__ __device__
Tuple operator-(Tuple lhs, Tuple rhs);

__host__ __device__
float magnitude(Tuple tuple);

__host__ __device__
Tuple normalize(Tuple tuple);