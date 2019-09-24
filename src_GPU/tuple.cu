#include "tuple.h"

Tuple operator*(Tuple lhs, float rhs) {
	return { lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs };
}

__host__ __device__
Tuple operator-(Tuple lhs, Tuple rhs) {
	return { lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w };
}

__host__ __device__
float magnitude(Tuple tuple) {
	return sqrt(pow(tuple.x, 2) + pow(tuple.y, 2) + pow(tuple.z, 2) + pow(tuple.w, 2));
}

__host__ __device__
Tuple normalize(Tuple tuple) {
	return { tuple.x / magnitude(tuple), tuple.y / magnitude(tuple), tuple.z / magnitude(tuple), tuple.w / magnitude(tuple) };
}

Tuple cross(Tuple tupleA, Tuple tupleB) {
	return { (tupleA.y * tupleB.z) - (tupleA.z * tupleB.y), (tupleA.z * tupleB.x) - (tupleA.x * tupleB.z), (tupleA.x * tupleB.y) - (tupleA.y * tupleB.x) };
}

__device__
float dot(Tuple tupleA, Tuple tupleB) {
	return ((tupleA.x * tupleB.x) + (tupleA.y * tupleB.y) + (tupleA.z * tupleB.z) + (tupleA.w * tupleB.w));
}