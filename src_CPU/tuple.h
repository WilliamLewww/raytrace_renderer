#pragma once
#include <iostream>
#include <cmath>

const float EPSILON_TUPLE = 0.00001;

struct Tuple {
	float x;
	float y;
	float z;
	float w;
};

bool operator==(Tuple tupleA, Tuple tupleB) {
	return (abs(tupleA.x - tupleB.x) < EPSILON_TUPLE && abs(tupleA.y - tupleB.y) < EPSILON_TUPLE && abs(tupleA.z - tupleB.z) < EPSILON_TUPLE && abs(tupleA.w - tupleB.w) < EPSILON_TUPLE);
}

std::ostream& operator<<(std::ostream& os, const Tuple& tuple) {
    os << "(" << tuple.x << ", " << tuple.y << ", " << tuple.z << ", " << tuple.w << ")";
    return os;
}

Tuple createVector(float x = 0.0, float y = 0.0, float z = 0.0) {
	return { x, y, z, 0.0 };
}

Tuple createPoint(float x = 0.0, float y = 0.0, float z = 0.0) {
	return { x, y, z, 1.0 };
}

Tuple createColor(float x = 0.0, float y = 0.0, float z = 0.0) {
	return { x, y, z, 1.0 };
}

Tuple operator+(Tuple tupleA, Tuple tupleB) {
	return { tupleA.x + tupleB.x, tupleA.y + tupleB.y, tupleA.z + tupleB.z, tupleA.w + tupleB.w };
}

Tuple operator-(Tuple tupleA, Tuple tupleB) {
	return { tupleA.x - tupleB.x, tupleA.y - tupleB.y, tupleA.z - tupleB.z, tupleA.w - tupleB.w };
}

Tuple negate(Tuple tuple) {
	return { -tuple.x, -tuple.y, -tuple.z, -tuple.w };
}

Tuple operator*(Tuple tuple, float scalar) {
	return { tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar };
}

Tuple operator*(float scalar, Tuple tuple) {
	return { tuple.x * scalar, tuple.y * scalar, tuple.z * scalar, tuple.w * scalar };
}

Tuple operator/(Tuple tuple, float scalar) {
	return { tuple.x / scalar, tuple.y / scalar, tuple.z / scalar, tuple.w / scalar };
}

float magnitude(Tuple tuple) {
	return sqrt(pow(tuple.x, 2) + pow(tuple.y, 2) + pow(tuple.z, 2) + pow(tuple.w, 2));
}

Tuple normalize(Tuple tuple) {
	return { tuple.x / magnitude(tuple), tuple.y / magnitude(tuple), tuple.z / magnitude(tuple), tuple.w / magnitude(tuple) };
}

float dot(Tuple tupleA, Tuple tupleB) {
	return ((tupleA.x * tupleB.x) + (tupleA.y * tupleB.y) + (tupleA.z * tupleB.z) + (tupleA.w * tupleB.w));
}

Tuple cross(Tuple tupleA, Tuple tupleB) {
	return { (tupleA.y * tupleB.z) - (tupleA.z * tupleB.y), (tupleA.z * tupleB.x) - (tupleA.x * tupleB.z), (tupleA.x * tupleB.y) - (tupleA.y * tupleB.x) };
}

Tuple hadamardProduct(Tuple tupleA, Tuple tupleB) {
	return { tupleA.x * tupleB.x, tupleA.y * tupleB.y, tupleA.z * tupleB.z, tupleA.w * tupleB.w };
}

Tuple reflect(Tuple vector, Tuple normal) {
	return vector - (normal * 2 * dot(vector, normal));
}