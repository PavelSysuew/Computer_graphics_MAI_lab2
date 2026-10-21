#include "geometry.hpp"

#include <cmath>

namespace {

constexpr float pi = 3.14159265358979f;

// Индексы для сетки rows x columns вершин (вершина (i, j) лежит в i * columns + j).
// Каждую клетку сетки делим на два треугольника.
// wrap_rows = true замыкает сетку по i: после последней строки снова идёт первая (как у тора)
void appendGridIndices(uint32_t rows, uint32_t columns, bool wrap_rows, bool wrap_columns,
                       std::vector<uint32_t>& indices) {
	const uint32_t row_cells = wrap_rows ? rows : rows - 1;
	const uint32_t column_cells = wrap_columns ? columns : columns - 1;

	for (uint32_t i = 0; i < row_cells; ++i) {
		const uint32_t i_next = (i + 1) % rows;

		for (uint32_t j = 0; j < column_cells; ++j) {
			const uint32_t j_next = (j + 1) % columns;

			const uint32_t p00 = i * columns + j;
			const uint32_t p10 = i_next * columns + j;
			const uint32_t p01 = i * columns + j_next;
			const uint32_t p11 = i_next * columns + j_next;

			// Порядок подобран так, что cross(p01 - p00, p10 - p00) смотрит наружу
			indices.push_back(p00);
			indices.push_back(p01);
			indices.push_back(p10);

			indices.push_back(p10);
			indices.push_back(p01);
			indices.push_back(p11);
		}
	}
}

} // namespace

void generateTorus(float major_radius, float minor_radius,
                   uint32_t major_segments, uint32_t minor_segments,
                   std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	vertices.clear();
	indices.clear();

	// Вершины: сетка по двум углам.
	// u — угол по большой окружности (вокруг оси Y),
	// v — угол по малой окружности (вокруг трубки)
	for (uint32_t i = 0; i < major_segments; ++i) {
		const float u = 2.0f * pi * float(i) / float(major_segments);

		for (uint32_t j = 0; j < minor_segments; ++j) {
			const float v = 2.0f * pi * float(j) / float(minor_segments);

			// Расстояние от оси Y до точки
			const float ring = major_radius + minor_radius * std::cos(v);

			const Vec3 position = {
				ring * std::cos(u),
				minor_radius * std::sin(v),
				ring * std::sin(u),
			};

			// Процедурная нормаль: направление от центра трубки к точке поверхности.
			// Центр трубки лежит на большой окружности: (R cos u, 0, R sin u).
			// Вычитаем его из позиции и делим на minor_radius — получаем
			// (cos v cos u, sin v, cos v sin u), длина этого вектора уже равна 1
			const Vec3 normal = {
				std::cos(v) * std::cos(u),
				std::sin(v),
				std::cos(v) * std::sin(u),
			};

			vertices.push_back({ position, normal });
		}
	}

	// Тор замкнут в обе стороны: и по u, и по v
	appendGridIndices(major_segments, minor_segments, true, true, indices);
}

void generateSphere(float radius, uint32_t slices, uint32_t stacks,
                    std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	vertices.clear();
	indices.clear();

	// Сфера — это тор с нулевым большим радиусом, у которого угол v идёт
	// только от южного полюса (-90°) до северного (+90°).
	// u — долгота (вокруг оси Y), v — широта
	for (uint32_t i = 0; i < slices; ++i) {
		const float u = 2.0f * pi * float(i) / float(slices);

		for (uint32_t j = 0; j <= stacks; ++j) {
			const float v = -0.5f * pi + pi * float(j) / float(stacks);

			// У сферы с центром в начале координат нормаль — это просто
			// направление из центра в точку
			const Vec3 normal = {
				std::cos(v) * std::cos(u),
				std::sin(v),
				std::cos(v) * std::sin(u),
			};

			vertices.push_back({ normal * radius, normal });
		}
	}

	// По долготе сетка замкнута, по широте — нет (между полюсами шов не нужен)
	appendGridIndices(slices, stacks + 1, true, false, indices);
}

void generatePlane(float size, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) {
	const float h = 0.5f * size;
	const Vec3 up = { 0.0f, 1.0f, 0.0f };

	// Освещение считается для каждого пикселя, поэтому пол может состоять всего
	// из двух треугольников: нормаль одна и та же во всех точках
	vertices = {
		{ { -h, 0.0f, -h }, up },
		{ { -h, 0.0f,  h }, up },
		{ {  h, 0.0f, -h }, up },
		{ {  h, 0.0f,  h }, up },
	};

	// cross(v1 - v0, v2 - v0) = cross((0, 0, 2h), (2h, 0, 0)) = (0, 4h², 0) — смотрит вверх
	indices = {
		0, 1, 2,
		2, 1, 3,
	};
}
