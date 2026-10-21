#pragma once

#include <cstdint>
#include <vector>

#include "linear_math.hpp"

// Одна вершина: позиция и нормаль (interleaved — атрибуты лежат подряд, лекция 3).
// Нормаль — единичный вектор, перпендикулярный поверхности в этой точке.
// Она нужна для освещения: от угла между нормалью и направлением на свет
// зависит, насколько ярко освещена точка (закон Ламберта, лекция 4)
struct Vertex {
	Vec3 position;
	Vec3 normal;
};

// Все генераторы ниже строят треугольники так, что векторное произведение
// cross(b - a, c - a) смотрит наружу. В нашем мире (y вверх, камера смотрит вдоль +z)
// такие треугольники видны на экране по часовой стрелке (VK_FRONT_FACE_CLOCKWISE)

// Генерирует сетку тора.
// major_radius — радиус большой окружности (от центра тора до центра трубки),
// minor_radius — радиус трубки,
// major_segments, minor_segments — на сколько частей делим большую и малую окружности
void generateTorus(float major_radius, float minor_radius,
                   uint32_t major_segments, uint32_t minor_segments,
                   std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

// Генерирует UV-сферу радиуса radius: slices делений по долготе, stacks — по широте
void generateSphere(float radius, uint32_t slices, uint32_t stacks,
                    std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);

// Генерирует квадрат size x size в плоскости XZ с центром в начале координат,
// нормаль смотрит вверх (+y). Используется как пол
void generatePlane(float size, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices);
