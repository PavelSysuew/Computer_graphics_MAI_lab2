#include "application.hpp"

#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include "linear_math.hpp"
#include "geometry.hpp"

namespace application {

namespace {

using graphics::internal::context;

// ---------------------------------------------------------------------------
// Ресурсы Vulkan
// ---------------------------------------------------------------------------

// Буфер в памяти видеокарты вместе с выделенной под него памятью
struct Buffer {
	VkBuffer buffer;
	VmaAllocation allocation;
	// Адрес, по которому процессор может писать в этот буфер
	void* mapped;
};

// Сетка (меш): вершины и индексы треугольников
struct Mesh {
	Buffer vertex_buffer;
	Buffer index_buffer;
	uint32_t index_count;
};

// ---------------------------------------------------------------------------
// Структуры, которые копируются в буферы видеокарты.
// Они обязаны байт в байт повторять раскладку блоков в шейдерах.
// В std140 и std430 vec3 выравнивается на 16 байт, поэтому после каждого Vec3
// (12 байт) идёт либо полезный float, либо float-заглушка (лекция 4).
// static_assert проверяет смещения ещё при компиляции
// ---------------------------------------------------------------------------

// Набор 0, binding 0: uniform-буфер сцены (SceneUniforms в lit.vert / lit.frag)
struct SceneUniforms {
	Mat4 view_projection;              // смещение 0
	Vec3 camera_position;              // 64
	float padding0;
	Vec3 directional_light_direction;  // 80
	float padding1;
	Vec3 directional_light_color;      // 96
	float padding2;
	Vec3 ambient_light;                // 112
	// В std140 одиночный uint может занять свободные 4 байта после vec3
	uint32_t point_light_count;        // 124
	uint32_t spot_light_count;         // 128
	float padding3[3];                 // добиваем размер до кратного 16
};
static_assert(offsetof(SceneUniforms, camera_position) == 64);
static_assert(offsetof(SceneUniforms, directional_light_direction) == 80);
static_assert(offsetof(SceneUniforms, directional_light_color) == 96);
static_assert(offsetof(SceneUniforms, ambient_light) == 112);
static_assert(offsetof(SceneUniforms, point_light_count) == 124);
static_assert(offsetof(SceneUniforms, spot_light_count) == 128);

// Набор 1, binding 0: uniform-буфер модели (ModelUniforms в шейдере)
struct ModelUniforms {
	Mat4 model;          // 0
	Mat4 normal_matrix;  // 64
	Vec3 albedo;         // 128
	float padding0;
	Vec3 specular;       // 144
	float shininess;     // 156 — занимает свободное место после vec3
};
static_assert(offsetof(ModelUniforms, albedo) == 128);
static_assert(offsetof(ModelUniforms, specular) == 144);
static_assert(offsetof(ModelUniforms, shininess) == 156);

// Элемент storage-буфера точечных источников (struct PointLight в lit.frag, std430)
struct GpuPointLight {
	Vec3 position;  // 0
	float padding0;
	Vec3 color;     // 16 (цвет * интенсивность)
	float padding1;
};
static_assert(sizeof(GpuPointLight) == 32);

// Элемент storage-буфера прожекторов (struct SpotLight в lit.frag, std430)
struct GpuSpotLight {
	Vec3 position;   // 0
	float inner_cos; // 12
	Vec3 direction;  // 16
	float outer_cos; // 28
	Vec3 color;      // 32 (цвет * интенсивность)
	float padding0;
};
static_assert(sizeof(GpuSpotLight) == 48);

// Push-константы маркера источника (MarkerConstants в marker.vert)
struct MarkerConstants {
	Mat4 model;
	float color[4];
};

// Вершина линии конуса прожектора (входы line.vert)
struct LineVertex {
	Vec3 position;
	Vec3 color;
};

// ---------------------------------------------------------------------------
// Сцена: камера, источники света, объекты
// ---------------------------------------------------------------------------

// Свободная камера: позиция и два угла поворота (лекция 4, матрица вида)
struct Camera {
	Vec3 position;
	// Поворот вокруг вертикальной оси (влево-вправо), в градусах
	float yaw;
	// Наклон вверх-вниз, в градусах. Положительный — смотрим вверх
	float pitch;

	bool use_perspective;
	float fov_degrees;
	float ortho_half_height;
	float near_plane;
	float far_plane;

	// Скорость полёта (единиц в секунду) и чувствительность мыши (градусов на пиксель)
	float move_speed;
	float mouse_sensitivity;
};

// Цвет и интенсивность храним отдельно, чтобы в интерфейсе было удобно их менять.
// В шейдер уходит их произведение
struct AmbientLight {
	float color[3];
	float intensity;
};

struct DirectionalLight {
	bool enabled;
	// Направление, куда летят лучи
	Vec3 direction;
	float color[3];
	float intensity;
};

struct PointLight {
	Vec3 position;
	float color[3];
	float intensity;

	// Анимация: вращение вокруг вертикальной оси мира, градусов в секунду
	bool orbit;
	float orbit_speed;
};

struct SpotLight {
	Vec3 position;
	// Направление, куда светит прожектор
	Vec3 direction;
	float color[3];
	float intensity;
	// Половины углов раствора конуса, в градусах.
	// Внутри inner — полная яркость, между inner и outer — плавное затухание
	float inner_angle;
	float outer_angle;
};

// Материал по модели Блинна–Фонга (лекция 4)
struct Material {
	// K_d — диффузный цвет (альбедо)
	float albedo[3];
	// K_s — цвет блика
	float specular[3];
	// α — коэффициент блеска: чем больше, тем меньше и резче блик
	float shininess;
};

// Один объект на сцене: параметры из интерфейса и свои ресурсы Vulkan
struct Object {
	const char* name;
	bool visible;
	const Mesh* mesh;

	// Положение, поворот (в градусах) и растяжение
	Vec3 position;
	Vec3 rotation;
	Vec3 scale;

	Material material;

	// Анимация из лабы 1: движение по «восьмёрке» с покачиванием вверх-вниз и вращение
	bool animation_playing;
	float animation_speed;
	float trajectory_radius;
	float trajectory_height;
	float spin_speed;
	float animation_time;

	// Свой uniform-буфер и свой набор дескрипторов (набор 1)
	Buffer uniform_buffer;
	VkDescriptorSet descriptor_set;
};

// Три тора из лабы 1 и пол
constexpr uint32_t object_count = 4;

// Параметры сеток
constexpr float torus_major_radius = 1.0f;
constexpr float torus_minor_radius = 0.4f;
constexpr uint32_t torus_major_segments = 64;
constexpr uint32_t torus_minor_segments = 32;
constexpr float floor_size = 30.0f;

// Сколько источников помещается в storage-буфер сразу после запуска.
// Это НЕ ограничение: если источников станет больше, буфер пересоздаётся побольше
constexpr uint32_t initial_light_capacity = 8;

// Сколько отрезков в окружности основания конуса прожектора
constexpr uint32_t cone_circle_segments = 32;
// Сколько образующих рисуем от вершины конуса к основанию
constexpr uint32_t cone_side_lines = 8;

// Раскладки наборов дескрипторов: 0 — сцена, 1 — модель
VkDescriptorSetLayout vk_scene_set_layout;
VkDescriptorSetLayout vk_model_set_layout;

// Раскладка для освещаемых объектов (наборы 0 и 1)
// и для маркеров/линий (набор 0 + push-константы)
VkPipelineLayout vk_lit_pipeline_layout;
VkPipelineLayout vk_unlit_pipeline_layout;

VkPipeline vk_lit_pipeline;
VkPipeline vk_marker_pipeline;
VkPipeline vk_line_pipeline;

VkDescriptorPool vk_descriptor_pool;

Mesh torus_mesh;
Mesh sphere_mesh;
Mesh floor_mesh;

// Набор 0: uniform-буфер сцены и два storage-буфера с источниками
Buffer scene_uniform_buffer;
Buffer point_light_buffer;
Buffer spot_light_buffer;
uint32_t point_light_capacity = 0;
uint32_t spot_light_capacity = 0;
VkDescriptorSet vk_scene_descriptor_set;

// Вершины линий конусов пересчитываются каждый кадр
Buffer line_vertex_buffer;
uint32_t line_vertex_capacity = 0;
uint32_t line_vertex_count = 0;

Camera camera;
AmbientLight ambient_light;
DirectionalLight directional_light;
std::vector<PointLight> point_lights;
std::vector<SpotLight> spot_lights;

Object objects[object_count];
int selected_object = 0;

// Визуализация источников (доп. задание 4)
bool show_light_markers = true;
bool show_spot_cones = true;
float marker_size = 0.12f;
float cone_length = 2.0f;

double last_time = 0.0;

// ---------------------------------------------------------------------------
// Значения по умолчанию
// ---------------------------------------------------------------------------

Camera defaultCamera() {
	return {
		// Стоим сзади и чуть сверху, смотрим вперёд (+z) и немного вниз
		.position = { 0.0f, 3.0f, -10.0f },
		.yaw = 0.0f,
		.pitch = -15.0f,
		.use_perspective = true,
		.fov_degrees = 60.0f,
		.ortho_half_height = 5.0f,
		.near_plane = 0.1f,
		.far_plane = 100.0f,
		.move_speed = 4.0f,
		.mouse_sensitivity = 0.2f,
	};
}

Object makeObject(const char* name, const Mesh* mesh, Vec3 position, Vec3 albedo, bool playing) {
	Object object = {};
	object.name = name;
	object.visible = true;
	object.mesh = mesh;
	object.position = position;
	object.rotation = { 0.0f, 0.0f, 0.0f };
	object.scale = { 1.0f, 1.0f, 1.0f };
	object.material = {
		.albedo = { albedo.x, albedo.y, albedo.z },
		.specular = { 0.6f, 0.6f, 0.6f },
		.shininess = 32.0f,
	};
	object.animation_playing = playing;
	object.animation_speed = 1.0f;
	object.trajectory_radius = 1.5f;
	object.trajectory_height = 0.5f;
	object.spin_speed = 60.0f;
	object.animation_time = 0.0f;
	return object;
}

PointLight makePointLight(Vec3 position, Vec3 color, float intensity) {
	return {
		.position = position,
		.color = { color.x, color.y, color.z },
		.intensity = intensity,
		.orbit = false,
		.orbit_speed = 45.0f,
	};
}

SpotLight makeSpotLight(Vec3 position, Vec3 direction, Vec3 color, float intensity) {
	return {
		.position = position,
		.direction = direction,
		.color = { color.x, color.y, color.z },
		.intensity = intensity,
		.inner_angle = 15.0f,
		.outer_angle = 25.0f,
	};
}

// Начальная сцена
void setupScene() {
	camera = defaultCamera();

	ambient_light = {
		.color = { 1.0f, 1.0f, 1.0f },
		.intensity = 0.08f,
	};

	directional_light = {
		.enabled = true,
		.direction = { -0.4f, -1.0f, 0.3f },
		.color = { 1.0f, 0.95f, 0.85f },
		.intensity = 0.5f,
	};

	point_lights = {
		makePointLight({ 0.0f, 1.5f, 0.0f }, { 1.0f, 0.6f, 0.2f }, 4.0f),
		makePointLight({ -5.0f, 1.0f, -2.0f }, { 0.2f, 0.5f, 1.0f }, 6.0f),
	};
	point_lights[0].orbit = false;
	point_lights[1].orbit = true;

	spot_lights = {
		makeSpotLight({ 3.5f, 4.0f, -2.0f }, { 0.0f, -1.0f, 0.5f }, { 1.0f, 1.0f, 1.0f }, 30.0f),
	};

	objects[0] = makeObject("Torus 1", &torus_mesh, { 0.0f, 0.0f, 0.0f }, { 0.9f, 0.9f, 0.9f }, false);
	objects[1] = makeObject("Torus 2", &torus_mesh, { -3.5f, 0.0f, 0.0f }, { 1.0f, 0.5f, 0.3f }, true);
	objects[2] = makeObject("Torus 3", &torus_mesh, { 3.5f, 0.0f, 0.0f }, { 0.3f, 0.7f, 1.0f }, true);
	objects[2].rotation = { 90.0f, 0.0f, 0.0f };
	objects[2].animation_speed = 0.6f;
	objects[2].spin_speed = -90.0f;
	objects[2].material.shininess = 128.0f;

	// Пол: без анимации и с матовым материалом (маленький тусклый блик)
	objects[3] = makeObject("Floor", &floor_mesh, { 0.0f, -2.0f, 0.0f }, { 0.6f, 0.6f, 0.6f }, false);
	objects[3].trajectory_radius = 0.0f;
	objects[3].trajectory_height = 0.0f;
	objects[3].spin_speed = 0.0f;
	objects[3].material.specular[0] = 0.2f;
	objects[3].material.specular[1] = 0.2f;
	objects[3].material.specular[2] = 0.2f;
	objects[3].material.shininess = 16.0f;
}

// ---------------------------------------------------------------------------
// Буферы
// ---------------------------------------------------------------------------

// Создаёт буфер, к которому есть доступ с процессора (лекция 3, загрузка через VMA)
bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& result) {
	// Анкета буфера
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		// Размер в байтах
		.size = size,
		// Для чего буфер: вершины, индексы, uniform- или storage-данные
		.usage = usage,
		// Буфер использует только одна очередь
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	// Анкета выделения памяти для VMA
	const VmaAllocationCreateInfo allocation_info = {
		// MAPPED — сразу получить адрес для записи с процессора,
		// HOST_ACCESS_SEQUENTIAL_WRITE — процессор пишет в память последовательно
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
		         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		// VMA сама выбирает подходящий тип памяти
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	// Сюда VMA запишет сведения о выделенной памяти, в том числе адрес для записи
	VmaAllocationInfo info = {};

	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &result.buffer, &result.allocation, &info) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	result.mapped = info.pMappedData;
	return true;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

// Копирует сгенерированную сетку в вершинный и индексный буферы
bool createMesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Mesh& mesh) {
	const VkDeviceSize vertices_size = vertices.size() * sizeof(Vertex);
	const VkDeviceSize indices_size = indices.size() * sizeof(uint32_t);

	if (!createBuffer(vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertex_buffer)) {
		return false;
	}

	if (!createBuffer(indices_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.index_buffer)) {
		return false;
	}

	// Копируем данные из оперативной памяти в буферы
	std::memcpy(mesh.vertex_buffer.mapped, vertices.data(), vertices_size);
	std::memcpy(mesh.index_buffer.mapped, indices.data(), indices_size);

	mesh.index_count = uint32_t(indices.size());
	return true;
}

void destroyMesh(Mesh& mesh) {
	destroyBuffer(mesh.vertex_buffer);
	destroyBuffer(mesh.index_buffer);
	mesh.index_count = 0;
}

bool createMeshes() {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	generateTorus(torus_major_radius, torus_minor_radius,
	              torus_major_segments, torus_minor_segments, vertices, indices);
	if (!createMesh(vertices, indices, torus_mesh)) {
		return false;
	}

	// Сфера единичного радиуса: размер маркера задаётся матрицей масштаба
	generateSphere(1.0f, 24, 12, vertices, indices);
	if (!createMesh(vertices, indices, sphere_mesh)) {
		return false;
	}

	generatePlane(floor_size, vertices, indices);
	if (!createMesh(vertices, indices, floor_mesh)) {
		return false;
	}

	return true;
}

// ---------------------------------------------------------------------------
// Дескрипторы (лекции 3 и 4)
// ---------------------------------------------------------------------------

// Записывает буфер в binding набора дескрипторов
void writeBufferDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type,
                           VkBuffer buffer, VkDeviceSize range) {
	// Какой буфер и какую его часть увидит шейдер
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer,
		.offset = 0,
		.range = range,
	};

	// Анкета записи в набор
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		// В какой набор пишем
		.dstSet = set,
		// В какой binding
		.dstBinding = binding,
		.descriptorCount = 1,
		// Тип должен совпадать с типом в раскладке набора
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
}

// Проверяет, что в storage-буфере хватает места на needed источников.
// Если нет — создаёт новый буфер вдвое больше и записывает его в набор дескрипторов.
// Так в шейдере не нужно задавать максимальное число источников.
// Удалять старый буфер здесь безопасно: render() вызывается после prepare(),
// а prepare() уже дождался, пока видеокарта закончит прошлый кадр
bool ensureLightBufferCapacity(Buffer& buffer, uint32_t& capacity, uint32_t needed,
                               VkDeviceSize element_size, uint32_t binding) {
	if (needed <= capacity && buffer.buffer != VK_NULL_HANDLE) {
		return true;
	}

	uint32_t new_capacity = std::max(capacity, initial_light_capacity);
	while (new_capacity < needed) {
		new_capacity *= 2;
	}

	destroyBuffer(buffer);
	if (!createBuffer(new_capacity * element_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer)) {
		capacity = 0;
		return false;
	}
	capacity = new_capacity;

	writeBufferDescriptor(vk_scene_descriptor_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
	                      buffer.buffer, VK_WHOLE_SIZE);
	return true;
}

// Набор 0 (сцена): uniform-буфер + два storage-буфера с массивами источников
bool createSceneSetLayout() {
	const VkDescriptorSetLayoutBinding bindings[] = {
		{
			// SceneUniforms: матрица нужна вершинному шейдеру, свет — фрагментному
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			// Массив точечных источников
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			// Массив прожекторов
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings = bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &vk_scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	return true;
}

// Набор 1 (модель): один uniform-буфер с матрицами и материалом
bool createModelSetLayout() {
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		// Матрицы читает вершинный шейдер, материал — фрагментный
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &vk_model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	return true;
}

// Создаёт раскладки, пул и наборы дескрипторов, а также буферы для них
bool createDescriptors() {
	if (!createSceneSetLayout() || !createModelSetLayout()) {
		return false;
	}

	// Uniform-буферов: 1 у сцены + по одному у каждого объекта. Storage-буферов: 2 у сцены
	const VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1 + object_count,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
		},
	};

	// Анкета пула дескрипторов
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		// Сколько наборов максимум можно выделить
		.maxSets = 1 + object_count,
		.poolSizeCount = 2,
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	// Набор 0: данные сцены
	const VkDescriptorSetAllocateInfo scene_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_allocate_info,
	                             &vk_scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_uniform_buffer)) {
		return false;
	}

	writeBufferDescriptor(vk_scene_descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	                      scene_uniform_buffer.buffer, sizeof(SceneUniforms));

	// Storage-буферы источников создаём сразу, даже если источников ещё нет:
	// каждый binding набора должен указывать на настоящий буфер
	if (!ensureLightBufferCapacity(point_light_buffer, point_light_capacity, 0,
	                               sizeof(GpuPointLight), 1) ||
	    !ensureLightBufferCapacity(spot_light_buffer, spot_light_capacity, 0,
	                               sizeof(GpuSpotLight), 2)) {
		return false;
	}

	// Набор 1: у каждого объекта свой набор и свой буфер
	for (Object& object : objects) {
		const VkDescriptorSetAllocateInfo model_allocate_info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = vk_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &vk_model_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &model_allocate_info,
		                             &object.descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate model descriptor set\n";
			return false;
		}

		if (!createBuffer(sizeof(ModelUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  object.uniform_buffer)) {
			return false;
		}

		writeBufferDescriptor(object.descriptor_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		                      object.uniform_buffer.buffer, sizeof(ModelUniforms));
	}

	return true;
}

// ---------------------------------------------------------------------------
// Конвейеры (лекция 2)
// ---------------------------------------------------------------------------

// Читает скомпилированный шейдер (.spv) с диска и создаёт из него VkShaderModule
VkShaderModule loadShaderModule(const char* path) {
	// Открываем файл в двоичном режиме, сразу в конце (ate), чтобы узнать размер
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const size_t size = size_t(file.tellg());

	// SPIR-V состоит из 32-битных слов, поэтому храним его в массиве uint32_t
	std::vector<uint32_t> code(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	// Анкета шейдерного модуля
	const VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		// Размер кода в байтах
		.codeSize = size,
		// Указатель на код
		.pCode = code.data(),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(context.device, &module_info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from " << path << '\n';
		return VK_NULL_HANDLE;
	}

	return module;
}

// Всё, чем три наших конвейера отличаются друг от друга
struct PipelineSettings {
	const char* vertex_shader;
	const char* fragment_shader;
	VkPipelineLayout layout;
	// Размер одной вершины и описание её атрибутов
	uint32_t vertex_stride;
	const VkVertexInputAttributeDescription* attributes;
	uint32_t attribute_count;
	// Треугольники или отрезки
	VkPrimitiveTopology topology;
	// Отбрасывать ли задние грани (у линий граней нет)
	VkCullModeFlags cull_mode;
};

// Создаёт графический конвейер (лекция 2, стадии конвейера)
bool createPipeline(const PipelineSettings& settings, VkPipeline& pipeline) {
	VkShaderModule vertex_module = loadShaderModule(settings.vertex_shader);
	VkShaderModule fragment_module = loadShaderModule(settings.fragment_shader);
	if (vertex_module == VK_NULL_HANDLE || fragment_module == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_module, nullptr);
		vkDestroyShaderModule(context.device, fragment_module, nullptr);
		return false;
	}

	// 1. Шейдеры: вершинный и фрагментный, точка входа — функция main
	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_module,
			.pName = "main",
		},
	};

	// 2. Откуда брать вершины: один буфер (binding 0), шаг — размер одной вершины
	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = settings.vertex_stride,
		// Переходить к следующему элементу на каждой вершине
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = settings.attribute_count,
		.pVertexAttributeDescriptions = settings.attributes,
	};

	// 3. Как собирать вершины в примитивы
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = settings.topology,
	};

	// 4. Один viewport и один scissor, сами значения задаются в render()
	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// 5. Растеризация. Лицевые грани — те, что на экране обходятся по часовой стрелке:
	// в лабе 1 было против часовой, но теперь проекция переворачивает ось y
	// (мир с осью y вверх), а отражение меняет направление обхода на противоположное
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = settings.cull_mode,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	// 6. Без сглаживания (одна выборка на пиксель)
	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	// 7. Тест глубины: рисуем пиксель, только если он ближе уже нарисованного
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	// 8. Пишем все четыре канала цвета, смешивание (прозрачность) выключено
	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	// 9. Viewport и scissor задаются командами при записи кадра (размер окна может меняться)
	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	// Анкета конвейера: собираем все стадии вместе
	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		// 10. Какие ресурсы видят шейдеры
		.layout = settings.layout,
		// В каком render pass и подпроходе будет работать конвейер
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &pipeline);

	// Модули нужны только при создании конвейера, потом их можно удалить
	vkDestroyShaderModule(context.device, vertex_module, nullptr);
	vkDestroyShaderModule(context.device, fragment_module, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline from " << settings.vertex_shader << '\n';
		return false;
	}

	return true;
}

bool createPipelines() {
	// Раскладка для освещаемых объектов: набор 0 — сцена, набор 1 — объект
	const VkDescriptorSetLayout lit_set_layouts[] = {
		vk_scene_set_layout,
		vk_model_set_layout,
	};

	const VkPipelineLayoutCreateInfo lit_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = lit_set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &lit_layout_info, nullptr,
	                           &vk_lit_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create lit pipeline layout\n";
		return false;
	}

	// Раскладка для маркеров и линий: только набор 0 (нужна матрица P * V)
	// плюс push-константы с матрицей и цветом маркера
	const VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(MarkerConstants),
	};

	const VkPipelineLayoutCreateInfo unlit_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_scene_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};

	if (vkCreatePipelineLayout(context.device, &unlit_layout_info, nullptr,
	                           &vk_unlit_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create unlit pipeline layout\n";
		return false;
	}

	// Атрибуты обычной вершины: позиция (location 0) и нормаль (location 1)
	const VkVertexInputAttributeDescription mesh_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
	};

	// Атрибуты вершины линии: позиция и цвет
	const VkVertexInputAttributeDescription line_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(LineVertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(LineVertex, color),
		},
	};

	// Освещаемые объекты: Блинн–Фонг во фрагментном шейдере
	const PipelineSettings lit = {
		.vertex_shader = "shaders/lit.vert.spv",
		.fragment_shader = "shaders/lit.frag.spv",
		.layout = vk_lit_pipeline_layout,
		.vertex_stride = sizeof(Vertex),
		.attributes = mesh_attributes,
		.attribute_count = 2,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.cull_mode = VK_CULL_MODE_BACK_BIT,
	};

	// Маркеры источников: та же сетка (Vertex), но берём только позицию, цвет постоянный
	const PipelineSettings marker = {
		.vertex_shader = "shaders/marker.vert.spv",
		.fragment_shader = "shaders/unlit.frag.spv",
		.layout = vk_unlit_pipeline_layout,
		.vertex_stride = sizeof(Vertex),
		.attributes = mesh_attributes,
		.attribute_count = 1,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.cull_mode = VK_CULL_MODE_BACK_BIT,
	};

	// Линии конусов прожекторов: каждые две вершины — отдельный отрезок
	const PipelineSettings line = {
		.vertex_shader = "shaders/line.vert.spv",
		.fragment_shader = "shaders/unlit.frag.spv",
		.layout = vk_unlit_pipeline_layout,
		.vertex_stride = sizeof(LineVertex),
		.attributes = line_attributes,
		.attribute_count = 2,
		.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
		.cull_mode = VK_CULL_MODE_NONE,
	};

	return createPipeline(lit, vk_lit_pipeline) &&
	       createPipeline(marker, vk_marker_pipeline) &&
	       createPipeline(line, vk_line_pipeline);
}

// ---------------------------------------------------------------------------
// Матрицы
// ---------------------------------------------------------------------------

// Смещение по траектории: «восьмёрка» в плоскости XZ и покачивание по Y
Vec3 trajectoryOffset(const Object& object) {
	const float t = object.animation_time;
	return {
		object.trajectory_radius * std::sin(t),
		object.trajectory_height * std::sin(3.0f * t),
		object.trajectory_radius * std::sin(t) * std::cos(t),
	};
}

// Матрица модели M = T * R * S: сначала растяжение, потом поворот, потом перенос (лекция 4)
Mat4 modelMatrix(const Object& object) {
	// Во время анимации объект ещё и вращается вокруг своих осей
	const float spin = object.spin_speed * object.animation_time;

	const Mat4 t = translation(object.position + trajectoryOffset(object));
	const Mat4 r = rotationY(radians(object.rotation.y + spin)) *
	               rotationX(radians(object.rotation.x + 0.5f * spin)) *
	               rotationZ(radians(object.rotation.z));
	const Mat4 s = scaling(object.scale);

	return t * r * s;
}

// Направление взгляда камеры в мире.
// При yaw = 0 и pitch = 0 камера смотрит вдоль +z; yaw поворачивает взгляд вправо (к +x),
// pitch поднимает его вверх (к +y). Это столбец Ry(yaw) * Rx(-pitch) * (0, 0, 1)
Vec3 cameraForward() {
	const float yaw = radians(camera.yaw);
	const float pitch = radians(camera.pitch);
	return {
		std::sin(yaw) * std::cos(pitch),
		std::sin(pitch),
		std::cos(yaw) * std::cos(pitch),
	};
}

// Направление «вправо» для камеры (горизонтальное, от наклона не зависит)
Vec3 cameraRight() {
	const float yaw = radians(camera.yaw);
	return { std::cos(yaw), 0.0f, -std::sin(yaw) };
}

// Матрица вида V (лекция 4). Матрица модели камеры C = T(position) * Ry(yaw) * Rx(-pitch):
// сначала наклоняем камеру, потом поворачиваем, потом ставим в её позицию.
// V = C^-1 = Rx(-pitch)^-1 * Ry(yaw)^-1 * T(position)^-1 = Rx(pitch) * Ry(-yaw) * T(-position).
// Обратный поворот — поворот на минус угол, обратный перенос — перенос на минус вектор
Mat4 viewMatrix() {
	return rotationX(radians(camera.pitch)) *
	       rotationY(radians(-camera.yaw)) *
	       translation(-camera.position);
}

Mat4 projectionMatrix(float aspect) {
	if (camera.use_perspective) {
		return perspective(radians(camera.fov_degrees), aspect, camera.near_plane, camera.far_plane);
	}
	return orthographic(camera.ortho_half_height, aspect, camera.near_plane, camera.far_plane);
}

// ---------------------------------------------------------------------------
// Управление камерой с клавиатуры и мыши
// ---------------------------------------------------------------------------

// ПКМ + мышь — поворот, WASD — полёт, Q/E — вниз/вверх, Shift — быстрее.
// Клавиши и мышь читаем через ImGui: GLFW-бэкенд ImGui уже получает все события окна
void updateCameraControls(float dt) {
	const ImGuiIO& io = ImGui::GetIO();

	// Мышь над окном интерфейса — камеру не трогаем
	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		camera.yaw += io.MouseDelta.x * camera.mouse_sensitivity;
		// Мышь вверх (delta.y < 0) — смотрим вверх
		camera.pitch -= io.MouseDelta.y * camera.mouse_sensitivity;
	}

	// Держим yaw в [-180, 180], чтобы слайдер в интерфейсе показывал понятное значение
	if (camera.yaw > 180.0f) {
		camera.yaw -= 360.0f;
	}
	if (camera.yaw < -180.0f) {
		camera.yaw += 360.0f;
	}
	// Ровно ±90° нельзя: взгляд станет вертикальным и «право» потеряет смысл
	camera.pitch = std::clamp(camera.pitch, -89.0f, 89.0f);

	// Пока печатаем в поле ввода ImGui, клавиши не должны двигать камеру
	if (io.WantCaptureKeyboard) {
		return;
	}

	Vec3 move = { 0.0f, 0.0f, 0.0f };
	if (ImGui::IsKeyDown(ImGuiKey_W)) move = move + cameraForward();
	if (ImGui::IsKeyDown(ImGuiKey_S)) move = move - cameraForward();
	if (ImGui::IsKeyDown(ImGuiKey_D)) move = move + cameraRight();
	if (ImGui::IsKeyDown(ImGuiKey_A)) move = move - cameraRight();
	if (ImGui::IsKeyDown(ImGuiKey_E)) move = move + Vec3{ 0.0f, 1.0f, 0.0f };
	if (ImGui::IsKeyDown(ImGuiKey_Q)) move = move - Vec3{ 0.0f, 1.0f, 0.0f };

	if (length(move) > 0.0f) {
		float speed = camera.move_speed;
		if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) {
			speed *= 3.0f;
		}
		// Нормируем, чтобы по диагонали камера не летела быстрее
		camera.position = camera.position + normalize(move) * (speed * dt);
	}
}

// ---------------------------------------------------------------------------
// Интерфейс ImGui
// ---------------------------------------------------------------------------

void drawCameraInterface() {
	if (!ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		return;
	}

	ImGui::TextDisabled("RMB + mouse: look around");
	ImGui::TextDisabled("WASD: move, Q/E: down/up, Shift: faster");

	ImGui::DragFloat3("Position", &camera.position.x, 0.05f);
	ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.0f deg");
	ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.0f deg");

	if (ImGui::RadioButton("Perspective", camera.use_perspective)) {
		camera.use_perspective = true;
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("Orthographic", !camera.use_perspective)) {
		camera.use_perspective = false;
	}

	if (camera.use_perspective) {
		ImGui::SliderFloat("FOV", &camera.fov_degrees, 20.0f, 120.0f, "%.0f deg");
	} else {
		ImGui::SliderFloat("Half height", &camera.ortho_half_height, 1.0f, 20.0f);
	}
	ImGui::SliderFloat("Near", &camera.near_plane, 0.01f, 5.0f);
	ImGui::SliderFloat("Far", &camera.far_plane, 10.0f, 200.0f);

	ImGui::SliderFloat("Move speed", &camera.move_speed, 0.5f, 20.0f);
	ImGui::SliderFloat("Mouse sensitivity", &camera.mouse_sensitivity, 0.05f, 1.0f);

	if (ImGui::Button("Reset camera")) {
		camera = defaultCamera();
	}
}

void drawLightsInterface() {
	if (ImGui::CollapsingHeader("Ambient light")) {
		ImGui::PushID("ambient");
		ImGui::ColorEdit3("Color", ambient_light.color);
		ImGui::SliderFloat("Intensity", &ambient_light.intensity, 0.0f, 1.0f);
		ImGui::PopID();
	}

	if (ImGui::CollapsingHeader("Directional light")) {
		ImGui::PushID("directional");
		ImGui::Checkbox("Enabled", &directional_light.enabled);
		ImGui::DragFloat3("Direction", &directional_light.direction.x, 0.01f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Color", directional_light.color);
		ImGui::SliderFloat("Intensity", &directional_light.intensity, 0.0f, 5.0f);
		ImGui::PopID();
	}

	// Доп. задание 2: массив точечных источников с добавлением и удалением
	char header[64];
	std::snprintf(header, sizeof(header), "Point lights (%zu)###point_lights", point_lights.size());
	if (ImGui::CollapsingHeader(header)) {
		ImGui::PushID("point");

		if (ImGui::Button("Add point light")) {
			// Новый источник ставим перед камерой, чтобы его сразу было видно
			const Vec3 position = camera.position + cameraForward() * 4.0f;
			point_lights.push_back(makePointLight(position, { 1.0f, 1.0f, 1.0f }, 5.0f));
		}

		// Удалять элемент прямо внутри цикла по вектору нельзя — запоминаем номер
		int remove_index = -1;
		for (size_t i = 0; i < point_lights.size(); ++i) {
			PointLight& light = point_lights[i];
			ImGui::PushID(int(i));

			char label[48];
			std::snprintf(label, sizeof(label), "Point light %zu", i + 1);
			if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::DragFloat3("Position", &light.position.x, 0.05f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);
				ImGui::Checkbox("Orbit", &light.orbit);
				if (light.orbit) {
					ImGui::SameLine();
					ImGui::SliderFloat("Speed", &light.orbit_speed, -180.0f, 180.0f, "%.0f deg/s");
				}
				if (ImGui::Button("Remove")) {
					remove_index = int(i);
				}
				ImGui::TreePop();
			}

			ImGui::PopID();
		}

		if (remove_index >= 0) {
			point_lights.erase(point_lights.begin() + remove_index);
		}

		ImGui::PopID();
	}

	// Доп. задание 3: массив прожекторов с добавлением и удалением
	std::snprintf(header, sizeof(header), "Spot lights (%zu)###spot_lights", spot_lights.size());
	if (ImGui::CollapsingHeader(header)) {
		ImGui::PushID("spot");

		if (ImGui::Button("Add spot light")) {
			// Новый прожектор ставим в позицию камеры и направляем туда же, куда смотрит камера —
			// получается «фонарик»
			spot_lights.push_back(makeSpotLight(camera.position, cameraForward(),
			                                    { 1.0f, 1.0f, 1.0f }, 30.0f));
		}

		int remove_index = -1;
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			SpotLight& light = spot_lights[i];
			ImGui::PushID(int(i));

			char label[48];
			std::snprintf(label, sizeof(label), "Spot light %zu", i + 1);
			if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::DragFloat3("Position", &light.position.x, 0.05f);
				ImGui::DragFloat3("Direction", &light.direction.x, 0.01f, -1.0f, 1.0f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 200.0f);
				ImGui::SliderFloat("Inner angle", &light.inner_angle, 0.0f, 80.0f, "%.1f deg");
				ImGui::SliderFloat("Outer angle", &light.outer_angle, 0.0f, 85.0f, "%.1f deg");
				// Внешний угол не может быть меньше внутреннего
				light.outer_angle = std::max(light.outer_angle, light.inner_angle);

				if (ImGui::Button("Point from camera")) {
					light.position = camera.position;
					light.direction = cameraForward();
				}
				ImGui::SameLine();
				if (ImGui::Button("Remove")) {
					remove_index = int(i);
				}
				ImGui::TreePop();
			}

			ImGui::PopID();
		}

		if (remove_index >= 0) {
			spot_lights.erase(spot_lights.begin() + remove_index);
		}

		ImGui::PopID();
	}

	// Доп. задание 4: показ/скрытие визуализации источников
	if (ImGui::CollapsingHeader("Light visualization")) {
		ImGui::Checkbox("Show light markers", &show_light_markers);
		ImGui::Checkbox("Show spot cones", &show_spot_cones);
		ImGui::SliderFloat("Marker size", &marker_size, 0.02f, 0.5f);
		ImGui::SliderFloat("Cone length", &cone_length, 0.5f, 10.0f);
	}
}

void drawObjectsInterface() {
	// Выбор объекта, который настраиваем ниже
	ImGui::SeparatorText("Objects");
	for (uint32_t i = 0; i < object_count; ++i) {
		ImGui::PushID(int(i));
		ImGui::Checkbox("##visible", &objects[i].visible);
		ImGui::SameLine();
		ImGui::RadioButton(objects[i].name, &selected_object, int(i));
		ImGui::PopID();
		// По два объекта в строке, чтобы всё помещалось в окно
		if (i % 2 == 0 && i + 1 < object_count) {
			ImGui::SameLine(170.0f);
		}
	}

	Object& object = objects[selected_object];

	if (ImGui::CollapsingHeader("Transform")) {
		ImGui::DragFloat3("Position", &object.position.x, 0.05f);
		ImGui::SliderFloat3("Rotation", &object.rotation.x, -180.0f, 180.0f, "%.0f deg");
		ImGui::DragFloat3("Scale", &object.scale.x, 0.01f, 0.1f, 5.0f);
		if (ImGui::Button("Reset transform")) {
			object.rotation = { 0.0f, 0.0f, 0.0f };
			object.scale = { 1.0f, 1.0f, 1.0f };
		}
	}

	// Параметры материала по Блинну–Фонгу
	if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Albedo", object.material.albedo);
		ImGui::ColorEdit3("Specular color", object.material.specular);
		// Логарифмический слайдер: удобно крутить и маленькие, и большие значения
		ImGui::SliderFloat("Shininess", &object.material.shininess, 1.0f, 512.0f, "%.0f",
		                   ImGuiSliderFlags_Logarithmic);
	}

	if (ImGui::CollapsingHeader("Animation")) {
		if (ImGui::Button(object.animation_playing ? "Pause" : "Play")) {
			object.animation_playing = !object.animation_playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Reset animation")) {
			object.animation_time = 0.0f;
		}
		ImGui::SliderFloat("Speed", &object.animation_speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Radius", &object.trajectory_radius, 0.0f, 5.0f);
		ImGui::SliderFloat("Height", &object.trajectory_height, 0.0f, 3.0f);
		ImGui::SliderFloat("Spin speed", &object.spin_speed, -360.0f, 360.0f, "%.0f deg/s");
	}
}

void drawInterface() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360.0f, 600.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 2: Camera and Lighting");

	// Поля ввода чуть уже окна, чтобы справа оставалось место под подписи
	ImGui::PushItemWidth(-130.0f);

	ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

	drawCameraInterface();
	drawLightsInterface();
	drawObjectsInterface();

	ImGui::PopItemWidth();
	ImGui::End();
}

// ---------------------------------------------------------------------------
// Подготовка данных для видеокарты
// ---------------------------------------------------------------------------

// Цвет из интерфейса, умноженный на интенсивность
Vec3 scaledColor(const float color[3], float intensity) {
	return { color[0] * intensity, color[1] * intensity, color[2] * intensity };
}

// Два единичных вектора, перпендикулярных direction и друг другу.
// Нужны, чтобы нарисовать окружность основания конуса вокруг его оси
void perpendicularBasis(Vec3 direction, Vec3& b1, Vec3& b2) {
	// Берём любой вектор, не параллельный direction
	const Vec3 helper = std::fabs(direction.y) < 0.99f ? Vec3{ 0.0f, 1.0f, 0.0f }
	                                                   : Vec3{ 1.0f, 0.0f, 0.0f };
	b1 = normalize(cross(direction, helper));
	b2 = cross(direction, b1);
}

// Строит отрезки, изображающие конусы прожекторов: ось, окружности основания
// внешнего и внутреннего конуса и несколько образующих
void buildConeLines(std::vector<LineVertex>& lines) {
	const float two_pi = 2.0f * 3.14159265358979f;
	const Vec3 white = { 1.0f, 1.0f, 1.0f };
	const Vec3 gray = { 0.5f, 0.5f, 0.5f };

	auto add_line = [&lines](Vec3 a, Vec3 b, Vec3 color) {
		lines.push_back({ a, color });
		lines.push_back({ b, color });
	};

	for (const SpotLight& light : spot_lights) {
		const Vec3 apex = light.position;
		const Vec3 axis = normalize(light.direction);
		const Vec3 base_center = apex + axis * cone_length;

		Vec3 b1;
		Vec3 b2;
		perpendicularBasis(axis, b1, b2);

		// Радиус основания: тангенс половины угла раствора, умноженный на длину конуса
		const float outer_radius = cone_length * std::tan(radians(light.outer_angle));
		const float inner_radius = cone_length * std::tan(radians(light.inner_angle));

		// Точка окружности основания радиуса r под углом a
		auto circle_point = [&](float r, float a) {
			return base_center + b1 * (r * std::cos(a)) + b2 * (r * std::sin(a));
		};

		// Ось прожектора
		add_line(apex, base_center, white);

		for (uint32_t i = 0; i < cone_circle_segments; ++i) {
			const float a0 = two_pi * float(i) / float(cone_circle_segments);
			const float a1 = two_pi * float(i + 1) / float(cone_circle_segments);
			// Внешний конус — белый, внутренний — серый
			add_line(circle_point(outer_radius, a0), circle_point(outer_radius, a1), white);
			add_line(circle_point(inner_radius, a0), circle_point(inner_radius, a1), gray);
		}

		for (uint32_t i = 0; i < cone_side_lines; ++i) {
			const float a = two_pi * float(i) / float(cone_side_lines);
			add_line(apex, circle_point(outer_radius, a), white);
		}
	}
}

// Копирует всё, что изменилось за кадр, в буферы видеокарты
bool uploadFrameData(float aspect) {
	// --- Набор 0: uniform-буфер сцены ---
	const Vec3 directional_color = directional_light.enabled
		? scaledColor(directional_light.color, directional_light.intensity)
		: Vec3{ 0.0f, 0.0f, 0.0f };

	const SceneUniforms scene = {
		.view_projection = projectionMatrix(aspect) * viewMatrix(),
		.camera_position = camera.position,
		.directional_light_direction = normalize(directional_light.direction),
		.directional_light_color = directional_color,
		.ambient_light = scaledColor(ambient_light.color, ambient_light.intensity),
		.point_light_count = uint32_t(point_lights.size()),
		.spot_light_count = uint32_t(spot_lights.size()),
	};
	std::memcpy(scene_uniform_buffer.mapped, &scene, sizeof(scene));

	// --- Набор 0: storage-буферы источников ---
	if (!ensureLightBufferCapacity(point_light_buffer, point_light_capacity,
	                               uint32_t(point_lights.size()), sizeof(GpuPointLight), 1) ||
	    !ensureLightBufferCapacity(spot_light_buffer, spot_light_capacity,
	                               uint32_t(spot_lights.size()), sizeof(GpuSpotLight), 2)) {
		return false;
	}

	GpuPointLight* gpu_points = static_cast<GpuPointLight*>(point_light_buffer.mapped);
	for (size_t i = 0; i < point_lights.size(); ++i) {
		gpu_points[i] = {
			.position = point_lights[i].position,
			.color = scaledColor(point_lights[i].color, point_lights[i].intensity),
		};
	}

	GpuSpotLight* gpu_spots = static_cast<GpuSpotLight*>(spot_light_buffer.mapped);
	for (size_t i = 0; i < spot_lights.size(); ++i) {
		const SpotLight& light = spot_lights[i];
		gpu_spots[i] = {
			.position = light.position,
			// В шейдере сравниваем косинусы, поэтому сразу передаём косинусы углов
			.inner_cos = std::cos(radians(light.inner_angle)),
			.direction = normalize(light.direction),
			.outer_cos = std::cos(radians(light.outer_angle)),
			.color = scaledColor(light.color, light.intensity),
		};
	}

	// --- Набор 1: uniform-буферы объектов ---
	for (Object& object : objects) {
		const Mat4 model = modelMatrix(object);
		const Material& material = object.material;

		const ModelUniforms uniforms = {
			.model = model,
			.normal_matrix = normalMatrix(model),
			.albedo = { material.albedo[0], material.albedo[1], material.albedo[2] },
			.specular = { material.specular[0], material.specular[1], material.specular[2] },
			.shininess = material.shininess,
		};
		std::memcpy(object.uniform_buffer.mapped, &uniforms, sizeof(uniforms));
	}

	// --- Вершины линий конусов ---
	line_vertex_count = 0;
	if (show_spot_cones && !spot_lights.empty()) {
		std::vector<LineVertex> lines;
		buildConeLines(lines);

		// Если буфер мал — пересоздаём побольше (как и storage-буферы)
		if (lines.size() > line_vertex_capacity) {
			uint32_t new_capacity = std::max(line_vertex_capacity, 256u);
			while (new_capacity < lines.size()) {
				new_capacity *= 2;
			}

			destroyBuffer(line_vertex_buffer);
			if (!createBuffer(new_capacity * sizeof(LineVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			                  line_vertex_buffer)) {
				line_vertex_capacity = 0;
				return false;
			}
			line_vertex_capacity = new_capacity;
		}

		std::memcpy(line_vertex_buffer.mapped, lines.data(), lines.size() * sizeof(LineVertex));
		line_vertex_count = uint32_t(lines.size());
	}

	return true;
}

// Рисует сферы-маркеры источников постоянным цветом (без освещения)
void drawLightMarkers(VkCommandBuffer cb) {
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cb, 0, 1, &sphere_mesh.vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cb, sphere_mesh.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	auto draw_marker = [&](Vec3 position, const float color[3]) {
		const MarkerConstants constants = {
			.model = translation(position) * scaling({ marker_size, marker_size, marker_size }),
			.color = { color[0], color[1], color[2], 1.0f },
		};
		// Push-константы записываются прямо в командный буфер перед вызовом отрисовки
		vkCmdPushConstants(cb, vk_unlit_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
		                   0, sizeof(constants), &constants);
		vkCmdDrawIndexed(cb, sphere_mesh.index_count, 1, 0, 0, 0);
	};

	for (const PointLight& light : point_lights) {
		draw_marker(light.position, light.color);
	}
	for (const SpotLight& light : spot_lights) {
		draw_marker(light.position, light.color);
	}
}

} // namespace

bool initialize() {
	// Сетки нужны раньше сцены: объекты хранят указатели на них
	if (!createMeshes()) {
		return false;
	}

	setupScene();

	if (!createDescriptors() || !createPipelines()) {
		return false;
	}

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, vk_line_pipeline, nullptr);
	vkDestroyPipeline(context.device, vk_marker_pipeline, nullptr);
	vkDestroyPipeline(context.device, vk_lit_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_unlit_pipeline_layout, nullptr);
	vkDestroyPipelineLayout(context.device, vk_lit_pipeline_layout, nullptr);

	// Наборы дескрипторов удаляются вместе с пулом
	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_scene_set_layout, nullptr);

	for (Object& object : objects) {
		destroyBuffer(object.uniform_buffer);
	}
	destroyBuffer(line_vertex_buffer);
	destroyBuffer(spot_light_buffer);
	destroyBuffer(point_light_buffer);
	destroyBuffer(scene_uniform_buffer);

	destroyMesh(floor_mesh);
	destroyMesh(sphere_mesh);
	destroyMesh(torus_mesh);
}

void update(double time) {
	// Сколько секунд прошло с прошлого кадра
	float dt = float(time - last_time);
	last_time = time;
	if (dt > 0.1f) {
		dt = 0.1f;
	}

	updateCameraControls(dt);

	// Время анимации идёт, только когда она не на паузе
	for (Object& object : objects) {
		if (object.animation_playing) {
			object.animation_time += dt * object.animation_speed;
		}
	}

	// Анимация точечных источников: поворачиваем позицию вокруг вертикальной оси мира
	for (PointLight& light : point_lights) {
		if (light.orbit) {
			light.position = transformPoint(rotationY(radians(light.orbit_speed * dt)), light.position);
		}
	}

	drawInterface();
}

void render(const graphics::internal::FrameData& fd) {
	// Короткое имя для командного буфера
	VkCommandBuffer cb = fd.command_buffer;

	// Записываем данные в буферы здесь, а не в update(): к этому моменту prepare()
	// уже дождался, пока видеокарта закончит прошлый кадр и перестанет читать буферы
	const VkExtent2D extent = context.swapchain_extent;
	const float aspect = float(extent.width) / float(extent.height);

	const bool data_ready = uploadFrameData(aspect);

	// Стираем записку с прошлого кадра
	vkResetCommandBuffer(cb, 0);

	// Анкета для начала записи командного буфера
	const VkCommandBufferBeginInfo begin_info = {
		// Сообщает вулкану, какая это анкета
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		// Значит, что мы будем использовать этот командный буфер только один раз
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	// Начинаем запись командного буфера
	vkBeginCommandBuffer(cb, &begin_info);

	// Значения очистки буфера кадра (цвет и глубина)
	const VkClearValue clear_values[] = {
		{ .color = { { 0.02f, 0.02f, 0.03f, 1.0f } } }, // тёмный фон, чтобы свет было лучше видно
		{ .depthStencil = { 1.0f, 0 } }, // для буфера глубины
	};

	// Анкета для начала прохода рисования
	const VkRenderPassBeginInfo render_pass_begin = {
		// Тип анкеты
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		// Какой проход начать
		.renderPass = context.render_pass,
		// В какие конкретные картинки рисовать
		.framebuffer = fd.framebuffer,
		// Прямоугольник, в котором идёт рисование
		.renderArea = { .extent = extent },
		// Сколько значений очистки передаём
		.clearValueCount = 2,
		// Указатель на массив значений очистки
		.pClearValues = clear_values,
	};

	// Записывает в командный буфер команду начала прохода рисования
	vkCmdBeginRenderPass(cb, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	// Viewport: переводит координаты [-1, 1] в пиксели окна, глубину — в [0, 1]
	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(extent.width),
		.height = float(extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(cb, 0, 1, &viewport);

	// Scissor: всё, что вне этого прямоугольника, отрезается
	const VkRect2D scissor = { .extent = extent };
	vkCmdSetScissor(cb, 0, 1, &scissor);

	// Если буферы не удалось подготовить, кадр остаётся просто очищенным
	if (data_ready) {
		const VkDeviceSize vertex_offset = 0;

		// --- 1. Освещаемые объекты ---
		vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_lit_pipeline);

		// Набор 0 (сцена) подключаем один раз
		vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_lit_pipeline_layout,
		                        0, 1, &vk_scene_descriptor_set, 0, nullptr);

		// Для каждого объекта подключаем его сетку и набор 1, затем рисуем
		for (const Object& object : objects) {
			if (!object.visible) {
				continue;
			}

			vkCmdBindVertexBuffers(cb, 0, 1, &object.mesh->vertex_buffer.buffer, &vertex_offset);
			vkCmdBindIndexBuffer(cb, object.mesh->index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_lit_pipeline_layout,
			                        1, 1, &object.descriptor_set, 0, nullptr);
			vkCmdDrawIndexed(cb, object.mesh->index_count, 1, 0, 0, 0);
		}

		// --- 2. Маркеры источников (доп. задание 4) ---
		if (show_light_markers && (!point_lights.empty() || !spot_lights.empty())) {
			vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_marker_pipeline);
			// У этой раскладки есть push-константы, поэтому она несовместима с раскладкой
			// освещаемых объектов — набор 0 нужно подключить заново
			vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_unlit_pipeline_layout,
			                        0, 1, &vk_scene_descriptor_set, 0, nullptr);
			drawLightMarkers(cb);
		}

		// --- 3. Конусы прожекторов (доп. задание 4) ---
		if (line_vertex_count > 0) {
			vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_line_pipeline);
			vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_unlit_pipeline_layout,
			                        0, 1, &vk_scene_descriptor_set, 0, nullptr);
			vkCmdBindVertexBuffers(cb, 0, 1, &line_vertex_buffer.buffer, &vertex_offset);
			vkCmdDraw(cb, line_vertex_count, 1, 0, 0);
		}
	}

	// Записывает в командный буфер команду "закончить проход рисования"
	vkCmdEndRenderPass(cb);

	// Заканчиваем запись командного буфера
	vkEndCommandBuffer(cb);
}

} // namespace application
