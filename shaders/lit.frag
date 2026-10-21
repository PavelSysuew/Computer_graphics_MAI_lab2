#version 450 core

// Позиция и нормаль в мире, интерполированные между вершинами треугольника
layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;

// Выход фрагментного шейдера: цвет пикселя в цветовое вложение 0
layout(location = 0) out vec4 out_color;

// Те же блоки, что и в lit.vert (одинаковое описание обязательно)
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
	vec3 directional_light_direction;
	vec3 directional_light_color;
	vec3 ambient_light;
	uint point_light_count;
	uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniforms {
	mat4 model;
	mat4 normal_matrix;
	vec3 albedo;
	vec3 specular;
	float shininess;
} model_uniforms;

// Точечный источник. Раскладка std430: vec3 выравнивается на 16 байт,
// поэтому структура занимает 32 байта (на C++ после каждого vec3 — float-заглушка)
struct PointLight {
	vec3 position;
	// Цвет, уже умноженный на интенсивность
	vec3 color;
};

// Прожектор. Свободные 4 байта после vec3 занимаем полезными float,
// чтобы не тратить место на заглушки. Размер — 48 байт
struct SpotLight {
	vec3 position;
	// Косинус внутреннего угла: внутри него прожектор светит в полную силу
	float inner_cos;
	// Направление, куда светит прожектор (единичный вектор)
	vec3 direction;
	// Косинус внешнего угла: за ним света уже нет
	float outer_cos;
	// Цвет, уже умноженный на интенсивность
	vec3 color;
};

// Storage-буферы (лекция 4): массивы без фиксированного размера в шейдере.
// readonly — шейдер только читает их
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

// Модель Блинна–Фонга для одного источника (лекция 4):
// C = I * (max(n·l, 0) * K_d + K_s * max(n·h, 0)^α)
// n — нормаль, v — направление на камеру, l — направление на источник,
// radiance — пришедший свет I (цвет * интенсивность * затухание)
vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 radiance) {
	float n_dot_l = max(dot(n, l), 0.0);

	// Если свет падает с обратной стороны поверхности, бликов тоже быть не должно
	if (n_dot_l <= 0.0) {
		return vec3(0.0);
	}

	// Половинный вектор между направлением на свет и на камеру
	vec3 h = normalize(l + v);
	float n_dot_h = max(dot(n, h), 0.0);

	vec3 diffuse = n_dot_l * model_uniforms.albedo;
	vec3 specular = model_uniforms.specular * pow(n_dot_h, model_uniforms.shininess);

	return radiance * (diffuse + specular);
}

void main() {
	// После интерполяции нормаль уже не единичная — нормируем заново
	vec3 n = normalize(in_world_normal);
	vec3 v = normalize(scene_uniforms.camera_position - in_world_position);

	// Рассеянный свет: одинаков для всех точек, без него теневая сторона чёрная.
	// В качестве K_a берём альбедо
	vec3 color = scene_uniforms.ambient_light * model_uniforms.albedo;

	// 1. Направленный источник: l одинаков для всех точек, затухания нет.
	// Свет летит по directional_light_direction, значит направление НА источник — обратное
	vec3 l_dir = -normalize(scene_uniforms.directional_light_direction);
	color += blinnPhong(n, v, l_dir, scene_uniforms.directional_light_color);

	// 2. Точечные источники
	for (uint i = 0; i < scene_uniforms.point_light_count; ++i) {
		vec3 to_light = point_lights[i].position - in_world_position;
		// Квадрат расстояния D² — это скалярное произведение вектора на себя
		float distance_squared = max(dot(to_light, to_light), 1e-4);
		vec3 l = to_light * inversesqrt(distance_squared);

		// Закон обратных квадратов: интенсивность падает как 1 / D²
		float attenuation = 1.0 / distance_squared;

		color += blinnPhong(n, v, l, point_lights[i].color * attenuation);
	}

	// 3. Прожекторы
	for (uint i = 0; i < scene_uniforms.spot_light_count; ++i) {
		SpotLight light = spot_lights[i];

		vec3 to_light = light.position - in_world_position;
		float distance_squared = max(dot(to_light, to_light), 1e-4);
		vec3 l = to_light * inversesqrt(distance_squared);

		float attenuation = 1.0 / distance_squared;

		// Косинус угла между осью прожектора и лучом от прожектора к точке.
		// Луч от прожектора к точке — это -l
		float cos_theta = dot(-l, normalize(light.direction));

		// Мягкие края: внутри внутреннего угла cone = 1, за внешним — 0,
		// между ними плавно убывает линейно по косинусу.
		// Если углы равны, знаменатель почти 0 и край получается резким
		float epsilon = max(light.inner_cos - light.outer_cos, 1e-4);
		float cone = clamp((cos_theta - light.outer_cos) / epsilon, 0.0, 1.0);

		color += blinnPhong(n, v, l, light.color * attenuation * cone);
	}

	out_color = vec4(color, 1.0);
}
