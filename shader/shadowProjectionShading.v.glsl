#version 330 core
layout(location = 1) in vec3 coord3d;

uniform mat4 MsVP;

void main(void) {
	vec4 position = MsVP * vec4(coord3d, 1.0);
	gl_Position = position / position.w;
}
