#version 330 core
layout(location = 1) in vec3 coord3d;

uniform mat4 MsVP;
uniform float clip_depth;
uniform bool change_depth;
void main(void) {
	vec4 position = MsVP * vec4(coord3d, 1.0);
	gl_Position = position / position.w;
	if(change_depth)gl_Position.z = clip_depth;
}
