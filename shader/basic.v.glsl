#version 330 core
layout(location = 1) in vec3 v_pos;

uniform mat4 MVP;

void main(void) {
  gl_Position = MVP * vec4(v_pos, 1.0);
}
