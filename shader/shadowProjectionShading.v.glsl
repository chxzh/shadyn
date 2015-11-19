#version 330 core
layout(location = 1) in vec3 coord3d;

//attribute vec3 v_color;
uniform mat4 M;
uniform mat4 ShadowProjMat;
uniform mat4 VP;
//out float depth;

void main(void) {
  gl_Position = VP * ShadowProjMat * M * vec4(coord3d, 1.0);
  //depth = gl_Position.z;
  //f_color = v_color;
}
