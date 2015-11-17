layout(location = 1) vec3 coord3d;
//attribute vec3 v_color;
uniform mat4 mvp;
out float depth;

void main(void) {
  gl_Position = mvp * vec4(coord3d, 1.0);
  depth = gl_Position.z;
  //f_color = v_color;
}
