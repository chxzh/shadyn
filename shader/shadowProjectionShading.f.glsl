//varying vec3 f_color;
uniform vec3 shadowColor;
void main(void) {
  //gl_FragColor = vec4(f_color.x, f_color.y, f_color.z, 1.0);
  gl_FragColor = vec4(shadowColor, 1.0);
}
