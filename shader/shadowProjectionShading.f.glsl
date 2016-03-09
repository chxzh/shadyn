//varying vec3 f_color;
uniform vec3 shadowColor;
uniform float alpha;
void main(void) {
  //gl_FragColor = vec4(f_color.x, f_color.y, f_color.z, 1.0);
  //gl_FragColor = vec4(alpha*shadowColor, 1.0) + (1-alpha)*gl_FragColor;
  gl_FragColor = vec4(shadowColor, alpha);
}
