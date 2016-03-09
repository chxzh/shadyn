in vec2 uv;
uniform sampler2D bg_tex;
void main() {
    gl_FragColor = vec4(1, texture(bg_tex, uv).r, 1, 1);
    //gl_FragColor = vec4(uv, 0.0, 1.0);
}