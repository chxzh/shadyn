#version 330 core
layout(location=1) in vec3 coord3d;


layout(location=2) in vec2 vertexUV;

out vec2 uv;

void main(void) {
    gl_Position = vec4(coord3d, 1.0);
    uv = vertexUV;    
}
