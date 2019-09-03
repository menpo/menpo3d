#version 330
uniform mat4 MVP;
in vec3 in_vert;
in vec3 in_color;
in vec3 in_f3v;
out vec3 v_color;
out vec3 v_f3v;

void main() {
    gl_Position = vec4(in_vert, 1.0) * MVP;
    v_color = in_color;
    v_f3v = in_f3v;
}