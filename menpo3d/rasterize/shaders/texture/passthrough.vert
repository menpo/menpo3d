#version 330
uniform mat4 MVP;
in vec3 in_vert;
in vec2 in_text;
in vec3 in_f3v;
out vec2 v_text;
out vec3 v_f3v;

void main() {
    gl_Position = vec4(in_vert, 1.0) * MVP;
    v_text = in_text;
    v_f3v = in_f3v;
}