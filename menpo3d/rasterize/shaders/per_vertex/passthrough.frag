#version 330
uniform sampler2D Texture;
in vec3 v_color;
in vec3 v_f3v;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_f3v;

void main() {
   f_color = vec4(v_color, 1.0);
   f_f3v = v_f3v;
}