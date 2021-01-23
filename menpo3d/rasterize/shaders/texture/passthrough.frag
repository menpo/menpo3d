#version 330
uniform sampler2D Texture;
in vec2 v_text;
in vec3 v_f3v;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_f3v;

void main() {
   // Note the Y texture coordinates are flipped
   vec3 color = texture(Texture, vec2(v_text.x, 1.0 - v_text.y)).rgb;
   f_color = vec4(color, 1.0);
   f_f3v = v_f3v;
}