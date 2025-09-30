#version 450
layout(location=0) in vec3 in_pos;
layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 color;
    float pointSize; float _pad0; float _pad1; float _pad2; // align to 16 bytes
} pc;
layout(location=0) out vec4 v_color;
void main(){
#ifdef VERTEX_AS_POINTS
    gl_PointSize = pc.pointSize;
#endif
    gl_Position = pc.mvp * vec4(in_pos, 1.0);
    v_color = pc.color;
}

