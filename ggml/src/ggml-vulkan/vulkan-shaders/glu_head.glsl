#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {A_TYPE data_b[];};
#ifdef ROWS_ID_SCALE
layout (binding = 2) readonly buffer Scale {float16_t data_scale[];};
layout (binding = 3) readonly buffer Ids {int data_ids[];};
layout (binding = 4) writeonly buffer D {D_TYPE data_d[];};
#else
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};
#endif

layout (push_constant) uniform parameter
{
    uint N;
    uint ne00;
    uint ne20;
    uint mode;
    float alpha;
    float limit;
    uint nb00;
    uint nb01;
    uint nb02;
    uint nb03;
    uint nb10;
    uint nb11;
    uint nb12;
    uint nb13;
    uint nb20;
    uint nb21;
    uint nb22;
    uint nb23;
    uint ne21;
    uint ne22;
    uint misalign_offsets;
    uint ne2_012mp; uint ne2_012L;
    uint ne2_01mp;  uint ne2_01L;
    uint ne2_0mp;   uint ne2_0L;
    uint ne11;
    uint ne12;
    uint nei0;
    uint nbi1;
} p;

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_boffset() { return (p.misalign_offsets >> 8) & 0xFF; }
uint get_doffset() { return p.misalign_offsets & 0xFF; }

// see init_fastdiv_values in ggml-vulkan.cpp
uint fastdiv(uint n, uint mp, uint L) {
    uint msbs, lsbs;
    umulExtended(n, mp, msbs, lsbs);
    return (msbs + n) >> L;
}
