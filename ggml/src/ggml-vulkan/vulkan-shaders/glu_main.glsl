void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;

    if (i >= p.N) {
        return;
    }

    const uint i23 = fastdiv(i, p.ne2_012mp, p.ne2_012L);
    const uint i23_offset = i23 * p.ne22*p.ne21*p.ne20;
    const uint i22 = fastdiv(i - i23_offset, p.ne2_01mp, p.ne2_01L);
    const uint i22_offset = i22*p.ne21*p.ne20;
    const uint i21 = fastdiv(i - i23_offset - i22_offset, p.ne2_0mp, p.ne2_0L);
    const uint i20 = i - i23_offset - i22_offset - i21*p.ne20;

    const uint src_idx_a = get_aoffset() + i23 * p.nb03 + i22 * p.nb02 + i21 * p.nb01 + i20 * p.nb00;
    const uint src_idx_b = get_boffset() + i23 * p.nb13 + i22 * p.nb12 + i21 * p.nb11 + i20 * p.nb10;
    const uint dst_idx = get_doffset() + i23 * p.nb23 + i22 * p.nb22 + i21 * p.nb21 + i20 * p.nb20;

    float value;
    if (p.mode == 0) {
        const uint offset = (p.ne00 / 2) * p.nb00;
        value = op(float(data_a[src_idx_a]), float(data_a[src_idx_a + offset]));
    } else if (p.mode == 1) {
        const uint offset = (p.ne00 / 2) * p.nb00;
        value = op(float(data_a[src_idx_a + offset]), float(data_a[src_idx_a]));
    } else {
        value = op(float(data_a[src_idx_a]), float(data_b[src_idx_b]));
    }

#ifdef ROWS_ID_SCALE
    const uint expert = uint(data_ids[i22 * p.nbi1 + i21]);
    value *= float(data_scale[expert * p.ne20 + i20]);
#endif
    data_d[dst_idx] = D_TYPE(value);
}
