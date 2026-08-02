#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "mmid.cuh"
#include "mma.cuh"
#include "convert.cuh"

using namespace ggml_cuda_mma;

template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false, bool is_multi_token_id = false>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride) {
    const T       * GGML_CUDA_RESTRICT x   = x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int row         = blockIdx.x;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
    const int channel_dst = blockIdx.y;
    const int tid         = threadIdx.x;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    ggml_cuda_pdl_sync();
    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? ids[blockIdx.y + token_idx * ids_stride]            : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
        dst += token_idx*stride_col_dst;
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    const T * gate_x = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        glu_op = fusion.glu_op;

        if (use_gate) {
            gate_x = static_cast<const T *>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float *>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float *>(fusion.gate_bias);
            use_gate_bias = use_gate;
        } else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        gate_x += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    }

    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;
    [[maybe_unused]] float * buf_iw_gate = nullptr;
    if constexpr (has_fusion) {
        buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
    }

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};
    float sumf_gate[ncols_dst];
    if constexpr (has_fusion) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf_gate[j] = 0.0f;
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
        [[maybe_unused]] const float2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const float2 *) gate_x;
            }
        }

        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            float2 tmpx_gate = make_float2(0.0f, 0.0f);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
        [[maybe_unused]] const half2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const half2 *) gate_x;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                float2 tmpx_gate = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = __half22float2(gate_x2[col2]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};
            half2 sumh2_gate[ncols_dst] = {{0.0f, 0.0f}};

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];
                half2 tmpx_gate = make_half2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
//TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        const int * x2 = (const int *) x;
        const int * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const int *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
            int tmpx_gate = 0;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[0]);
                        const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[1]);
                        ggml_cuda_mad(sumf_gate[j], tmpx0_gate, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx1_gate, tmpy.y);
                    }
                }
            }
        }
#else
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        [[maybe_unused]] const nv_bfloat162 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162 *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
            [[maybe_unused]] nv_bfloat162 tmpx_gate;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    ggml_cuda_pdl_lc();
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if constexpr (has_fusion) {
            if (use_gate) {
                sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
            }
        }

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid/warp_size] = sumf_gate[j];
                }
            }
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[j] = buf_iw_gate[tid];
                        sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                    }
                }
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    float value = sumf[tid];

    if constexpr (has_fusion) {
        if (use_bias) {
            value += x_bias[tid*stride_col_dst + row];
        }

        if (use_gate) {
            float gate_value = sumf_gate[tid];
            if (use_gate_bias) {
                gate_value += gate_bias[tid*stride_col_dst + row];
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    value *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    value *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI: {
                    value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                    break;
                }
                default:
                    break;
            }
        }
    }

    dst[tid*stride_col_dst + row] = value;

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, glu_op, gate_x, x_bias, gate_bias, sumf_gate);
    }
}

template<typename T, typename type_acc, int ncols_dst, int block_size, bool is_multi_token_id = false>
static void mul_mat_vec_f_switch_fusion(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const uint3 nchannels_y,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const dim3 & block_dims, const dim3 & block_nums, const int nbytes_shared, const int ids_stride, const cudaStream_t stream) {

    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, nbytes_shared, stream};

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>, launch_params,
                x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
       }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>, launch_params,
        x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);

}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, cudaStream_t stream) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);
    const dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);
    switch (block_size_best) {
        case   32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   64: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case   96: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  128: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  160: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  192: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  224: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, stream);
        } break;
        default: {
            GGML_ABORT("fatal error");
        } break;
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, cudaStream_t stream) {

    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst, true>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_cuda<T, type_acc, 1>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 2:
            launch_mul_mat_vec_f_cuda<T, type_acc, 2>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 3:
            launch_mul_mat_vec_f_cuda<T, type_acc, 3>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 4:
            launch_mul_mat_vec_f_cuda<T, type_acc, 4>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 5:
            launch_mul_mat_vec_f_cuda<T, type_acc, 5>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 6:
            launch_mul_mat_vec_f_cuda<T, type_acc, 6>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 7:
            launch_mul_mat_vec_f_cuda<T, type_acc, 7>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 8:
            launch_mul_mat_vec_f_cuda<T, type_acc, 8>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream) {

    if constexpr(std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
}

// Native packed SIGN1 x F32 path. Each warp owns one output row, reads sign
// bits in place, and flips the F32 sign bit directly. RHS values are loaded once
// into shared memory and reused across eight output rows; no sign matrix or
// quantized RHS representation is materialized.
static __global__ void mul_mat_tile_sign1_f32_mask(
        const block_sign1 * __restrict__ x, const float * __restrict__ y,
        const int32_t * __restrict__ ids_src_compact, const int32_t * __restrict__ ids_dst_compact,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
        const int ncols, const int nrows, const int ncols_dst, const int ncol_tiles,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sis1, const int nchannels_x, const int nchannels_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x,
        const int stride_sample_y, const int stride_sample_dst) {
    constexpr int warp_size = 32;
    constexpr int nwarps = 16;
    constexpr int rows_per_block = nwarps;
    constexpr int cols_per_block = 16;
    constexpr int k_tile = 128;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int linear_tid = warp*warp_size + lane;
    const int row0 = blockIdx.x*rows_per_block;
    const int row = row0 + warp;

    __shared__ int y_offsets[cols_per_block];
    __shared__ int dst_offsets[cols_per_block];
    __shared__ int x_offset;
    __shared__ int valid_cols;
    __shared__ float y_tile[cols_per_block][k_tile];

    if (linear_tid == 0) {
        if (ids_src_compact != nullptr) {
            const int expert = blockIdx.y;
            const int expert_start = expert_bounds[expert];
            const int expert_end = expert_bounds[expert + 1];
            const int col_base = blockIdx.z*cols_per_block;
            const int ncols_expert = expert_end - expert_start;
            valid_cols = max(0, min(cols_per_block, ncols_expert - col_base));
            x_offset = expert*stride_channel_x + row0*stride_row_x;
            for (int j = 0; j < valid_cols; ++j) {
                const int compact = expert_start + col_base + j;
                const int src_entry = ids_src_compact[compact];
                const int token = src_entry/sis1;
                const int channel = src_entry - token*sis1;
                const int dst_entry = ids_dst_compact[compact];
                const int dst_token = dst_entry/nchannels_dst;
                const int dst_channel = dst_entry - dst_token*nchannels_dst;
                y_offsets[j] = channel*stride_channel_y + token*stride_col_y;
                dst_offsets[j] = dst_channel*stride_channel_dst + dst_token*stride_col_dst;
            }
        } else {
            const int sample_dst = blockIdx.z/ncol_tiles;
            const int col_base = (blockIdx.z - sample_dst*ncol_tiles)*cols_per_block;
            const int channel_dst = blockIdx.y;
            const int channel_x = channel_dst/(nchannels_dst/nchannels_x);
            const int sample_x = sample_dst/(nsamples_dst/nsamples_x);
            valid_cols = min(cols_per_block, ncols_dst - col_base);
            x_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
            for (int j = 0; j < valid_cols; ++j) {
                y_offsets[j] = sample_dst*stride_sample_y + channel_dst*stride_channel_y +
                    (col_base + j)*stride_col_y;
                dst_offsets[j] = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst +
                    (col_base + j)*stride_col_dst;
            }
        }
    }
    __syncthreads();
    if (valid_cols == 0) {
        return;
    }

    float sums[cols_per_block] = {};
    const block_sign1 * xr = x + x_offset + warp*stride_row_x;
    for (int k0 = 0; k0 < ncols; k0 += k_tile) {
        for (int idx = linear_tid; idx < cols_per_block*k_tile; idx += nwarps*warp_size) {
            const int j = idx/k_tile;
            const int k = idx - j*k_tile;
            y_tile[j][k] = j < valid_cols && k0 + k < ncols ? y[y_offsets[j] + k0 + k] : 0.0f;
        }
        __syncthreads();

        if (row < nrows) {
#pragma unroll
            for (int k = lane; k < k_tile; k += warp_size) {
                if (k0 + k >= ncols) {
                    break;
                }
                const uint64_t bits = xr[(k0 + k)/QK_SIGN1].qs;
                const uint32_t sign_mask = ((bits >> ((k0 + k) % QK_SIGN1)) & 1ULL) ? 0x80000000u : 0u;
#pragma unroll
                for (int j = 0; j < cols_per_block; ++j) {
                    const float value = __uint_as_float(__float_as_uint(y_tile[j][k]) ^ sign_mask);
                    sums[j] += value;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < cols_per_block; ++j) {
        sums[j] = warp_reduce_sum<warp_size>(sums[j]);
        if (lane == 0 && row < nrows && j < valid_cols) {
            dst[dst_offsets[j] + row] = sums[j];
        }
    }
}

// Native packed SIGN1 x FP16 path. Signs are expanded only into shared-memory
// WMMA tiles; the full weight matrix is never unpacked or materialized.
template <typename TY>
static __global__ void mul_mat_tile_sign1_mma(
        const block_sign1 * __restrict__ x, const TY * __restrict__ y,
        const int32_t * __restrict__ ids_src_compact, const int32_t * __restrict__ ids_dst_compact,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
        const int ncols_pairs, const int nrows, const int ncols_dst, const int ncol_tiles, const int stride_row_x,
        const int stride_col_y, const int stride_col_dst, const int stride_channel_x,
        const int stride_channel_y, const int stride_channel_dst, const int sis1,
        const int nchannels_x, const int nchannels_dst, const int nsamples_x,
        const int nsamples_dst, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst) {
    constexpr int rows_per_block = 32;
    constexpr int cols_per_block = 16;
    constexpr int nwarps = 8;
    constexpr int warp_size = 32;
    constexpr int tile_k_padded = warp_size + 4;
    using tile_A = tile<16, 8, half2, get_input_data_layout()>;
    using tile_B = tile<16, 8, half2, get_input_data_layout()>;
    using tile_C = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
    constexpr int ntA = rows_per_block/tile_A::I;

    const int row0 = blockIdx.x*rows_per_block;
    __shared__ int y_offsets[cols_per_block];
    __shared__ int dst_offsets[cols_per_block];
    __shared__ int valid_cols;
    __shared__ int x_offset;
    extern __shared__ char data_mma[];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (ids_src_compact != nullptr) {
            const int expert = blockIdx.y;
            const int expert_start = expert_bounds[expert];
            const int expert_end = expert_bounds[expert + 1];
            const int col_base = blockIdx.z*cols_per_block;
            const int ncols_expert = expert_end - expert_start;
            valid_cols = max(0, min(cols_per_block, ncols_expert - col_base));
            x_offset = expert*stride_channel_x + row0*stride_row_x;
            for (int j = 0; j < valid_cols; ++j) {
                const int compact = expert_start + col_base + j;
                const int src_entry = ids_src_compact[compact];
                const int token = src_entry/sis1;
                const int channel = src_entry - token*sis1;
                const int dst_entry = ids_dst_compact[compact];
                const int dst_token = dst_entry/nchannels_dst;
                const int dst_channel = dst_entry - dst_token*nchannels_dst;
                y_offsets[j] = channel*stride_channel_y + token*stride_col_y;
                dst_offsets[j] = dst_channel*stride_channel_dst + dst_token*stride_col_dst;
            }
        } else {
            const int sample_dst = blockIdx.z/ncol_tiles;
            const int col_base = (blockIdx.z - sample_dst*ncol_tiles)*cols_per_block;
            const int channel_dst = blockIdx.y;
            const int channel_x = channel_dst/(nchannels_dst/nchannels_x);
            const int sample_x = sample_dst/(nsamples_dst/nsamples_x);
            valid_cols = min(cols_per_block, ncols_dst - col_base);
            x_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;
            for (int j = 0; j < valid_cols; ++j) {
                y_offsets[j] = sample_dst*stride_sample_y + channel_dst*stride_channel_y +
                    (col_base + j)*stride_col_y;
                dst_offsets[j] = sample_dst*stride_sample_dst + channel_dst*stride_channel_dst +
                    (col_base + j)*stride_col_dst;
            }
        }
    }
    __syncthreads();
    if (valid_cols == 0) {
        return;
    }

    tile_C C[ntA];
    half2 * tile_xy = reinterpret_cast<half2 *>(data_mma) +
        threadIdx.y*(tile_A::I*tile_k_padded);
    const block_sign1 * xe = x + x_offset;

    for (int col = threadIdx.y*warp_size + threadIdx.x;
         col < ncols_pairs; col += nwarps*warp_size) {
        const int scalar_col = 2*col;
        tile_A A[ntA][warp_size/tile_A::J];
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int i = 0; i < tile_A::I; ++i) {
                const int global_row = row0 + itA*tile_A::I + i;
                half2 signs = make_half2(0.0f, 0.0f);
                if (global_row < nrows) {
                    const block_sign1 * row = xe + (itA*tile_A::I + i)*stride_row_x;
                    const uint64_t bits = row[scalar_col/QK_SIGN1].qs;
                    const float s0 = ((bits >> (scalar_col % QK_SIGN1)) & 1ULL) ? -1.0f : 1.0f;
                    const float s1 = ((bits >> ((scalar_col + 1) % QK_SIGN1)) & 1ULL) ? -1.0f : 1.0f;
                    signs = make_half2(s0, s1);
                }
                tile_xy[i*tile_k_padded + threadIdx.x] = signs;
            }
#pragma unroll
            for (int k0 = 0; k0 < warp_size; k0 += tile_A::J) {
                load_ldmatrix(A[itA][k0/tile_A::J], tile_xy + k0, tile_k_padded);
            }
        }
#pragma unroll
        for (int j = 0; j < tile_B::I; ++j) {
            half2 v = make_half2(0.0f, 0.0f);
            if (j < valid_cols) {
                if constexpr (std::is_same_v<TY, half>) {
                    v = *reinterpret_cast<const half2 *>(y + y_offsets[j] + scalar_col);
                } else {
                    const float2 vf = *reinterpret_cast<const float2 *>(y + y_offsets[j] + scalar_col);
                    v = ggml_cuda_cast<half2>(vf);
                }
            }
            tile_xy[j*tile_k_padded + threadIdx.x] = v;
        }
#pragma unroll
        for (int k0 = 0; k0 < warp_size; k0 += tile_B::J) {
            tile_B B;
            load_ldmatrix(B, tile_xy + k0, tile_k_padded);
#pragma unroll
            for (int itA = 0; itA < ntA; ++itA) {
                mma(C[itA], A[itA][k0/tile_A::J], B);
            }
        }
    }

    float * combine = reinterpret_cast<float *>(data_mma);
    constexpr int kiw = nwarps*rows_per_block + 4;
    __syncthreads();
#pragma unroll
    for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
            const int i = threadIdx.y*rows_per_block + itA*tile_C::I + tile_C::get_i(l);
            const int j = tile_C::get_j(l);
            combine[j*kiw + i] = C[itA].x[l];
        }
    }
    __syncthreads();

    for (int j0 = 0; j0 < cols_per_block; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j >= valid_cols) {
            continue;
        }
        float sum = 0.0f;
#pragma unroll
        for (int iw = 0; iw < nwarps; ++iw) {
            sum += combine[j*kiw + iw*rows_per_block + threadIdx.x];
        }
        if (row0 + threadIdx.x < nrows) {
            dst[dst_offsets[j] + row0 + threadIdx.x] = sum;
        }
    }
}

template <typename TY>
static void launch_mul_mat_tile_sign1_f(
        ggml_backend_cuda_context & ctx, const block_sign1 * x, const TY * y,
        const int32_t * ids, float * dst, const int64_t ncols, const int64_t nrows,
        const int64_t ncols_dst, const int64_t stride_row_x, const int64_t stride_col_y,
        const int64_t stride_col_dst, const int64_t nchannels_x, const int64_t nchannels_y,
        const int64_t nchannels_dst, const int64_t stride_channel_x,
        const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t nsamples_x, const int64_t nsamples_dst, const int64_t stride_sample_x,
        const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, const int64_t sis1, cudaStream_t stream) {


    if (ids != nullptr) {
        const int64_t n_experts = nchannels_x;
        const int64_t n_expert_used = nchannels_dst;
        const int64_t ne_get_rows = ncols_dst*n_expert_used;
        ggml_cuda_pool_alloc<int32_t> ids_src_compact(ctx.pool(), ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> ids_dst_compact(ctx.pool(), ne_get_rows);
        ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
        ggml_cuda_launch_mm_ids_helper(ids, ids_src_compact.get(), ids_dst_compact.get(),
            expert_bounds.get(), n_experts, ncols_dst, n_expert_used, nchannels_y,
            ids_stride, sis1, false, stream);
        if constexpr (std::is_same_v<TY, float>) {
            const dim3 blocks((nrows + 15)/16, n_experts, (ncols_dst + 15)/16);
            const dim3 threads(32, 16, 1);
            mul_mat_tile_sign1_f32_mask<<<blocks, threads, 0, stream>>>(
                x, y, ids_src_compact.get(), ids_dst_compact.get(), expert_bounds.get(), dst,
                ncols, nrows, ncols_dst, (ncols_dst + 15)/16, stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, sis1, nchannels_x,
                nchannels_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y,
                stride_sample_dst);
        } else {
            GGML_ASSERT(ncols % 2 == 0);
            const dim3 blocks((nrows + 31)/32, n_experts, (ncols_dst + 15)/16);
            const dim3 threads(32, 8, 1);
            constexpr int shared = 8*16*(32 + 4)*sizeof(half2);
            mul_mat_tile_sign1_mma<TY><<<blocks, threads, shared, stream>>>(
                x, y, ids_src_compact.get(), ids_dst_compact.get(), expert_bounds.get(), dst,
                ncols/2, nrows, ncols_dst, (ncols_dst + 15)/16, stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, sis1, nchannels_x,
                nchannels_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y,
                stride_sample_dst);
        }
        return;
    } else {
        if constexpr (std::is_same_v<TY, float>) {
            const dim3 blocks((nrows + 15)/16, nchannels_dst, ((ncols_dst + 15)/16)*nsamples_dst);
            const dim3 threads(32, 16, 1);
            mul_mat_tile_sign1_f32_mask<<<blocks, threads, 0, stream>>>(
                x, y, nullptr, nullptr, nullptr, dst, ncols, nrows, ncols_dst,
                (ncols_dst + 15)/16, stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, 1, nchannels_x,
                nchannels_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y,
                stride_sample_dst);
        } else {
            GGML_ASSERT(ncols % 2 == 0);
            const dim3 blocks((nrows + 31)/32, nchannels_dst, ((ncols_dst + 15)/16)*nsamples_dst);
            const dim3 threads(32, 8, 1);
            constexpr int shared = 8*16*(32 + 4)*sizeof(half2);
            mul_mat_tile_sign1_mma<TY><<<blocks, threads, shared, stream>>>(
                x, y, nullptr, nullptr, nullptr, dst, ncols/2, nrows, ncols_dst,
                (ncols_dst + 15)/16, stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst, 1, nchannels_x,
                nchannels_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y,
                stride_sample_dst);
        }
        return;
    }
}

template <typename TY, bool has_ids>
static __global__ void mul_mat_vec_sign1_f(
        const block_sign1 * __restrict__ x, const TY * __restrict__ y,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ncols, const int nrows, const int ncols_dst, const int stride_row_x,
        const int stride_col_y, const int stride_col_dst, const int nchannels_y, const int channel_ratio,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst, const int ids_stride) {
    constexpr int block_size = 256;
    constexpr int rows_per_block = 16;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x*rows_per_block;
    const int channel_dst = blockIdx.y;
    const int token_or_sample = blockIdx.z;

    int channel_x;
    int channel_y;
    int sample_x;
    int sample_y;
    int sample_dst;
    if constexpr (has_ids) {
        channel_x = ids[channel_dst + token_or_sample*ids_stride];
        channel_y = channel_dst % nchannels_y;
        sample_x = 0;
        sample_y = 0;
        sample_dst = 0;
    } else {
        channel_x = channel_dst / channel_ratio;
        channel_y = channel_dst;
        sample_dst = token_or_sample / ncols_dst;
        sample_x = sample_dst / sample_ratio;
        sample_y = sample_dst;
    }

    const int col_dst = has_ids ? token_or_sample : token_or_sample % ncols_dst;
    const block_sign1 * xr = x + int64_t(sample_x)*stride_sample_x +
        channel_x*stride_channel_x + row0*stride_row_x;
    const TY * yr = y + int64_t(sample_y)*stride_sample_y + channel_y*stride_channel_y +
        int64_t(col_dst)*stride_col_y;
    float * dr = dst + int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst +
        int64_t(col_dst)*stride_col_dst;

    // Process one byte of packed signs per task. This keeps all threads busy for
    // the common K=2048 case and loads each 64-bit sign word only once per row
    // and 8 input values, rather than redundantly reloading it for every value.
    constexpr int values_per_task = 8;
    static_assert(QK_SIGN1 % values_per_task == 0, "SIGN1 block must be byte-addressable");
    float sums[rows_per_block] = {0.0f};
    const int ntasks = (ncols + values_per_task - 1)/values_per_task;
    for (int task = tid; task < ntasks; task += block_size) {
        const int col0 = task*values_per_task;
        const int sign_block = col0/QK_SIGN1;
        const int bit0 = col0 % QK_SIGN1;
        uint64_t bits[rows_per_block];
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            bits[r] = row0 + r < nrows ? xr[r*stride_row_x + sign_block].qs : 0;
        }
#pragma unroll
        for (int j = 0; j < values_per_task; ++j) {
            const int col = col0 + j;
            if (col >= ncols) {
                break;
            }
            const float value = ggml_cuda_cast<float>(yr[col]);
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                if (row0 + r < nrows) {
                    sums[r] += ((bits[r] >> (bit0 + j)) & 1ULL) ? -value : value;
                }
            }
        }
    }

    const int lane = tid % warp_size;
    const int warp = tid / warp_size;
    constexpr int max_warps = 8;
    const int nwarps = block_size / warp_size;
    __shared__ float warp_sums[rows_per_block][max_warps];
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        sums[r] = warp_reduce_sum<warp_size>(sums[r]);
        if (lane == 0) {
            warp_sums[r][warp] = sums[r];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int r = 0; r < rows_per_block; ++r) {
            float total = lane < nwarps ? warp_sums[r][lane] : 0.0f;
            total = warp_reduce_sum<warp_size>(total);
            if (lane == 0 && row0 + r < nrows) {
                dr[row0 + r] = total;
            }
        }
    }
}

template <typename TY>
static void launch_mul_mat_vec_sign1_f(
        const block_sign1 * x, const TY * y, const int32_t * ids, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
        const int64_t nsamples_x, const int64_t nsamples_dst, const int64_t stride_sample_x,
        const int64_t stride_sample_y, const int64_t stride_sample_dst, const int64_t ids_stride,
        cudaStream_t stream) {
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(nsamples_dst % nsamples_x == 0);
    const dim3 block_dims(256, 1, 1);
    constexpr int rows_per_block = 16;
    const dim3 block_nums((nrows + rows_per_block - 1)/rows_per_block,
        nchannels_dst, ids ? ncols_dst : ncols_dst*nsamples_dst);
    const int channel_ratio = ids ? 1 : nchannels_dst / nchannels_x;
    const int sample_ratio = nsamples_dst / nsamples_x;
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
    if (ids) {
        ggml_cuda_kernel_launch(mul_mat_vec_sign1_f<TY, true>, launch_params,
            x, y, ids, dst, ncols, nrows, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
            nchannels_y, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_sign1_f<TY, false>, launch_params,
            x, y, ids, dst, ncols, nrows, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
            nchannels_y, channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
    }
}

void ggml_cuda_mul_mat_vec_sign1_f(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_SIGN1);
    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;
    const size_t ts_y = ggml_type_size(src1->type);
    const size_t ts_dst = sizeof(float);
    const size_t ts_x = sizeof(block_sign1);
    GGML_ASSERT(nb00 == ts_x);
    GGML_ASSERT(nb10 == ts_y);
    GGML_ASSERT(nb0 == ts_dst);

    const int64_t sx1 = nb01 / ts_x;
    const int64_t sx2 = nb02 / ts_x;
    const int64_t sx3 = nb03 / ts_x;
    const int64_t sy1 = nb11 / ts_y;
    const int64_t sy2 = nb12 / ts_y;
    const int64_t sy3 = nb13 / ts_y;
    const int64_t sd1 = nb1 / ts_dst;
    const int64_t sd2 = nb2 / ts_dst;
    const int64_t sd3 = nb3 / ts_dst;

    const int64_t ncols_dst = ids ? ne2 : ne1;
    const int64_t nchannels_y = ids ? ne11 : ne12;
    const int64_t nchannels_dst = ids ? ne1 : ne2;
    const int64_t stride_col_y = ids ? sy2 : sy1;
    const int64_t stride_col_dst = ids ? sd2 : sd1;
    const int64_t stride_channel_y = ids ? sy1 : sy2;
    const int64_t stride_channel_dst = ids ? sd1 : sd2;
    const int64_t ids_stride = ids ? ids->nb[1] / sizeof(int32_t) : 0;
    const int64_t sis1 = ids ? src1->nb[2] / src1->nb[1] : 1;
    const int32_t * ids_d = ids ? static_cast<const int32_t *>(ids->data) : nullptr;

    if (src1->type == GGML_TYPE_F16) {
        if (ncols_dst > 1) {
            launch_mul_mat_tile_sign1_f(ctx, static_cast<const block_sign1 *>(src0->data),
                static_cast<const half *>(src1->data), ids_d, static_cast<float *>(dst->data),
                ne00, ne01, ncols_dst, sx1, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, sx2, stride_channel_y, stride_channel_dst,
                ne03, ne3, sx3, sy3, sd3, ids_stride, sis1, ctx.stream());
        } else {
            launch_mul_mat_vec_sign1_f(static_cast<const block_sign1 *>(src0->data),
                static_cast<const half *>(src1->data), ids_d, static_cast<float *>(dst->data),
                ne00, ne01, ncols_dst, sx1, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, sx2, stride_channel_y, stride_channel_dst,
                ne03, ne3, sx3, sy3, sd3, ids_stride, ctx.stream());
        }
    } else {
        if (ncols_dst > 1) {
            launch_mul_mat_tile_sign1_f(ctx, static_cast<const block_sign1 *>(src0->data),
                static_cast<const float *>(src1->data), ids_d, static_cast<float *>(dst->data),
                ne00, ne01, ncols_dst, sx1, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, sx2, stride_channel_y, stride_channel_dst,
                ne03, ne3, sx3, sy3, sd3, ids_stride, sis1, ctx.stream());
        } else {
            launch_mul_mat_vec_sign1_f(static_cast<const block_sign1 *>(src0->data),
                static_cast<const float *>(src1->data), ids_d, static_cast<float *>(dst->data),
                ne00, ne01, ncols_dst, sx1, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, sx2, stride_channel_y, stride_channel_dst,
                ne03, ne3, sx3, sy3, sd3, ids_stride, ctx.stream());
        }
    }
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 <= MMVF_MAX_BATCH_SIZE);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  =  dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t stride_col_y       = ne10;
    const int64_t stride_col_dst     = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    ggml_cuda_mm_fusion_args_device empty{};
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED_VARS(ctx, src1, dst, src1_ddq_i, src1_ncols, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        default:
            return false;
    }
}
