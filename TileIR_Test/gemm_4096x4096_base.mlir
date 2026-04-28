// cuda_tile.module @gemm_square_4096_tile_64x64_module {
//     entry @gemm_square_4096_tile_64x64_kernel(
//         %a_ptr_base_scalar: tile<ptr<f32>>,
//         %b_ptr_base_scalar: tile<ptr<f32>>,
//         %c_ptr_base_scalar: tile<ptr<f32>>
//     ) {
//         %block_x_index, %block_y_index, %block_z_index = get_tile_block_id : tile<i32>

//         %m_tile_size    = cuda_tile.constant <i32: 64> : tile<i32>
//         %m_stride_factor = cuda_tile.constant <i32: 64> : tile<64x64xi32>
//         %k_tile_size    = cuda_tile.constant <i32: 64> : tile<i32>
//         %num_k_tiles    = cuda_tile.constant <i32: 64> : tile<i32>

//         %range_start = cuda_tile.constant <i32: 0> : tile<i32>
//         %range_step  = cuda_tile.constant <i32: 1> : tile<i32>
//         %init_accum  = cuda_tile.constant <f32: 0.000000e+00> : tile<64x64xf32>

//         %tile_size_range = cuda_tile.iota : tile<64xi32>

//         // ----------------------------------------------------------------
//         // 计算 A 的 tile offset 矩阵
//         // a_tile[m, k] = (block_x * 64 + m) * 64 + k
//         // ----------------------------------------------------------------
//         %a_tile_base          = cuda_tile.muli %block_x_index, %m_tile_size : tile<i32>
//         %a_tile_base_reshape  = cuda_tile.reshape %a_tile_base : tile<i32> -> tile<1xi32>
//         %a_tile_base_tensor   = cuda_tile.broadcast %a_tile_base_reshape : tile<1xi32> -> tile<64xi32>
//         %m_offsets_vec        = cuda_tile.addi %a_tile_base_tensor, %tile_size_range : tile<64xi32>

//         %m_offsets_matrix    = cuda_tile.reshape %m_offsets_vec : tile<64xi32> -> tile<64x1xi32>
//         %m_offsets_broadcast = cuda_tile.broadcast %m_offsets_matrix : tile<64x1xi32> -> tile<64x64xi32>
//         %m_offsets           = cuda_tile.muli %m_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

//         %ak_offsets_matrix    = cuda_tile.reshape %tile_size_range : tile<64xi32> -> tile<1x64xi32>
//         %ak_offsets_broadcast = cuda_tile.broadcast %ak_offsets_matrix : tile<1x64xi32> -> tile<64x64xi32>

//         // K 方向步长为 1，不需要乘 stride
//         %a_tile_offsets = cuda_tile.addi %m_offsets, %ak_offsets_broadcast : tile<64x64xi32>

//         // ----------------------------------------------------------------
//         // 计算 B 的 tile offset 矩阵
//         // b_tile[k, n] = k * 4096 + (block_y * 64 + n)
//         // ----------------------------------------------------------------
//         %b_tile_base         = cuda_tile.muli %block_y_index, %m_tile_size : tile<i32>
//         %b_tile_base_reshape = cuda_tile.reshape %b_tile_base : tile<i32> -> tile<1xi32>
//         %b_tile_base_tensor  = cuda_tile.broadcast %b_tile_base_reshape : tile<1xi32> -> tile<64xi32>
//         %n_offsets_vec       = cuda_tile.addi %b_tile_base_tensor, %tile_size_range : tile<64xi32>

//         // K 方向：bk_offsets = arange(0, 64)[:, None] * 4096
//         %bk_offsets_matrix    = cuda_tile.reshape %tile_size_range : tile<64xi32> -> tile<64x1xi32>
//         %bk_offsets_broadcast = cuda_tile.broadcast %bk_offsets_matrix : tile<64x1xi32> -> tile<64x64xi32>
//         %bk_offsets           = cuda_tile.muli %bk_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

//         // N 方向：n_offsets = (block_y * 64 + arange(0, 64))[None, :]，步长为 1
//         %n_offsets_matrix    = cuda_tile.reshape %n_offsets_vec : tile<64xi32> -> tile<1x64xi32>
//         %n_offsets_broadcast = cuda_tile.broadcast %n_offsets_matrix : tile<1x64xi32> -> tile<64x64xi32>

//         // ✅ Fix 2: addi，不是 muli
//         %b_tile_offsets = cuda_tile.addi %bk_offsets, %n_offsets_broadcast : tile<64x64xi32>

//         // ----------------------------------------------------------------
//         // 构造 A、B 的指针张量
//         // ----------------------------------------------------------------
//         %a_ptr_base_tensor = cuda_tile.reshape %a_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
//         %a_ptr             = cuda_tile.broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
//         %a_tile_ptr        = cuda_tile.offset %a_ptr, %a_tile_offsets :
//                                  tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//         %b_ptr_base_tensor = cuda_tile.reshape %b_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
//         %b_ptr             = cuda_tile.broadcast %b_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
//         %b_tile_ptr        = cuda_tile.offset %b_ptr, %b_tile_offsets :
//                                  tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//         // ----------------------------------------------------------------
//         // K 方向循环，累加 64 个 tile 的部分积
//         // ✅ Fix 1: step 改为 %range_step（值为 1），上界为 %num_k_tiles（64）
//         // ----------------------------------------------------------------
//         %C_tile, %a_ptr_final, %b_ptr_final = for %k in (%range_start to %num_k_tiles, step %range_step) : tile<i32>
//             iter_values(
//                 %acc_prev       = %init_accum,
//                 %a_tile_ptr_prev = %a_tile_ptr,
//                 %b_tile_ptr_prev = %b_tile_ptr
//             ) -> (tile<64x64xf32>, tile<64x64xptr<f32>>, tile<64x64xptr<f32>>)
//         {
//             %A_tile, %token_a = cuda_tile.load_ptr_tko weak %a_tile_ptr_prev :
//                 tile<64x64xptr<f32>> -> tile<64x64xf32>, token

//             %B_tile, %token_b = cuda_tile.load_ptr_tko weak %b_tile_ptr_prev :
//                 tile<64x64xptr<f32>> -> tile<64x64xf32>, token

//             %C_tile_acc = cuda_tile.mmaf %A_tile, %B_tile, %acc_prev :
//                 tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>

//             // 每次迭代 A 向右移动 64 列（K 方向），B 向下移动 64 行（K 方向）
//             %block_size_a   = cuda_tile.constant <i32: 64> : tile<64x64xi32>
//             %block_size_b   = cuda_tile.constant <i32: 262144> : tile<64x64xi32>  // 64 * 4096

//             %a_tile_ptr_next = cuda_tile.offset %a_tile_ptr_prev, %block_size_a :
//                 tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
//             %b_tile_ptr_next = cuda_tile.offset %b_tile_ptr_prev, %block_size_b :
//                 tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//             continue %C_tile_acc, %a_tile_ptr_next, %b_tile_ptr_next :
//                 tile<64x64xf32>, tile<64x64xptr<f32>>, tile<64x64xptr<f32>>
//         }

//         // ----------------------------------------------------------------
//         // 计算 C 的 tile offset 矩阵并写回
//         // c_tile[m, n] = (block_x * 64 + m) * 4096 + (block_y * 64 + n)
//         // ----------------------------------------------------------------
//         %c_tile_x_start        = cuda_tile.muli %block_x_index, %m_tile_size : tile<i32>
//         %c_tile_x_start_reshape = cuda_tile.reshape %c_tile_x_start : tile<i32> -> tile<1xi32>
//         %c_tile_x_start_tensor  = cuda_tile.broadcast %c_tile_x_start_reshape : tile<1xi32> -> tile<64xi32>
//         %c_tile_x_offsets_vec   = cuda_tile.addi %c_tile_x_start_tensor, %tile_size_range : tile<64xi32>

//         // ✅ Fix 3: 使用 block_y_index 计算 N 方向起点
//         %c_tile_y_start        = cuda_tile.muli %block_y_index, %m_tile_size : tile<i32>
//         %c_tile_y_start_reshape = cuda_tile.reshape %c_tile_y_start : tile<i32> -> tile<1xi32>
//         %c_tile_y_start_tensor  = cuda_tile.broadcast %c_tile_y_start_reshape : tile<1xi32> -> tile<64xi32>
//         %c_tile_y_offsets_vec   = cuda_tile.addi %c_tile_y_start_tensor, %tile_size_range : tile<64xi32>

//         // M 方向乘以行 stride（4096）
//         %c_tile_x_offsets_matrix    = cuda_tile.reshape %c_tile_x_offsets_vec : tile<64xi32> -> tile<64x1xi32>
//         %c_tile_x_offsets_broadcast = cuda_tile.broadcast %c_tile_x_offsets_matrix : tile<64x1xi32> -> tile<64x64xi32>
//         %c_tile_x_offsets           = cuda_tile.muli %c_tile_x_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

//         // N 方向步长为 1
//         %c_tile_y_offsets_matrix    = cuda_tile.reshape %c_tile_y_offsets_vec : tile<64xi32> -> tile<1x64xi32>
//         %c_tile_y_offsets_broadcast = cuda_tile.broadcast %c_tile_y_offsets_matrix : tile<1x64xi32> -> tile<64x64xi32>

//         // ✅ Fix 4: addi，不是 muli
//         %c_tile_offsets = cuda_tile.addi %c_tile_x_offsets, %c_tile_y_offsets_broadcast : tile<64x64xi32>

//         %c_ptr_base_tensor = cuda_tile.reshape %c_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
//         %c_ptr             = cuda_tile.broadcast %c_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
//         %c_tile_ptr        = cuda_tile.offset %c_ptr, %c_tile_offsets :
//                                  tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//         cuda_tile.store_ptr_tko weak %c_tile_ptr, %C_tile :
//             tile<64x64xptr<f32>>, tile<64x64xf32> -> token
//     }
// }


// =============================================================================
// gemm_v0_baseline.tileir
//
// SOURCE: NVIDIA Tile IR Documentation, Appendix §11.1.7 (verbatim, lightly
//         formatted for readability).
//
// SHAPE:  C[M×N] += A[M×K] @ B[K×N]   (fp32 += fp16 * fp16)
// TILE:   Each tile block computes a 128×128 output tile of C.
//         Per K-step: loads A(128×64), B(64×128), one mmaf → acc(128×128).
//
// SCHEDULE (per K-step):
//   load_A(128×64) ──┐
//                    ├──► mmaf(128×128) ──► continue
//   load_B(64×128) ──┘
//
// NOTE: There are NO explicit token chains between the two loads.
//       However, there is only ONE mmaf per K-step.  The compiler has
//       limited ILP to exploit within a single mmaf of this size.
// =============================================================================

cuda_tile.module @gemm_baseline_module {
    entry @gemm_baseline_kernel(
        %A_ptr:     !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %B_ptr:     !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %C_ptr:     !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %M:         !cuda_tile.tile<i32>,
        %N:         !cuda_tile.tile<i32>,
        %K:         !cuda_tile.tile<i32>,
        %stride_ak: !cuda_tile.tile<i32>,
        %stride_bn: !cuda_tile.tile<i32>,
        %stride_cm: !cuda_tile.tile<i32>
    ) {
        // ── Alignment hints ──────────────────────────────────────────────────
        %A_ptr_a     = assume #cuda_tile.div_by<16>, %A_ptr     : tile<ptr<f16>>
        %B_ptr_a     = assume #cuda_tile.div_by<16>, %B_ptr     : tile<ptr<f16>>
        %C_ptr_a     = assume #cuda_tile.div_by<16>, %C_ptr     : tile<ptr<f32>>
        %stride_ak_a = assume #cuda_tile.div_by<8>,  %stride_ak : tile<i32>
        %stride_bn_a = assume #cuda_tile.div_by<8>,  %stride_bn : tile<i32>
        %stride_cm_a = assume #cuda_tile.div_by<8>,  %stride_cm : tile<i32>

        // ── Constants ────────────────────────────────────────────────────────
        %i0  = constant <i32: 0>           : !cuda_tile.tile<i32>
        %i1  = constant <i32: 1>           : !cuda_tile.tile<i32>
        %cst = constant <f32: 0.000000e+00>: !cuda_tile.tile<128x128xf32>

        // ── Tensor views (structured pointers) ───────────────────────────────
        // A is stored transposed: layout (K × M)
        %A = make_tensor_view %A_ptr_a, shape = [%K, %M], strides = [%stride_ak, 1]
            : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        // B is stored transposed: layout (N × K)
        %B = make_tensor_view %B_ptr_a, shape = [%N, %K], strides = [%stride_bn, 1]
            : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
        // C: layout (M × N)
        %C = make_tensor_view %C_ptr_a, shape = [%M, %N], strides = [%stride_cm, 1]
            : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

        // ── Partition views ───────────────────────────────────────────────────
        // A_block[m, k] → tile<128×64xf16>   (dim_map=[1,0] handles the transpose)
        %A_block = make_partition_view %A :
            partition_view<tile=(128x64),  tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        // B_block[k, n] → tile<64×128xf16>
        %B_block = make_partition_view %B :
            partition_view<tile=(64x128),  tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
        // C_block[m, n] → tile<128×128xf32>
        %C_block = make_partition_view %C :
            partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>

        // ── Grid coordinates ──────────────────────────────────────────────────
        %bidx, %bidy, %bidz = get_tile_block_id : tile<i32>

        // ── K-dimension tile count ────────────────────────────────────────────
        // get_index_space_shape returns (M_tiles, K_tiles); we want #1 = K/64
        %mk_len:2 = get_index_space_shape %A_block :
            partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>
            -> tile<i32>

        // ════════════════════════════════════════════════════════════════════
        // K-REDUCTION LOOP
        //   Each iteration: load A + load B + 1 × mmaf(128×128)
        //   ILP surface: only the two loads can overlap with each other.
        // ════════════════════════════════════════════════════════════════════
        %result = for %k in (%i0 to %mk_len#1, step %i1) : tile<i32>
            iter_values(%acc = %cst) -> (tile<128x128xf32>)
        {
            // Load A tile (128×64) for this K-step
            %A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] :
                partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>,
                tile<i32> -> tile<128x64xf16>, token

            // Load B tile (64×128) for this K-step
            // (no token=%t1 here, so A and B loads can proceed in parallel)
            %B_frag, %t2 = load_view_tko weak %B_block[%k, %bidy] :
                partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>,
                tile<i32> -> tile<64x128xf16>, token

            // Single MMA: (128×64) × (64×128) → accumulate into 128×128
            %acc_new = mmaf %A_frag, %B_frag, %acc :
                tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>

            continue %acc_new : tile<128x128xf32>
        }

        // ── Store result tile to C ────────────────────────────────────────────
        %t_store = store_view_tko weak %result, %C_block[%bidx, %bidy] :
            tile<128x128xf32>,
            partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>,
            tile<i32> -> token
    }
}
