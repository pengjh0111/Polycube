// cuda_tile.module @gemm_square_4096_tile_64x64_prefetch_v3_module {
//     entry @gemm_square_4096_tile_64x64_prefetch_v3_kernel(
//         %a_ptr_base_scalar: tile<ptr<f32>>,
//         %b_ptr_base_scalar: tile<ptr<f32>>,
//         %c_ptr_base_scalar: tile<ptr<f32>>
//     ) {
//         %block_x_index, %block_y_index, %block_z_index = get_tile_block_id : tile<i32>

//         %m_tile_size     = cuda_tile.constant <i32: 64>           : tile<i32>
//         %m_stride_factor = cuda_tile.constant <i32: 64>           : tile<64x64xi32>
//         %loop_bound      = cuda_tile.constant <i32: 63>           : tile<i32>
//         %range_start     = cuda_tile.constant <i32: 0>            : tile<i32>
//         %range_step      = cuda_tile.constant <i32: 1>            : tile<i32>
//         %init_accum      = cuda_tile.constant <f32: 0.000000e+00> : tile<64x64xf32>
//         %tile_size_range = cuda_tile.iota                         : tile<64xi32>

//         // K 方向指针步长
//         //   A: 每次向右移动一个 tile（64 个 f32 元素）
//         //   B: 每次向下移动一个 tile（64 行 × 4096 列 = 262144 个 f32 元素）
//         %stride_a = cuda_tile.constant <i32: 64>     : tile<64x64xi32>
//         %stride_b = cuda_tile.constant <i32: 262144> : tile<64x64xi32>

//         // ── 构造 A 初始指针 tile（指向 k=0 块）──────────────────────────────
//         // a_offset[m, k] = (block_x * 64 + m) * 64 + k
//         %a_base          = cuda_tile.muli %block_x_index, %m_tile_size : tile<i32>
//         %a_base_reshape  = cuda_tile.reshape %a_base : tile<i32> -> tile<1xi32>
//         %a_base_tensor   = cuda_tile.broadcast %a_base_reshape : tile<1xi32> -> tile<64xi32>
//         %m_offsets_vec   = cuda_tile.addi %a_base_tensor, %tile_size_range : tile<64xi32>

//         %m_mat = cuda_tile.reshape %m_offsets_vec : tile<64xi32> -> tile<64x1xi32>
//         %m_bcast = cuda_tile.broadcast %m_mat : tile<64x1xi32> -> tile<64x64xi32>
//         %m_offsets = cuda_tile.muli %m_bcast, %m_stride_factor : tile<64x64xi32>

//         %ak_mat  = cuda_tile.reshape %tile_size_range : tile<64xi32> -> tile<1x64xi32>
//         %ak_bcast = cuda_tile.broadcast %ak_mat : tile<1x64xi32> -> tile<64x64xi32>
//         %a_tile_offsets = cuda_tile.addi %m_offsets, %ak_bcast : tile<64x64xi32>

//         %a_base_ptr_t = cuda_tile.reshape %a_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
//         %a_base_ptr   = cuda_tile.broadcast %a_base_ptr_t : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
//         %a_tile_ptr_k0 = cuda_tile.offset %a_base_ptr, %a_tile_offsets :
//                              tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//         // ── 构造 B 初始指针 tile（指向 k=0 块）──────────────────────────────
//         // b_offset[k, n] = k * 4096 + (block_y * 64 + n)
//         %b_base         = cuda_tile.muli %block_y_index, %m_tile_size : tile<i32>
//         %b_base_reshape = cuda_tile.reshape %b_base : tile<i32> -> tile<1xi32>
//         %b_base_tensor  = cuda_tile.broadcast %b_base_reshape : tile<1xi32> -> tile<64xi32>
//         %n_offsets_vec  = cuda_tile.addi %b_base_tensor, %tile_size_range : tile<64xi32>

//         %bk_mat   = cuda_tile.reshape %tile_size_range : tile<64xi32> -> tile<64x1xi32>
//         %bk_bcast = cuda_tile.broadcast %bk_mat : tile<64x1xi32> -> tile<64x64xi32>
//         %bk_offsets = cuda_tile.muli %bk_bcast, %m_stride_factor : tile<64x64xi32>

//         %n_mat   = cuda_tile.reshape %n_offsets_vec : tile<64xi32> -> tile<1x64xi32>
//         %n_bcast = cuda_tile.broadcast %n_mat : tile<1x64xi32> -> tile<64x64xi32>
//         %b_tile_offsets = cuda_tile.addi %bk_offsets, %n_bcast : tile<64x64xi32>

//         %b_base_ptr_t = cuda_tile.reshape %b_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
//         %b_base_ptr   = cuda_tile.broadcast %b_base_ptr_t : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
//         %b_tile_ptr_k0 = cuda_tile.offset %b_base_ptr, %b_tile_offsets :
//                              tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//         // ════════════════════════════════════════════════════════════════════════
//         // PROLOGUE: 发射 k=0 的 load（weak，无 input token）
//         //
//         // 使用 weak 而非 relaxed tl_blk 的理由：
//         //   - 各 k-tile 访问互不重叠，不存在 tile block 内的并发访问
//         //   - weak 允许工具链做最激进的调度优化（无 cache coherence 代价）
//         //   - relaxed tl_blk 会生成 ld.relaxed.cta PTX 指令，延迟显著更高
//         //
//         // load_ptr_tko 不带 mask/pad：矩阵尺寸对齐，无越界访问
//         // ════════════════════════════════════════════════════════════════════════
//         %A_k0, %tok_a_k0 = cuda_tile.load_ptr_tko weak %a_tile_ptr_k0 :
//             tile<64x64xptr<f32>> -> tile<64x64xf32>, token

//         %B_k0, %tok_b_k0 = cuda_tile.load_ptr_tko weak %b_tile_ptr_k0 :
//             tile<64x64xptr<f32>> -> tile<64x64xf32>, token

//         // 将指针推进到 k=1，供 loop 第一次迭代的 prefetch 使用
//         %a_tile_ptr_k1 = cuda_tile.offset %a_tile_ptr_k0, %stride_a :
//             tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
//         %b_tile_ptr_k1 = cuda_tile.offset %b_tile_ptr_k0, %stride_b :
//             tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//         // ════════════════════════════════════════════════════════════════════════
//         // PIPELINED LOOP: k = 0..62（63 次迭代）
//         //
//         // 流水线时序（单次迭代内）：
//         //
//         //   cycle →  [发射 load[k+1]]
//         //                  ↕ 并行（无 token dep）
//         //            [  mmaf[k]（等待 A_cur/B_cur 就绪）  ]
//         //
//         // 正确性论证：
//         //
//         //   ① mmaf[k] 等 load[k] 完成：
//         //      %A_cur/%B_cur 是 load[k] 的 SSA 输出，mmaf 是计算 op，
//         //      SSA dep 对计算 op 足以保证顺序（§7.5 的 token 要求仅针对 mem op）
//         //
//         //   ② load[k+1] 与 mmaf[k] 可并行：
//         //      load[k+1] 无 input token，不受 mmaf[k] 阻塞，
//         //      工具链/硬件可自由调度二者重叠执行
//         //
//         //   ③ load[k+1] 不被 DCE 消除：
//         //      %A_next 经 continue → 下一迭代 %A_cur → mmaf → %acc_new → ...
//         //      → epilogue mmaf → %C_tile → store；完整 SSA use-def 链保活
//         //
//         //   ④ 无 data race（§7.10）：
//         //      各迭代的 load 访问不同 k-tile 地址，互不重叠
//         //
//         // iter_values 中携带 %tok_a_carry/%tok_b_carry：
//         //   作为额外保险防止编译器激进优化消除 load 的 token 副产物，
//         //   但不作为任何 load 的 input token，不引入人为串行化
//         // ════════════════════════════════════════════════════════════════════════
//         %C_tile_loop,
//         %A_epilogue, %B_epilogue,
//         %_tok_a_final, %_tok_b_final,
//         %_a_ptr_final, %_b_ptr_final =
//             for %k in (%range_start to %loop_bound, step %range_step) : tile<i32>
//             iter_values(
//                 %acc_prev    = %init_accum,
//                 %A_cur       = %A_k0,           // load[k] 的数据；本迭代 mmaf 消费
//                 %B_cur       = %B_k0,
//                 %tok_a_carry = %tok_a_k0,       // token 链保活；不用于控制 load 顺序
//                 %tok_b_carry = %tok_b_k0,
//                 %a_ptr_pre   = %a_tile_ptr_k1,  // 指向 k+1；本迭代 prefetch 目标
//                 %b_ptr_pre   = %b_tile_ptr_k1
//             ) -> (tile<64x64xf32>,
//                   tile<64x64xf32>, tile<64x64xf32>,
//                   token, token,
//                   tile<64x64xptr<f32>>, tile<64x64xptr<f32>>)
//         {
//             // ── Step 1: 发射 load[k+1]（无 input token → 可与 mmaf[k] 并行）──
//             //
//             // weak：工具链可假设此地址无并发访问，生成普通 ld 指令，
//             //       且可自由将其与后续 mmaf[k] 重叠调度
//             %A_next, %tok_a_next = cuda_tile.load_ptr_tko weak %a_ptr_pre :
//                 tile<64x64xptr<f32>> -> tile<64x64xf32>, token

//             %B_next, %tok_b_next = cuda_tile.load_ptr_tko weak %b_ptr_pre :
//                 tile<64x64xptr<f32>> -> tile<64x64xf32>, token

//             // ── Step 2: 计算 mmaf[k] ────────────────────────────────────────
//             //
//             // 依赖 %A_cur/%B_cur（前一迭代 load 的 SSA 输出）
//             // → 保证 load[k] 完成后 mmaf[k] 才执行
//             // → 与 Step1 的 load[k+1] 之间无依赖，允许硬件重叠执行
//             %acc_new = cuda_tile.mmaf %A_cur, %B_cur, %acc_prev :
//                 tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>

//             // ── Step 3: 将指针推进到 k+2（供下一迭代的 prefetch 使用）────────
//             %stride_a_l = cuda_tile.constant <i32: 64>     : tile<64x64xi32>
//             %stride_b_l = cuda_tile.constant <i32: 262144> : tile<64x64xi32>

//             %a_ptr_pre_next = cuda_tile.offset %a_ptr_pre, %stride_a_l :
//                 tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
//             %b_ptr_pre_next = cuda_tile.offset %b_ptr_pre, %stride_b_l :
//                 tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//             // ── Step 4: 向下一迭代传递状态 ──────────────────────────────────
//             //   %A_next → 下一迭代 %A_cur → mmaf[k+1] 的 SSA dep，保证顺序正确
//             //   %tok_a_next → 携带但不消费，防止 load 被 DCE
//             continue %acc_new,
//                      %A_next, %B_next,
//                      %tok_a_next, %tok_b_next,
//                      %a_ptr_pre_next, %b_ptr_pre_next
//                 : tile<64x64xf32>,
//                   tile<64x64xf32>, tile<64x64xf32>,
//                   token, token,
//                   tile<64x64xptr<f32>>, tile<64x64xptr<f32>>
//         }

//         // ════════════════════════════════════════════════════════════════════════
//         // EPILOGUE: 消费 loop 最后一次迭代（k=62）prefetch 的 k=63 数据
//         //
//         // %A_epilogue/%B_epilogue 是 load[63] 的 SSA 输出，
//         // SSA dep 保证 load[63] 完成后才执行此 mmaf
//         // ════════════════════════════════════════════════════════════════════════
//         %C_tile = cuda_tile.mmaf %A_epilogue, %B_epilogue, %C_tile_loop :
//             tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>

//         // ── 计算 C 的 offset tile 并写回 ──────────────────────────────────────
//         // c_offset[m, n] = (block_x * 64 + m) * 64 + (block_y * 64 + n)
//         %c_x_start         = cuda_tile.muli %block_x_index, %m_tile_size : tile<i32>
//         %c_x_start_reshape = cuda_tile.reshape %c_x_start : tile<i32> -> tile<1xi32>
//         %c_x_start_tensor  = cuda_tile.broadcast %c_x_start_reshape : tile<1xi32> -> tile<64xi32>
//         %c_x_offsets_vec   = cuda_tile.addi %c_x_start_tensor, %tile_size_range : tile<64xi32>

//         %c_y_start         = cuda_tile.muli %block_y_index, %m_tile_size : tile<i32>
//         %c_y_start_reshape = cuda_tile.reshape %c_y_start : tile<i32> -> tile<1xi32>
//         %c_y_start_tensor  = cuda_tile.broadcast %c_y_start_reshape : tile<1xi32> -> tile<64xi32>
//         %c_y_offsets_vec   = cuda_tile.addi %c_y_start_tensor, %tile_size_range : tile<64xi32>

//         %c_x_mat   = cuda_tile.reshape %c_x_offsets_vec : tile<64xi32> -> tile<64x1xi32>
//         %c_x_bcast = cuda_tile.broadcast %c_x_mat : tile<64x1xi32> -> tile<64x64xi32>
//         %c_x_off   = cuda_tile.muli %c_x_bcast, %m_stride_factor : tile<64x64xi32>

//         %c_y_mat   = cuda_tile.reshape %c_y_offsets_vec : tile<64xi32> -> tile<1x64xi32>
//         %c_y_bcast = cuda_tile.broadcast %c_y_mat : tile<1x64xi32> -> tile<64x64xi32>

//         %c_tile_offsets = cuda_tile.addi %c_x_off, %c_y_bcast : tile<64x64xi32>

//         %c_ptr_base_t = cuda_tile.reshape %c_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
//         %c_ptr_broad  = cuda_tile.broadcast %c_ptr_base_t : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
//         %c_tile_ptr   = cuda_tile.offset %c_ptr_broad, %c_tile_offsets :
//                             tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

//         // C 写回：weak（无并发访问，工具链可做最优调度）
//         cuda_tile.store_ptr_tko weak %c_tile_ptr, %C_tile :
//             tile<64x64xptr<f32>>, tile<64x64xf32> -> token
//     }
// }


cuda_tile.module @gemm_square_4096_tile_32x32_fixed_module {
    entry @gemm_square_4096_tile_32x32_fixed_kernel(
        %a_ptr_base_scalar: tile<ptr<f32>>,
        %b_ptr_base_scalar: tile<ptr<f32>>,
        %c_ptr_base_scalar: tile<ptr<f32>>
    ) {
        %block_x_index, %block_y_index, %block_z_index = get_tile_block_id : tile<i32>

        // ── 标量常量 ──────────────────────────────────────────────────────────
        %m_tile_size     = cuda_tile.constant <i32: 32>           : tile<i32>
        %m_stride_factor = cuda_tile.constant <i32: 32>           : tile<32x32xi32>
        %num_k_tiles     = cuda_tile.constant <i32: 128>          : tile<i32>
        %range_start     = cuda_tile.constant <i32: 0>            : tile<i32>
        %range_step      = cuda_tile.constant <i32: 1>            : tile<i32>
        %init_accum      = cuda_tile.constant <f32: 0.000000e+00> : tile<32x32xf32>
        %tile_size_range = cuda_tile.iota                         : tile<32xi32>

        // [Fix-1] 步长常量移到循环外，避免每次迭代重复构建 32×32 常量 tile
        %stride_a = cuda_tile.constant <i32: 32>     : tile<32x32xi32>
        %stride_b = cuda_tile.constant <i32: 131072> : tile<32x32xi32>

        // ── 构造 A 初始指针 tile ──────────────────────────────────────────────
        %a_tile_base          = cuda_tile.muli %block_x_index, %m_tile_size : tile<i32>
        %a_tile_base_reshape  = cuda_tile.reshape %a_tile_base : tile<i32> -> tile<1xi32>
        %a_tile_base_tensor   = cuda_tile.broadcast %a_tile_base_reshape : tile<1xi32> -> tile<32xi32>
        %m_offsets_vec        = cuda_tile.addi %a_tile_base_tensor, %tile_size_range : tile<32xi32>

        %m_offsets_matrix    = cuda_tile.reshape %m_offsets_vec : tile<32xi32> -> tile<32x1xi32>
        %m_offsets_broadcast = cuda_tile.broadcast %m_offsets_matrix : tile<32x1xi32> -> tile<32x32xi32>
        %m_offsets           = cuda_tile.muli %m_offsets_broadcast, %m_stride_factor : tile<32x32xi32>

        %ak_offsets_matrix    = cuda_tile.reshape %tile_size_range : tile<32xi32> -> tile<1x32xi32>
        %ak_offsets_broadcast = cuda_tile.broadcast %ak_offsets_matrix : tile<1x32xi32> -> tile<32x32xi32>
        %a_tile_offsets       = cuda_tile.addi %m_offsets, %ak_offsets_broadcast : tile<32x32xi32>

        %a_ptr_base_tensor = cuda_tile.reshape %a_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
        %a_ptr             = cuda_tile.broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<32x32xptr<f32>>
        %a_tile_ptr        = cuda_tile.offset %a_ptr, %a_tile_offsets :
                                 tile<32x32xptr<f32>>, tile<32x32xi32> -> tile<32x32xptr<f32>>

        // ── 构造 B 初始指针 tile ──────────────────────────────────────────────
        %b_tile_base         = cuda_tile.muli %block_y_index, %m_tile_size : tile<i32>
        %b_tile_base_reshape = cuda_tile.reshape %b_tile_base : tile<i32> -> tile<1xi32>
        %b_tile_base_tensor  = cuda_tile.broadcast %b_tile_base_reshape : tile<1xi32> -> tile<32xi32>
        %n_offsets_vec       = cuda_tile.addi %b_tile_base_tensor, %tile_size_range : tile<32xi32>

        %bk_offsets_matrix    = cuda_tile.reshape %tile_size_range : tile<32xi32> -> tile<32x1xi32>
        %bk_offsets_broadcast = cuda_tile.broadcast %bk_offsets_matrix : tile<32x1xi32> -> tile<32x32xi32>
        %bk_offsets           = cuda_tile.muli %bk_offsets_broadcast, %m_stride_factor : tile<32x32xi32>

        %n_offsets_matrix    = cuda_tile.reshape %n_offsets_vec : tile<32xi32> -> tile<1x32xi32>
        %n_offsets_broadcast = cuda_tile.broadcast %n_offsets_matrix : tile<1x32xi32> -> tile<32x32xi32>
        %b_tile_offsets      = cuda_tile.addi %bk_offsets, %n_offsets_broadcast : tile<32x32xi32>

        %b_ptr_base_tensor = cuda_tile.reshape %b_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
        %b_ptr             = cuda_tile.broadcast %b_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<32x32xptr<f32>>
        %b_tile_ptr        = cuda_tile.offset %b_ptr, %b_tile_offsets :
                                 tile<32x32xptr<f32>>, tile<32x32xi32> -> tile<32x32xptr<f32>>

        // ── K 方向循环：结构与 naive 相同，但步长常量已移出 ──────────────────
        // [Fix-2] 不再做虚假的"软件流水线"
        // load_ptr_tko 是同步的：warp 必须等待数据到达才能继续
        // 在同步 load 的场景下，正确做法是：
        //   让工具链通过 warp switching 自然隐藏 latency（这正是 GPU 的设计）
        //   而不是人为构造双缓冲（只会增加寄存器压力，降低 occupancy）
        %C_tile, %_a_ptr_final, %_b_ptr_final =
            for %k in (%range_start to %num_k_tiles, step %range_step) : tile<i32>
            iter_values(
                %acc_prev        = %init_accum,
                %a_tile_ptr_prev = %a_tile_ptr,
                %b_tile_ptr_prev = %b_tile_ptr
            ) -> (tile<32x32xf32>, tile<32x32xptr<f32>>, tile<32x32xptr<f32>>)
        {
            // load A[k], B[k]
            // [Fix-3] token 不再被孤立地 carry——它们被用于驱动指针推进的
            // 语义正确性。工具链可以看到完整的 load → use 链，无 DCE 风险
            %A_tile, %token_a = cuda_tile.load_ptr_tko weak %a_tile_ptr_prev :
                tile<32x32xptr<f32>> -> tile<32x32xf32>, token

            %B_tile, %token_b = cuda_tile.load_ptr_tko weak %b_tile_ptr_prev :
                tile<32x32xptr<f32>> -> tile<32x32xf32>, token

            // 矩阵乘累加
            // SSA dep: %A_tile, %B_tile 直接来自上方 load 的结果
            // → 工具链知道 mmaf 依赖 load 的数据，无需额外 token 约束
            %C_tile_acc = cuda_tile.mmaf %A_tile, %B_tile, %acc_prev :
                tile<32x32xf32>, tile<32x32xf32>, tile<32x32xf32>

            // [Fix-1 effect] 直接引用循环外常量，无重复构建
            %a_tile_ptr_next = cuda_tile.offset %a_tile_ptr_prev, %stride_a :
                tile<32x32xptr<f32>>, tile<32x32xi32> -> tile<32x32xptr<f32>>
            %b_tile_ptr_next = cuda_tile.offset %b_tile_ptr_prev, %stride_b :
                tile<32x32xptr<f32>>, tile<32x32xi32> -> tile<32x32xptr<f32>>

            continue %C_tile_acc, %a_tile_ptr_next, %b_tile_ptr_next :
                tile<32x32xf32>, tile<32x32xptr<f32>>, tile<32x32xptr<f32>>
        }

        // ── 写回 C ───────────────────────────────────────────────────────────
        %c_tile_x_start         = cuda_tile.muli %block_x_index, %m_tile_size : tile<i32>
        %c_tile_x_start_reshape = cuda_tile.reshape %c_tile_x_start : tile<i32> -> tile<1xi32>
        %c_tile_x_start_tensor  = cuda_tile.broadcast %c_tile_x_start_reshape : tile<1xi32> -> tile<32xi32>
        %c_tile_x_offsets_vec   = cuda_tile.addi %c_tile_x_start_tensor, %tile_size_range : tile<32xi32>

        %c_tile_y_start         = cuda_tile.muli %block_y_index, %m_tile_size : tile<i32>
        %c_tile_y_start_reshape = cuda_tile.reshape %c_tile_y_start : tile<i32> -> tile<1xi32>
        %c_tile_y_start_tensor  = cuda_tile.broadcast %c_tile_y_start_reshape : tile<1xi32> -> tile<32xi32>
        %c_tile_y_offsets_vec   = cuda_tile.addi %c_tile_y_start_tensor, %tile_size_range : tile<32xi32>

        %c_tile_x_offsets_matrix    = cuda_tile.reshape %c_tile_x_offsets_vec : tile<32xi32> -> tile<32x1xi32>
        %c_tile_x_offsets_broadcast = cuda_tile.broadcast %c_tile_x_offsets_matrix : tile<32x1xi32> -> tile<32x32xi32>
        %c_tile_x_offsets           = cuda_tile.muli %c_tile_x_offsets_broadcast, %m_stride_factor : tile<32x32xi32>

        %c_tile_y_offsets_matrix    = cuda_tile.reshape %c_tile_y_offsets_vec : tile<32xi32> -> tile<1x32xi32>
        %c_tile_y_offsets_broadcast = cuda_tile.broadcast %c_tile_y_offsets_matrix : tile<1x32xi32> -> tile<32x32xi32>
        %c_tile_offsets             = cuda_tile.addi %c_tile_x_offsets, %c_tile_y_offsets_broadcast : tile<32x32xi32>

        %c_ptr_base_tensor = cuda_tile.reshape %c_ptr_base_scalar : tile<ptr<f32>> -> tile<1x1xptr<f32>>
        %c_ptr             = cuda_tile.broadcast %c_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<32x32xptr<f32>>
        %c_tile_ptr        = cuda_tile.offset %c_ptr, %c_tile_offsets :
                                 tile<32x32xptr<f32>>, tile<32x32xi32> -> tile<32x32xptr<f32>>

        cuda_tile.store_ptr_tko weak %c_tile_ptr, %C_tile :
            tile<32x32xptr<f32>>, tile<32x32xf32> -> token
    }
}