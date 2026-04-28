"""Probe whether ct.Constant supports compile-time array indexing."""
import cupy as cp
import cuda.tile as ct

# Test data: small list of (pk, pm, pn) tuples — simulates unit_points
UNIT_POINTS = [(0,0,0), (0,1,0), (0,1,1), (1,0,1)]
f = len(UNIT_POINTS)
tm_dim = 8
tn_dim = 8
sk_dim = 2

# Attempt 1: Constant list parameter with static_eval indexing
try:
    @ct.kernel
    def probe_constant_list(
        out_sk,
        out_m,
        out_n,
        points: ct.Constant,       # pass unit_points as Constant
        f_val:  ct.Constant[int],
    ):
        bid   = ct.bid(0)
        local = bid % f_val
        # Try static_eval to index at compile time
        pk, pm, pn = ct.static_eval(points[local])
        ct.store(out_sk, index=(bid,), tile=pk)
        ct.store(out_m,  index=(bid,), tile=pm)
        ct.store(out_n,  index=(bid,), tile=pn)

    out_sk = cp.zeros(f, dtype=cp.int32)
    out_m  = cp.zeros(f, dtype=cp.int32)
    out_n  = cp.zeros(f, dtype=cp.int32)
    ct.launch(
        cp.cuda.get_current_stream(),
        (f, 1, 1),
        probe_constant_list,
        (out_sk, out_m, out_n, UNIT_POINTS, f),
    )
    cp.cuda.get_current_stream().synchronize()
    print('Attempt 1 (Constant list + static_eval): SUCCESS')
    print('  out_sk:', cp.asnumpy(out_sk).tolist())
    print('  out_m: ', cp.asnumpy(out_m).tolist())
    print('  out_n: ', cp.asnumpy(out_n).tolist())
    expected = [(p[0], p[1], p[2]) for p in UNIT_POINTS]
    print('  expected:', expected)
except Exception as e:
    print(f'Attempt 1 FAILED: {type(e).__name__}: {e}')


# Attempt 2: flat Constant arrays (separate sk/m/n lists)
try:
    pts_sk = [p[0] for p in UNIT_POINTS]
    pts_m  = [p[1] for p in UNIT_POINTS]
    pts_n  = [p[2] for p in UNIT_POINTS]

    @ct.kernel
    def probe_flat_constants(
        out_sk,
        out_m,
        out_n,
        cst_sk: ct.Constant,
        cst_m:  ct.Constant,
        cst_n:  ct.Constant,
        f_val:  ct.Constant[int],
    ):
        bid   = ct.bid(0)
        local = bid % f_val
        pk = ct.static_eval(cst_sk[local])
        pm = ct.static_eval(cst_m[local])
        pn = ct.static_eval(cst_n[local])
        ct.store(out_sk, index=(bid,), tile=pk)
        ct.store(out_m,  index=(bid,), tile=pm)
        ct.store(out_n,  index=(bid,), tile=pn)

    out_sk = cp.zeros(f, dtype=cp.int32)
    out_m  = cp.zeros(f, dtype=cp.int32)
    out_n  = cp.zeros(f, dtype=cp.int32)
    ct.launch(
        cp.cuda.get_current_stream(),
        (f, 1, 1),
        probe_flat_constants,
        (out_sk, out_m, out_n, pts_sk, pts_m, pts_n, f),
    )
    cp.cuda.get_current_stream().synchronize()
    print('Attempt 2 (flat Constant arrays + static_eval): SUCCESS')
    print('  out_sk:', cp.asnumpy(out_sk).tolist())
    print('  out_m: ', cp.asnumpy(out_m).tolist())
    print('  out_n: ', cp.asnumpy(out_n).tolist())
except Exception as e:
    print(f'Attempt 2 FAILED: {type(e).__name__}: {e}')


# Attempt 3: closure-captured tuple, static_eval on dynamic index
try:
    pts_tuple = tuple(UNIT_POINTS)

    @ct.kernel
    def probe_closure_tuple(
        out_sk,
        out_m,
        out_n,
        f_val: ct.Constant[int],
    ):
        bid   = ct.bid(0)
        local = bid % f_val
        # pts_tuple is captured from Python closure
        pk, pm, pn = ct.static_eval(pts_tuple[local])
        ct.store(out_sk, index=(bid,), tile=pk)
        ct.store(out_m,  index=(bid,), tile=pm)
        ct.store(out_n,  index=(bid,), tile=pn)

    out_sk = cp.zeros(f, dtype=cp.int32)
    out_m  = cp.zeros(f, dtype=cp.int32)
    out_n  = cp.zeros(f, dtype=cp.int32)
    ct.launch(
        cp.cuda.get_current_stream(),
        (f, 1, 1),
        probe_closure_tuple,
        (out_sk, out_m, out_n, f),
    )
    cp.cuda.get_current_stream().synchronize()
    print('Attempt 3 (closure tuple + static_eval): SUCCESS')
    print('  out_sk:', cp.asnumpy(out_sk).tolist())
except Exception as e:
    print(f'Attempt 3 FAILED: {type(e).__name__}: {e}')
