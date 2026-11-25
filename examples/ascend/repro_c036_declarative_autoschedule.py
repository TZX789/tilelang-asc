"""Declarative TileLang (T.MixedKernel + T.Pipelined + T.Stage + T.annotate_buffer_versions)
version of FlashMask Prefill Backward (C036 BF16/D128 algorithm).

Purpose:
  Standalone upstream compiler bug reproducer demonstrating how cleanly the
  delivered C036 algorithm (adaptive G_sec L2 sections + WorkPrefix load balance
  + flat-head single 1D pipeline + 4-slot L1 Q/dO + dual L0C dQ + independent
  QK/dP L0C + fused 256B SimdVF P/dS) can be expressed declaratively (~230 lines
  vs ~1,163 lines of manual synchronization), and profiling the exact TileLang
  Ascend auto-schedule / InsertSync lowering bottleneck or failure.

NOTE: This file does NOT replace or modify cases/flashmask_prefill_bwd/.
"""
import sys
import time
from typing import Any

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tvm.ir.instrument import pass_instrument

_NEG = -1.0e30


def _tile_dtype(dtype: str) -> str:
    if dtype in ("float16", "bfloat16", "float32"):
        return dtype
    raise ValueError(f"Unsupported kernel dtype: {dtype}")


def _make_block_keep(seq_q, seq_k, block_m: int, block_n: int):
    """In-kernel whole-block-keep test for non-causal R=2 intervals."""
    def block_keep(ext, m0, j1):
        q_min = T.cast(m0, "float32")
        q_max = T.cast(T.min(m0 + block_m, seq_q) - 1, "float32")
        tail_ok = T.if_then_else((j1 + 1) * block_n <= seq_k, T.float32(1), T.float32(0))
        umin_ok = T.if_then_else(ext[1, 0] - 1.0 >= q_max, T.float32(1), T.float32(0))
        lmax_ok = T.if_then_else(ext[0, 1] <= q_min, T.float32(1), T.float32(0))
        return umin_ok * lmax_ok * tail_ok

    return block_keep


def make_declarative_c036_bwd_prim_func(
    batch: int = 1,
    heads_q: int = 32,
    heads_kv: int = 4,
    head_dim: int = 128,
    mask_heads: int = 4,
    mask_dim: int = 2,
    dtype_name: str = "bfloat16",
    softmax_scale: float = 0.08838834764831845,
    block_q: int = 64,
    block_k: int = 128,
    core_count: int = 32,
    l2_budget_bytes: int = 128 << 20,
    version_mode: str = "auto",  # "auto" (like C008) or "counter" (like C009)
    enable_sections: bool = True,
) -> Any:
    dtype = _tile_dtype(dtype_name)
    element_bytes = 2
    groups = heads_q // heads_kv
    scale = float(softmax_scale)
    seq_q = T.dynamic("seq_q")
    seq_k = T.dynamic("seq_k")
    num_kv_blocks = seq_k // block_k
    num_tasks = batch * heads_kv * num_kv_blocks
    q_shape = [batch, heads_q, seq_q, head_dim]
    kv_shape = [batch, heads_kv, seq_k, head_dim]
    bounds_shape = [batch, mask_heads, mask_dim, seq_k]
    ext_shape = [batch, mask_heads, num_kv_blocks, 2, mask_dim]
    lse_shape = [batch, heads_q, seq_q, 1]
    delta_shape = [batch, heads_q, seq_q]
    vec_q = block_q // 2
    block_keep = _make_block_keep(seq_q, seq_k, block_q, block_k)

    def _ver(n: int):
        if version_mode == "counter" and n > 1:
            return (n, "counter")
        return n

    @T.prim_func
    def flashmask_prefill_bwd_declarative(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(q_shape, dtype),
        LSE: T.Tensor(lse_shape, "float32"),
        Delta: T.Tensor(delta_shape, "float32"),
        Bounds: T.Tensor(bounds_shape, "int32"),
        QRange: T.Tensor([batch, mask_heads, num_kv_blocks, 2], "int32"),
        Extrema: T.Tensor(ext_shape, "float32"),
        dQ: T.Tensor(q_shape, "float32"),
        dK: T.Tensor(kv_shape, "float32"),
        dV: T.Tensor(kv_shape, "float32"),
        Prefix: T.Tensor([num_tasks + 1 + batch * heads_kv * (num_kv_blocks + 1)], "int64"),
    ):
        with T.MixedKernel(core_count, sids=2) as (cid, _sid):
            # L1 buffers
            k_l1 = T.alloc_l1((block_k, head_dim), dtype)
            v_l1 = T.alloc_l1((block_k, head_dim), dtype)
            kt_l1 = T.alloc_l1((head_dim, block_k), dtype)
            q_l1 = T.alloc_l1((block_q, head_dim), dtype)
            do_l1 = T.alloc_l1((block_q, head_dim), dtype)
            p_l1 = T.alloc_l1((block_q, block_k), dtype)
            ds_l1 = T.alloc_l1((block_q, block_k), dtype)

            # L0A / L0B operand scratch
            qk_a = T.alloc_l0a((block_q, head_dim), dtype)
            qk_b = T.alloc_l0b((block_k, head_dim), dtype)
            dk_a = T.alloc_l0a((block_k, block_q), dtype)
            dk_b = T.alloc_l0b((head_dim, block_q), dtype)
            dq_a = T.alloc_l0a((block_q, block_q), dtype)

            # L0C accumulators (C036: independent qk_c / dp_c, 2-slot dq_c)
            qk_c = T.alloc_l0c((block_q, block_k), "float32")
            dp_c = T.alloc_l0c((block_q, block_k), "float32")
            dk_c = T.alloc_l0c((block_k, head_dim), "float32")
            dv_c = T.alloc_l0c((block_k, head_dim), "float32")
            dq_c = T.alloc_l0c((block_q, head_dim), "float32")

            # UB buffers (per AIV)
            qk_ub = T.alloc_shared((vec_q, block_k), "float32")
            dp_ub = T.alloc_shared((vec_q, block_k), "float32")
            p_ub = T.alloc_shared((vec_q, block_k), dtype)
            ds_ub = T.alloc_shared((vec_q, block_k), dtype)
            lse_ub = T.alloc_shared((vec_q, 1), "float32")
            delta_ub = T.alloc_shared((vec_q,), "float32")
            ext_ub = T.alloc_shared((2, mask_dim), "float32")
            b0 = T.alloc_shared((block_k,), "int32")
            b1 = T.alloc_shared((block_k,), "int32")

            # Declarative multi-buffer versions matching C036 capacities:
            # q_l1/do_l1=4 (l1-ring4), p_l1/ds_l1=2, dq_c=2 (256 KiB L0C), lse/delta=2
            T.annotate_buffer_versions({
                k_l1: 1,
                v_l1: 1,
                kt_l1: 1,
                q_l1: _ver(4),
                do_l1: _ver(4),
                p_l1: _ver(2),
                ds_l1: _ver(2),
                qk_a: _ver(2),
                qk_b: _ver(2),
                dk_a: _ver(2),
                dk_b: _ver(2),
                dq_a: _ver(2),
                qk_c: 1,
                dp_c: 1,
                dk_c: 1,
                dv_c: 1,
                dq_c: _ver(2),
                qk_ub: _ver(2),
                dp_ub: _ver(2),
                p_ub: _ver(2),
                ds_ub: _ver(2),
                lse_ub: _ver(2),
                delta_ub: _ver(2),
                b0: 1,
                b1: 1,
                ext_ub: 1,
            })

            # Adaptive L2 cache reuse sections (G_sec in {8, 4, 2, 1})
            if enable_sections:
                footprint = (
                    T.cast(seq_q, "int64") * batch * heads_q * (head_dim * (2 * element_bytes + 4) + 8)
                    + T.cast(seq_k, "int64") * batch * heads_kv * head_dim * (2 * element_bytes + 8)
                )
                use_sections = footprint > l2_budget_bytes
                q_head_bytes = T.cast(seq_q, "int64") * (head_dim * (2 * element_bytes + 4) + 8)
                q_target = T.cast(l2_budget_bytes, "int64") * 5 // 8
                g_sec = T.alloc_var("int32", init=1)
                for cand in T.unroll(1, groups + 1):
                    if groups % cand == 0:
                        if T.cast(cand, "int64") * q_head_bytes <= q_target:
                            g_sec = cand
                sec_per_b = heads_q // g_sec
                num_sections = T.if_then_else(use_sections, batch * sec_per_b, 1)
                scale_p = T.if_then_else(use_sections, T.cast(g_sec, "int64"), T.int64(1))
            else:
                use_sections = False
                sec_per_b = 1
                g_sec = 1
                num_sections = 1
                scale_p = T.int64(1)

            for section in range(num_sections):
                bz_sec = section // sec_per_b
                sec_in_b = section % sec_per_b
                kvh_sec = (sec_in_b * g_sec) // groups
                head_base_sec = sec_in_b * g_sec
                prefix_offset = T.if_then_else(
                    use_sections,
                    num_tasks + 1 + (bz_sec * heads_kv + kvh_sec) * (num_kv_blocks + 1),
                    0,
                )
                section_tasks = T.if_then_else(use_sections, num_kv_blocks, num_tasks)
                total = Prefix[prefix_offset + section_tasks] * scale_p
                rank = T.if_then_else(
                    use_sections,
                    (cid + core_count - section % core_count) % core_count,
                    cid,
                )
                work_begin = total * rank // core_count
                work_end = total * (rank + 1) // core_count

                # [ABLATION A] while binary-search + dynamic base bounds removed.
                for base in range(section_tasks):
                    task_begin = Prefix[prefix_offset + base] * scale_p
                    task_end = Prefix[prefix_offset + base + 1] * scale_p
                    local_begin = T.max(work_begin, task_begin) - task_begin
                    local_end = T.min(work_end, task_end) - task_begin
                    if local_end > local_begin:
                        kv_head = T.if_then_else(use_sections, kvh_sec, (base // num_kv_blocks) % heads_kv)
                        by_kv = T.if_then_else(use_sections, base, base % num_kv_blocks)
                        bz = T.if_then_else(use_sections, bz_sec, base // (heads_kv * num_kv_blocks))
                        mh = 0 if mask_heads == 1 else kv_head
                        k_start = by_kv * block_k
                        q_lo = QRange[bz, mh, by_kv, 0]
                        q_count = QRange[bz, mh, by_kv, 1] + 1 - q_lo
                        total_local = T.cast(local_end - local_begin, "int32")

                        T.copy(K[bz, kv_head, k_start : k_start + block_k, :], k_l1)
                        T.copy(V[bz, kv_head, k_start : k_start + block_k, :], v_l1)
                        T.copy(K[bz, kv_head, k_start : k_start + block_k, :], kt_l1, transpose=True)
                        T.copy(Bounds[bz, mh, 0, k_start : k_start + block_k], b0)
                        T.copy(Bounds[bz, mh, 1, k_start : k_start + block_k], b1)
                        T.copy(Extrema[bz, mh, by_kv, :, :], ext_ub)

                        # C036 flat-head: single 1D pipelined loop across all Q heads in this KV tile
                        for step in T.Pipelined(
                            0, total_local, num_stages=2, annotations={"enable_offset": True}
                        ):
                            pos = T.cast(local_begin, "int32") + step
                            h_idx = pos // q_count
                            qi = q_lo + pos - h_idx * q_count
                            by_head = T.if_then_else(
                                use_sections, head_base_sec + h_idx, kv_head * groups + h_idx
                            )
                            q_start = qi * block_q

                            # Stage 0: GM -> L1/UB loads + Q -> L0A
                            with T.Stage(0):
                                T.copy(Q[bz, by_head, q_start : q_start + block_q, :], q_l1)
                                T.copy(dO[bz, by_head, q_start : q_start + block_q, :], do_l1)
                                T.copy(
                                    LSE[bz, by_head, q_start + _sid * vec_q : q_start + (_sid + 1) * vec_q, :],
                                    lse_ub,
                                )
                                T.copy(
                                    Delta[bz, by_head, q_start + _sid * vec_q : q_start + (_sid + 1) * vec_q],
                                    delta_ub,
                                )
                                T.copy(q_l1, qk_a)

                            # Stage 1: QK + dP + fused 256B SimdVF P & dS + dV
                            with T.Stage(1):
                                T.copy(k_l1, qk_b)
                                T.gemm(qk_a, qk_b, qk_c, transpose_B=True, clear_accum=True)
                                T.dual_copy(qk_c, qk_ub)

                            with T.Stage(1):
                                T.copy(do_l1, qk_a)
                                T.copy(v_l1, qk_b)
                                T.gemm(qk_a, qk_b, dp_c, transpose_B=True, clear_accum=True)
                                T.dual_copy(dp_c, dp_ub)

                            with T.Stage(1):
                                fast = block_keep(ext_ub, q_start, by_kv)
                                if fast == 0:
                                    with T.SimdVF():
                                        full = S.pset(32, "PAT_ALL")
                                        b16 = S.pset(16, "PAT_VL128")
                                        upper_e, upper_o = S.vld2(b0[0], "DINTLV_B32")
                                        lower_e, lower_o = S.vld2(b1[0], "DINTLV_B32")
                                        masked = S.vdup(T.float32(_NEG), "float32", full)
                                        for r in range(vec_q):
                                            se, so = S.vld2(qk_ub[r, 0], "DINTLV_B32")
                                            se = S.vmuls(se, T.float32(scale), full)
                                            so = S.vmuls(so, T.float32(scale), full)
                                            row = q_start + _sid * vec_q + r
                                            ke = S.pand(S.vcmps(upper_e, row, full, op="gt"),
                                                        S.vcmps(lower_e, row, full, op="le"), full)
                                            ko = S.pand(S.vcmps(upper_o, row, full, op="gt"),
                                                        S.vcmps(lower_o, row, full, op="le"), full)
                                            se = S.vsel(se, masked, ke)
                                            so = S.vsel(so, masked, ko)
                                            lse_b = S.vld(lse_ub[r, 0], "BRC_B32")
                                            pe = S.vexp(S.vsub(se, lse_b, full), full)
                                            po = S.vexp(S.vsub(so, lse_b, full), full)
                                            S.vst2(qk_ub[r, 0], pe, po)
                                            pe_b = S.vcvt(pe, "bfloat16", full, round="ROUND_R", sat=False, part=0, mode="MODE_ZEROING")
                                            po_b = S.vcvt(po, "bfloat16", full, round="ROUND_R", sat=False, part=1, mode="MODE_ZEROING")
                                            pm = S.vor(T.reinterpret(pe_b, "uint16x128"), T.reinterpret(po_b, "uint16x128"), b16)
                                            S.vsts(p_ub[r, 0], T.reinterpret(pm, "bfloat16x128"), b16, "NORM_B16")
                                else:
                                    with T.SimdVF():
                                        full = S.pset(32, "PAT_ALL")
                                        b16 = S.pset(16, "PAT_VL128")
                                        for r in range(vec_q):
                                            se, so = S.vld2(qk_ub[r, 0], "DINTLV_B32")
                                            se = S.vmuls(se, T.float32(scale), full)
                                            so = S.vmuls(so, T.float32(scale), full)
                                            lse_b = S.vld(lse_ub[r, 0], "BRC_B32")
                                            pe = S.vexp(S.vsub(se, lse_b, full), full)
                                            po = S.vexp(S.vsub(so, lse_b, full), full)
                                            S.vst2(qk_ub[r, 0], pe, po)
                                            pe_b = S.vcvt(pe, "bfloat16", full, round="ROUND_R", sat=False, part=0, mode="MODE_ZEROING")
                                            po_b = S.vcvt(po, "bfloat16", full, round="ROUND_R", sat=False, part=1, mode="MODE_ZEROING")
                                            pm = S.vor(T.reinterpret(pe_b, "uint16x128"), T.reinterpret(po_b, "uint16x128"), b16)
                                            S.vsts(p_ub[r, 0], T.reinterpret(pm, "bfloat16x128"), b16, "NORM_B16")
                                T.dual_copy(p_ub, p_l1)

                            with T.Stage(1):
                                with T.SimdVF():
                                    full = S.pset(32, "PAT_ALL")
                                    b16 = S.pset(16, "PAT_VL128")
                                    for r in range(vec_q):
                                        de, do = S.vld2(dp_ub[r, 0], "DINTLV_B32")
                                        pe, po = S.vld2(qk_ub[r, 0], "DINTLV_B32")
                                        dl = S.vld(delta_ub[r], "BRC_B32")
                                        dse = S.vmuls(S.vmul(S.vsub(de, dl, full), pe, full), T.float32(scale), full)
                                        dso = S.vmuls(S.vmul(S.vsub(do, dl, full), po, full), T.float32(scale), full)
                                        dse_b = S.vcvt(dse, "bfloat16", full, round="ROUND_R", sat=False, part=0, mode="MODE_ZEROING")
                                        dso_b = S.vcvt(dso, "bfloat16", full, round="ROUND_R", sat=False, part=1, mode="MODE_ZEROING")
                                        dsm = S.vor(T.reinterpret(dse_b, "uint16x128"), T.reinterpret(dso_b, "uint16x128"), b16)
                                        S.vsts(ds_ub[r, 0], T.reinterpret(dsm, "bfloat16x128"), b16, "NORM_B16")
                                T.dual_copy(ds_ub, ds_l1)

                            with T.Stage(1):
                                T.copy(p_l1, dk_a, transpose=True)
                                T.copy(do_l1, dk_b, transpose=True)
                                T.gemm(dk_a, dk_b, dv_c, transpose_B=True, clear_accum=(step == 0))
                                T.copy(ds_l1, dk_a, transpose=True)
                                T.copy(ds_l1[:, :block_q], dq_a)

                            # Stage 2: dK + dQ (two 64x64 chunks)
                            with T.Stage(2):
                                T.copy(q_l1, dk_b, transpose=True)
                                T.gemm(dk_a, dk_b, dk_c, transpose_B=True, clear_accum=(step == 0))
                                T.copy(kt_l1[:, :block_q], dk_b)
                                T.gemm(dq_a, dk_b, dq_c, transpose_B=True, clear_accum=True)
                                T.copy(kt_l1[:, block_q:], dk_b)
                                T.copy(ds_l1[:, block_q:], dq_a)
                                T.gemm(dq_a, dk_b, dq_c, transpose_B=True, clear_accum=False)

                            # Stage 3: atomic dQ writeback
                            with T.Stage(3):
                                T.set_atomic("add", "float32")
                                T.copy(dq_c, dQ[bz, by_head, q_start : q_start + block_q, :])
                                T.set_atomic_none()

                        T.set_atomic("add", "float32")
                        T.copy(dv_c, dV[bz, kv_head, k_start : k_start + block_k, :])
                        T.copy(dk_c, dK[bz, kv_head, k_start : k_start + block_k, :])
                        T.set_atomic_none()

    return flashmask_prefill_bwd_declarative


@pass_instrument
class PassTimer:
    def __init__(self):
        self.t0 = {}
        self.records = []

    def run_before_pass(self, mod, info):
        self.t0[info.name] = time.perf_counter()
        print(f"[PASS ENTER] {info.name}", flush=True)

    def run_after_pass(self, mod, info):
        dt = time.perf_counter() - self.t0.get(info.name, time.perf_counter())
        self.records.append((info.name, dt))
        if dt >= 0.05:
            print(f"[PASS EXIT ] {info.name}: {dt:.3f}s", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    enable_sections = (sys.argv[2] != "nosec") if len(sys.argv) > 2 else True
    print(f"=== Lowering declarative C036 BF16/D128 (version_mode={mode}, enable_sections={enable_sections}) ===", flush=True)
    pf = make_declarative_c036_bwd_prim_func(version_mode=mode, enable_sections=enable_sections)
    timer = PassTimer()
    t_start = time.perf_counter()
    with tilelang.tvm.transform.PassContext(instruments=[timer]):
        artifact = tilelang.lower(pf, target="ascend")
    dt_all = time.perf_counter() - t_start
    print(f"\n=== LOWERING SUCCEEDED in {dt_all:.3f}s (code len={len(artifact.kernel_source)}) ===", flush=True)
    top = sorted(timer.records, key=lambda x: -x[1])[:12]
    print("Top 12 slowest passes:")
    for name, dt in top:
        print(f"  {dt:8.3f}s  {name}")


if __name__ == "__main__":
    main()
