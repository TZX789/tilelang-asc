# T.MixedKernel() 设计文档（Ascend 方言）


---

## 1. 函数简介

`T.MixedKernel()` 是 Ascend 方言里声明 **AIC（Cube）与 AIV（Vector）混合核（mixed kernel）launch 入口**的原语。与 `T.Kernel()` 相比，它多返回一个 **AIV 子核 ID（`sid`）**，并允许显式指定激活几个 AIV。

1. 声明 1-D AI core 网格（`blockIdx.x`），返回 `bx`。
2. 额外返回 **`sid`** —— 每个 AIC 下的 AIV 子核编号（`asc_get_sub_block_id()`，取值 0/1），用于按子核分派不同工作。
3. 可选 `sids` 控制活跃 AIV 数量：`sids=1` 只激活 AIV0，`sids=2` 激活 AIV0+AIV1。

典型形态：

```python
with T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid):
    with T.Cube():    # AIC（Cube）侧，用 bx
        ...
    with T.Vector():  # AIV（Vector）侧，用 bx 和 sid
        ...
```

---

## 2. 功能规格（输入参数支持的规格）

函数签名（[kernel.py](../../tilelang/ascend/language/kernel.py#L175-L180)）：

```python
@kernel_launch_factory
def MixedKernel(
    *blocks: int | tirx.PrimExpr,
    sids: int = 2,
    prelude: str | None = None,
) -> KernelLaunchFrame:
```

### 2.1 `*blocks`（网格维度）

| 项目    | 规格                                                                                                                    |
| ----- | --------------------------------------------------------------------------------------------------------------------- |
| 维度数   | **恰好 1 个**，1-D 网格。多传抛 `ValueError`（见 `len(blocks) != 1` 分支）                                                           |
| 类型    | `int`（静态常量）或 `tirx.PrimExpr`（动态表达式）                                                                                   |
| 语义    | `blockIdx.x` 的 extent，即启动的 AI core 数量                                                                                 |
| 动态表达式 | 符号化 shape 变量（`tirx.Var`）及其派生的整型算术表达式，如 `T.ceildiv(N, 128)`、`(N + 127) // 128`、`T.min(N, 256)`、`m * n` 等；表达式必须为整型标量表达式 |
| 运行时校验 | 必须为整型标量、`extent > 0`、可装入 `uint32`（值在运行时从 host 参数提取后校验）|

### 2.2 `sids`（活跃 AIV 子核数）

| 项目 | 规格 |
|---|---|
| 类型 | `int` |
| 默认值 | `2` |
| 允许值 | **`1` 或 `2`**，其它值抛 `ValueError`（[kernel.py](../../tilelang/ascend/language/kernel.py#L220-L221)） |
| 语义 | `sids=1` 只激活 AIV0；`sids=2` 保留双 AIV 启动。
| 落点 | 作为 `cthread_extent`（即 `vector_count`）传给后端，生成 `sid` 绑定 |

### 2.3 `prelude`（可选注入源码）

| 项目 | 规格                                                                |
| -- | ----------------------------------------------------------------- |
| 类型 | `str` 或 `None`                                                    |
| 语义 | 注入到生成 kernel 之前的 AscendC 源码（`#include`、`__aicore__ inline` 辅助函数等） |

### 2.4 不支持的参数


- **没有 `threads=`**：签名不含该形参，传 `threads=128` 直接 `TypeError`。线程域须在 body 内用 `T.SimtVF(threads=...)` 或 `T.SimdVF()` 声明。
- **多维网格**：`len(blocks) != 1` 抛 `ValueError`；`sids` 之外的任意非 1/2 取值（如 `sids=4`）抛 `ValueError`。

### 2.5 调用上下文约束

- 仅能在 `@T.prim_func` / `@tilelang.jit` 内调用，否则抛 `JITNoBuilderError`（[kernel.py](../../tilelang/ascend/language/kernel.py#L214-L215)）。
---

## 3. 实现方案分析（前端 `with T.MixedKernel()` 到后端发生了什么）

四个阶段：**前端 trace → IR 物化 → Codegen → Runtime 启动**。

1. **前端 trace**：`T.MixedKernel(N, sids=2, prelude=...)` 校验 `len(blocks)==1` 与 `sids ∈ {1,2}`，把 `prelude` 写入 `attrs["pragma_import_c"]`，再调用 `_ffi_api.MixedKernelLaunch(blocks, sids, attrs)`（[kernel.py](../../tilelang/ascend/language/kernel.py#L228)）。`with ... as (bx, sid)` 由 frame 按序解包出两个绑定变量。

2. **Frame 构造**（[ir.cc](../../src/ascend/ir.cc#L37-L87)）：`MixedKernelLaunch` 构造一个 `KernelLaunchFrame`，内部依次：
   - **Frame 0**：`bx = blockIdx.x`（target-neutral 的 thread_binding For）。
   - **Frame 1**：`sid` = `"cthread"` 线程 tag，extent 为 `sids`（即 `vector_count`）；codegen 把它落成 `asc_get_sub_block_id()`。
   - **Frame 2**：线程占位符（Ascend 管线丢弃，同 `T.Kernel()`）。
   - **Frame 3**：MainBlock，annotation 记录 `vector_count = sids` 与 `pragma_import_c`。
   - C++ 侧同样 `ICHECK` 1-D 网格、`vector_count` 必须是常量 1 或 2（[ir.cc](../../src/ascend/ir.cc#L43-L48)）。

3. **IR 物化**（`MaterializeKernelLaunch`）：Ascend 管线声明 `launch_dim_tags=["cthread"]`，于是 `blockIdx.x` 和 `cthread`（`sid`）两个 launch 维度都被物化成 `thread_extent` AttrStmt；线程占位符丢弃。`sid` 因此以 block 级 launch 维度存在，而非 SIMT 线程。

4. **Codegen / Runtime**：codegen（[codegen_ascend.cc](../../src/ascend/codegen/codegen_ascend.cc#L3046-L3047)）把 `cthread` tag emit 为 `asc_get_sub_block_id()`；launch 的 block 数仍由 runtime 从 host 参数累乘 `num_blocks` 得到，与 `T.Kernel()` 一致。

---

## 4. 接口设计

```python
def MixedKernel(
    *blocks: int | tirx.PrimExpr,
    sids: int = 2,
    prelude: str | None = None,
) -> KernelLaunchFrame:
```

### 4.1 参数

| 名称 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `blocks` | `int \| tirx.PrimExpr` | 是（恰好 1 个） | 1-D AI core 网格 extent（`blockIdx.x`） |
| `sids` | `int` | 否（默认 `2`） | 活跃 AIV 子核数，仅 `1` / `2` |
| `prelude` | `str \| None` | 否 | 注入的 AscendC 源码，映射为 `pragma_import_c` 注解 |

### 4.2 返回值

`KernelLaunchFrame`。`with ... as (bx, sid)` 返回两个绑定：

- `bx`：`blockIdx.x` 的 grid var。
- `sid`：`cthread` 维度的 var，运行时对应 `asc_get_sub_block_id()`，取值 0/1。

### 4.3 异常

| 场景 | 异常 |
|---|---|
| `len(blocks) != 1` 的多维网格 | `ValueError` |
| `sids ∉ {1, 2}` | `ValueError` |
| `sids` 非常量 1/2（C++ 侧兜底校验） | `ICHECK` 失败 |
| 传入 `threads=` 等未声明关键字 | `TypeError`（Python 签名层） |
| 在 `@T.prim_func` / `@tilelang.jit` 之外调用 | `JITNoBuilderError` |

### 4.4 与 `T.Kernel()` 的边界

| | `T.Kernel(N)` | `T.MixedKernel(N, sids=...)` |
|---|---|---|
| 返回绑定 | `bx` | `(bx, sid)` |
| AIV 子核 ID | 不可见 | 可见（`asc_get_sub_block_id()`） |
| 显式控制 AIV 数 | 不支持（编译器自动拆核） | `sids=1` / `sids=2` |
| 适用场景 | 纯 Cube / 纯 Vector，或无需区分 AIV 角色 | 需要按 `sid` 分派 AIV 工作，或显式指定 AIV 数量 |

---

## 5. 示例程序


```python
import torch
import tilelang
import tilelang.ascend.language as T


def make_split_scale(N: int, TILE: int, NUM_BLOCKS: int):
    NUM_TILES = N // TILE  # 每条 slice 64 个 float32 = 一条 2048-bit 向量

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        # sids=2 -> 每个 block 两个 AIV 子核；sid ∈ {0, 1}
        with T.MixedKernel(NUM_BLOCKS, sids=2) as (bx, sid):
            a = T.alloc_shared((TILE,), "float32")
            c = T.alloc_shared((TILE,), "float32")

            # 关键：全局 AIV 线性编号。AIV0 承担偶数 tile，AIV1 承担奇数 tile。
            aiv_idx = bx * 2 + sid
            for t in T.serial(aiv_idx, NUM_TILES, NUM_BLOCKS * 2):
                base = t * TILE
                T.copy(A[base : base + TILE], a)

                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(TILE // 64):
                        r = T.simd.vld(a[i * 64])
                        r2 = T.simd.vadd(r, r, mask)   # 示例：x2
                        T.simd.vsts(c[i * 64], r2, mask)

                T.copy(c, C[base : base + TILE])

    return main


if __name__ == "__main__":
    N, TILE, NUM_BLOCKS = 65536, 64, 8
    kernel = tilelang.compile(
        make_split_scale(N, TILE, NUM_BLOCKS), target="ascend", out_idx=-1
    )
    a = torch.randn(N, device="npu")
    c = torch.empty_like(a)
    kernel(a, c)
    torch.npu.synchronize()
    assert torch.allclose(c, a * 2)
```

要点：

- `aiv_idx = bx * 2 + sid` 把 `bx`、`sid` 拼成全局 AIV 编号，`T.serial(aiv_idx, NUM_TILES, NUM_BLOCKS * 2)` 让两个子核交错分派 tile，互不重叠。
- 若同时有 Cube 与 Vector 工作，把 `T.Cube()` / `T.Vector()` 作用域放进 `with T.MixedKernel(...) as (bx, sid):` 内，Cube 侧用 `bx`、Vector 侧用 `bx` 和 `sid`。