# T.Kernel() 设计文档（Ascend 方言）

***

## 1. 函数简介

`T.Kernel()` 是 Ascend 方言里声明 **kernel launch（核函数启动）入口**的原语。

1. **声明一个 1-D 的 AI core 网格**，即 NPU 上的 `blockIdx.x`（核/l2 core 索引），作为 kernel body 的程序索引空间。
2. **打开一个 launch 作用域**（`with` 语句），并把这个网格索引绑定成变量（`as bx`），供 body 内所有算子引用。
3. **可选地注入 AscendC 源码**（`prelude`），用于引入 `#include`、辅助函数等，供 `T.call_extern` 等调用。

典型形态：

```python
with T.Kernel(NUM_CORES) as bx:
    # bx 即 blockIdx.x，为 block 级程序索引
    ...
```

**关键设计约束**：Ascend 的两类计算核（AIC Cube / AIV Vector）在 launch 这一层**没有 SIMT 线程域**——即 `T.Kernel()` 没有 `threads=` 参数，也不分 `threadIdx`。真实线程域要写进 body 内部，用：

- `T.SimtVF(threads=N)`：线程并行域（真实 threadIdx 语义）
- `T.SimdVF()`：寄存器级向量域（CCE MicroAPI，无线程概念）

需要区分/指定 AIV 子核时，用配套的 `T.MixedKernel()`（返回 `(bx, sid)`）；`T.Kernel()` 则由编译器自动判定 core 类型：AIC算子、Vector算子或CV混合算子。

***

## 2. 功能规格（输入参数支持的规格）

函数签名（[kernel.py](../../tilelang/ascend/language/kernel.py#L138-L142)）：

```python
@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
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
| 运行时校验 | 必须为整型标量、`extent > 0`、可装入 `uint32`（值在运行时从 host 参数提取后校验）                                                                |

### 2.2 `prelude`（可选注入源码）

| 项目 | 规格                                                                |
| -- | ----------------------------------------------------------------- |
| 类型 | `str` 或 `None`                                                    |
| 语义 | 注入到生成 kernel 之前的 AscendC 源码（`#include`、`__aicore__ inline` 辅助函数等） |

### 2.3 不支持的参数

- **没有 `threads=`**：签名不含该形参，传 `threads=128` 直接 `TypeError`（Python 层面拒绝，而非到后端才报错）。
- **1-D 之外的多维网格**：`len(blocks) != 1` 抛 `ValueError`。

### 2.4 调用上下文约束

- 仅能在 `@T.prim_func` 或 `@tilelang.jit` 上下文中调用，否则抛 `JITNoBuilderError`（无 Builder）。

***

## 3. 实现方案分析

四个阶段：**前端 trace → IR 物化（MaterializeKernelLaunch）→ Codegen → Runtime 启动**。

1. **前端 trace**：`T.Kernel(N, prelude=...)` → `launch_kernel` → `_ffi_api.KernelLaunch(blocks, threads=None, attrs)`，创建 `KernelLaunchFrame`（`tl.KernelLaunchFrame`）；`with ... as bx` 由 `__enter__` 返回绑定变量 `bx`（`blockIdx.x` 的 grid var）。`prelude` 写入 `pragma_import_c` 注解。trace 出的 IR 是 `for bx in thread_binding(blockIdx.x)` 的 launch 嵌套，外加三个 `tl.launch_thread_idx` 线程占位符和带 annotation 的 `tilelang_root` block。
2. **IR 物化**（[materialize\_kernel\_launch.cc](../../src/transform/materialize_kernel_launch.cc)）：Ascend 配置为 `lower_grid_binding=true`（`blockIdx.x` 循环 → `thread_extent` AttrStmt）、`lower_thread_binding=false`（丢弃线程占位符，body 引用则报错）、`launch_dim_tags=["cthread"]`（供 `MixedKernel` 用）。
3. **Codegen**（`src/ascend/codegen/rt_mod_ascend.cc`）：把 `blockIdx.x` extent 注册为 launch 参数（`kKernelLaunchParams` → `launch_param_tags`），使网格大小成为 host 端可动态传入的标量；把 `pragma_import_c`（`prelude`）输出到源码头部；生成 AscendC kernel。
4. **Runtime**（[ascend\_module.cc](../../src/ascend/runtime/ascend_module.cc)）：`LaunchParamConfig::Extract(args)` 从 host 标量参数取出各轴 extent，累乘得 `num_blocks` 并校验 `>0` 与 `uint32` 上限，再启动。

> 结论：`blocks` 的动态表达式在 host 端求值、作为标量 launch 参数传入，capacity 校验在 launch 前发生。

***

## 4. 接口设计

```python
def Kernel(
    *blocks: int | tirx.PrimExpr,
    prelude: str | None = None,
) -> KernelLaunchFrame:
```

### 4.1 参数

| 名称        | 类型                     | 必填        | 说明                                      |
| --------- | ---------------------- | --------- | --------------------------------------- |
| `blocks`  | `int \| tirx.PrimExpr` | 是（恰好 1 个） | 1-D AI core 网格 extent（`blockIdx.x`）     |
| `prelude` | `str \| None`          | 否         | 注入的 AscendC 源码，映射为 `pragma_import_c` 注解 |

### 4.2 返回值

`KernelLaunchFrame`（对象 `tl.KernelLaunchFrame`，继承 `TIRFrame`）。`with ... as bx` 返回绑定变量：

- 1-D 时归一化为**单个 `Var`**（`bx`），同时支持 `as (bx,)` 解包语义。

### 4.3 异常

| 场景                                      | 异常                      |
| --------------------------------------- | ----------------------- |
| `len(blocks) != 1` 的多维网格                | `ValueError`            |
| 传入 `threads=` 等未声明关键字                   | `TypeError`（Python 签名层） |
| 在 `@T.prim_func` / `@tilelang.jit` 之外调用 | `JITNoBuilderError`     |
| body 引用被丢弃的线程索引（Ascend）                 | 编译期 `LOG(FATAL)`        |
| 运行时 extent ≤ 0 或超出 uint32               | launch 前校验失败            |

### 4.4 与 `T.MixedKernel()` 的边界

|            | `T.Kernel(N)`                  | `T.MixedKernel(N, sids=...)`       |
| ---------- | ------------------------------ | ---------------------------------- |
| 返回绑定       | `bx`                           | `(bx, sid)`                        |
| AIV 子核 ID  | 不可见                            | 可见（`asc_get_sub_block_id()`）       |
| 显式控制 AIV 数 | 不支持（编译器自动拆核）                   | `sids=1` / `sids=2`                |
| 适用场景       | 纯 Cube / 纯 Vector，或无需区分 AIV 角色 | 需要按 `sid` 分派不同 AIV 工作，或显式指定 AIV 数量 |

二者共享 1-D 网格、无 `threads=` 的约束。

***

## 5. 示例程序

一个完整的向量加 TileLang 算子：每个 block 处理一段数据，block 内部用 `T.SimtVF` 打开线程域做元素加。

```python
import torch
import tilelang
import tilelang.ascend.language as T


def make_vecadd(N: int, threads: int = 128):
    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        # 1-D core 网格：N // threads 个 block，动态用 ceildiv 兜底
        with T.Kernel(T.ceildiv(N, threads)) as bx:
            a_ub = T.alloc_shared((threads,), "float32")
            b_ub = T.alloc_shared((threads,), "float32")
            base = bx * threads

            # GM -> UB
            T.copy(A[base : base + threads], a_ub)
            T.copy(B[base : base + threads], b_ub)

            # 打开真实线程域（Ascend 无 launch 级线程）
            with T.SimtVF(threads=threads):
                for i in T.Parallel(threads):
                    a_ub[i] = a_ub[i] + b_ub[i]

            # UB -> GM
            T.copy(a_ub, C[base : base + threads])

    return main


if __name__ == "__main__":
    N = 1024 * 1024
    kernel = tilelang.compile(
        make_vecadd(N), target="ascend", out_idx=-1
    )
    a = torch.randn(N, device="npu")
    b = torch.randn(N, device="npu")
    c = torch.empty_like(a)
    kernel(a, b, c)
    torch.npu.synchronize()
    assert torch.allclose(c, a + b)
```

要点：

- `T.Kernel(T.ceildiv(N, threads))` 演示了 `blocks` 支持动态表达式。
- `T.SimtVF(threads=threads)` 演示了线程域必须声明在 body 内（`T.Kernel()` 本身无 `threads=`）。
- `T.alloc_shared` 分配 UB，`T.copy` 完成 GM↔UB 搬运。
