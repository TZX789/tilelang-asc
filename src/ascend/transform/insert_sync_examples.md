# InsertSync 功能示例手册

> 本文以**具体 TileLang 例子**演示 `src/ascend/transform/insert_sync.cc` 的各项功能。
> 该 Pass 的职责是：**Synchronization analysis, flag allocation, and insertion**（同步分析、flag 分配与插入）。
>
> 它读入已完成 `ResolveCore` / `PrepareMultiBuffer` / 调度的 *scheduled TIR*，
> 在存在数据依赖的任务之间插入 Ascend 硬件同步原语。所有示例中「插入的同步」均指向
> `SyncMaterializer`（`insert_sync.cc` 中的 `InsertSyncPoints` / `MakeFlagActions`）生成的 IR。

---

## 0. 先备知识：四种同步原语

InsertSync 最终只产出五条底层调用（见 `MakeAscend*` 系列辅助函数）：

| 原语 | 底层调用 | 何时使用 |
| --- | --- | --- |
| 同核 flag | `ascend_set_flag(硬事件, id)` / `ascend_wait_flag(硬事件, id)` | 同一 core 上、跨 pipe 的依赖 |
| 同 pipe 屏障 | `ascend_pipe_barrier("PIPE_xxx")` | 同一 core 上、同 pipe 的乱序访问（MTE/FIX） |
| 跨核 flag | `ascend_cross_core_set_flag(4, "PIPE_xxx", id)` / `..._wait_flag(...)` | Cube ↔ AIV 之间的依赖 |
| 隐式顺序 | （无） | 同 core、同 pipe 且为 S / V / M |

**核心（CoreMask）**：`kCoreVector0=1`（AIV0）、`kCoreCube=2`（Cube）、`kCoreVector1=4`（AIV1）。
**流水线（ResourcePipe）**：`MTE1`（L1→L0A/L0B）、`MTE2`（GM→UB/L1）、`MTE3`（UB/L1→GM）、
`M`（矩阵乘 Cube）、`V`（向量）、`FIX`（L0C→UB/GM）、`S`（标量/配置）。

**同一物理依赖边的判定（`InsertSyncPoints` 中的三分支）**：

```
if (非跨核 && src.pipe == dst.pipe)  → 隐式顺序 ? 跳过 : 插 pipe_barrier
else if (跨核)                        → 插 cross_core_set/wait_flag (mode 4)
else                                  → 插 set_flag / wait_flag
```

下面每个例子分别对应一个分支或一个功能点。

---

## 1. 同核跨 pipe：`set_flag` / `wait_flag`

### 1.1 WAR：DMA 写回必须先于下一次搬运（`MTE3_MTE2`）

同一块 `ub`（UB）先被 MTE3 写回全局，再被 MTE2 从全局搬入覆盖，
这两个访问在不同 pipe（MTE3 vs MTE2）但同一 core（AIV）上乱序执行，
必须用 flag 保证「写完才允许覆盖」。

```python
@T.prim_func
def main(A: T.Tensor((64,), "float32"), C: T.Tensor((64,), "float32")):
    with T.Kernel(1):
        ub = T.alloc_shared((64,), "float32")
        T.copy(ub, C)      # UB -> GM，MTE3
        T.copy(A, ub)      # GM -> UB，MTE2
```

InsertSync 插入：

```
set_flag("MTE3_MTE2", 0)    # 生产者(MTE3) 结尾之后
wait_flag("MTE3_MTE2", 0)   # 消费者(MTE2) 开头之前
```

> 对应测试：`testing/ascend/auto_schedule/test_sync_dependencies.py::test_dma_issue_order_is_not_completion_order`
> 验证了「DMA 发射顺序 ≠ 完成顺序」，因此必须显式 set/wait。

### 1.2 RAW：L0A 载入完成后才能 GEMM（`MTE1_M`）

`T.copy(L1 → L0A)` 走 MTE1，`T.gemm` 走 M pipe，二者都在 Cube 上，
但 M 依赖 MTE1 搬入的 L0A 数据：

```python
T.copy(A_l1, A_l0a)                       # L1 -> L0A，MTE1（Cube）
T.gemm(A_l0a, B_l0b, C_l0c, ...)          # M（Cube）
```

插入：

```
set_flag("MTE1_M", 0)    # 在 MTE1 载入之后
wait_flag("MTE1_M", 0)   # 在 gemm 之前
```

> 对应测试：`test_partial_l0_loads_complete_before_gemm_and_order_overlapping_writes`
> 显式断言了 `set_flag("MTE1_M")` / `wait_flag("MTE1_M")` 且 set 位置 < wait 位置 < gemm 位置。

### 1.3 硬事件归一化

`GetHardEvent(producer_pipe, consumer_pipe)` 先拼接 `producer_consumer`，
再做少量归一化（涉及 Cube 的 M pipe 时改走 MTE1 完成事件）：

| 输入 | 归一化结果 |
| --- | --- |
| `MTE2_M` | `MTE2_MTE1` |
| `M_MTE2` | `MTE1_MTE2` |
| `MTE3_M` | `MTE3_MTE1` |
| `M_MTE3` | `MTE1_MTE3` |

例如 gemm（M）产出、MTE2 消费：`M_MTE2` → `MTE1_MTE2`。

---

## 2. 同核同 pipe：`pipe_barrier`

MTE/FIX 的**发射顺序并不能保护重叠访问的完成顺序**（`HasImplicitSamePipeOrder`
只对 `S`/`V`/`M` 返回 true）。因此同 pipe 的重叠访问需要 `pipe_barrier`。

### 2.1 重叠的 MTE2 写（`PIPE_MTE2`）

```python
for row in T.serial(2):
    T.copy(A[row, :], ub[0, :])   # 每次迭代写 ub 第 0 行，重叠 -> MTE2 vs MTE2
```

插入：

```
pipe_barrier("PIPE_MTE2")
```

> 对应测试：`test_dma_row_writes_only_serialize_overlapping_regions`——
> 行不相交（`disjoint-rows`）时不插屏障，重叠（`same-row`）时插 `PIPE_MTE2`。

### 2.2 重叠的 MTE1 载入（`PIPE_MTE1`）

两个都写到同一 L0A 区域、且 pitch 不同导致物理重叠的 L1→L0A 拷贝：

```python
T.copy(l1[0:32], l0a[0:32])    # pitch=4
T.copy(l1[32:64], l0a[32:64])  # 与上一条物理重叠
```

插入：

```
pipe_barrier("PIPE_MTE1")
```

> 对应测试：`test_partial_l0_loads_complete_before_gemm_and_order_overlapping_writes`
> 与 `test_different_l0_pitches_conflict_within_same_plane`。

---

## 3. 隐式顺序：不插任何同步

当依赖边满足「同 core + 同 pipe + 该 pipe ∈ {S, V, M}」时，
`HasImplicitSamePipeOrder` 判定为隐式顺序，`InsertSyncPoints` 直接 `continue`。

```python
# 两个标量 (S pipe) 任务在同一个 AIV 上顺序发射，天然有序
T.some_scalar_op()
T.some_scalar_op()
```

不插入任何 `set_flag` / `wait_flag` / `pipe_barrier`。

> 代码依据：`HasImplicitSamePipeOrder` 返回 `pipe == "S" || pipe == "V" || pipe == "M"`。

---

## 4. 跨核 flag：`cross_core_set_flag` / `cross_core_wait_flag`（mode 4）

Cube 与 AIV 之间用 mode-4 跨核 flag。AIV 有 16 个槽，按 `sid * 16 + id` 寻址；
Cube 端要同时覆盖两个 AIV bank（`id` 与 `id + 16`）。

### 4.1 Cube → AIV：Fixpipe 产出、MTE3 消费（`PIPE_FIX` → `PIPE_MTE3`）

```python
@T.prim_func
def main(C: T.Tensor((16, 16), "float32")):
    with T.MixedKernel(1):
        accum = T.alloc_l0c((16, 16), "float32")
        ub    = T.alloc_shared((8, 16), "float32")
        T.dual_copy(accum, ub)   # FIX (L0C->UB, Cube) 写 ub
        T.dual_copy(ub, C)       # MTE3 (UB->GM, AIV) 读 ub
```

插入（生产者是 Cube，set 在 Cube 端且覆盖两个 bank）：

```
cross_core_set_flag(4, "PIPE_FIX",  id)     # Cube 端, bank0
cross_core_set_flag(4, "PIPE_FIX",  id+16)  # Cube 端, bank1
cross_core_wait_flag(4, "PIPE_MTE3", id)    # AIV 端
```

> 对应测试：`testing/ascend/auto_schedule/test_cross_core_flags.py::test_alternating_epilogue_uses_each_aiv_flag_bank`
> 断言 Cube 端 wait 的 id 同时覆盖 `<16` 与 `>=16` 两个 bank。

### 4.2 AIV → Cube：标量写后读（`PIPE_S`）

两个 AIV 各自写 `ss`，Cube 读它们（或反向）。标量走 `PIPE_S`：

```python
# AIV0 / AIV1 分别写 ss，Cube 读 ss
```

插入（AIV → Cube 方向）：

```
cross_core_set_flag(4, "PIPE_S", id)     # AIV 端
cross_core_wait_flag(4, "PIPE_S", id)    # Cube 端（覆盖对应 bank）
```

> 对应测试：`test_ss_uses_standard_cross_core_sync`（`aivs-to-cube` / `cube-to-aivs` 两个方向）。

### 4.3 显式 flag 与本 Pass 生成 flag 的共存（避让）

用户可以在 kernel 里显式使用跨核 flag，InsertSync 的分配器必须**避让**这些槽位：

```python
T.dual_copy(accum, ub)
T.ascend_sync_inter_arrive("PIPE_FIX", 15)   # 显式占用槽 15
T.ascend_sync_inter_wait ("PIPE_FIX", 15)
T.dual_copy(ub, C)
```

InsertSync 生成的 `cross_core_set/wait_flag` 的 id 会避开 15（`%16` 后不与 `{15}` 相交）。

> 对应测试：`test_generated_flags_do_not_collide_with_explicit_flags`。
> 显式 flag 的占用由 `ExplicitCrossCoreFlagCollector` 用 `ConstIntBound` 推断，
> 若 id 区间无界则直接报错（`test_unbounded_explicit_flag_is_rejected`）。

---

## 5. 跨迭代环（multibuffer / ring）：版本化 flag

多缓冲区（`annotate_buffer_versions`）产生跨越迭代的依赖（`distance ≥ 1`），
flag id 需要**随存储时钟取模**，并在 kernel 最前/最后做首尾的 seed/drain 形成环。

```python
@T.prim_func
def main(A, C):
    with T.Kernel(1):
        ub = T.alloc_shared((64,), "float32")
        T.annotate_buffer_versions({ub: 2})          # 2 个版本

        for i in T.Pipelined(4, num_stages=2):
            T.copy(A[i*64:(i+1)*64], ub)             # MTE2 写 ub（生产者）
            T.copy(ub, C[i*64:(i+1)*64])             # MTE3 读 ub（消费者）
```

依赖结构（`slot = i % 2`，2 个版本）：

- **同迭代 RAW（forward）**：第 `i` 次迭代内，MTE2 写 `ub[slot]`、MTE3 读 `ub[slot]`，
  走 `MTE2_MTE3`，距离 `0`。
- **跨迭代 WAR（backward）**：第 `i` 次迭代的 MTE2 会覆盖第 `i-2` 次迭代 MTE3 读过的
  同一个 `slot`（因为 `(i-2) % 2 == i % 2`），必须先等它读完，走 `MTE3_MTE2`，距离 `2`。

一前一后两条边首尾相接，才构成**环**。插入（示意，`base` 由分配器给定）：

```
# 环的 seed（prefix）：预置每个槽「已腾空」令牌，让首轮即可直接写
set_flag("MTE3_MTE2", base + 0)
set_flag("MTE3_MTE2", base + 1)

# 循环体内（每个 slot 一进一出，共 4 条）
wait_flag("MTE3_MTE2", base + i % 2)   # ① 等上一代读者(i-2)腾出 slot
set_flag( "MTE2_MTE3", base + i % 2)   # ② MTE2 写完后通知「数据就绪」
wait_flag("MTE2_MTE3", base + i % 2)   # ③ 消费者等数据就绪
set_flag( "MTE3_MTE2", base + i % 2)   # ④ MTE3 读完后通知「槽已腾空」

# 环的 drain（suffix）：回收末轮 ④ 的令牌，与 seed 配对成闭环
wait_flag("MTE3_MTE2", base + 0)
wait_flag("MTE3_MTE2", base + 1)
```

时序示意：

```
一次迭代 i 的 flag 链（slot = i % 2）

  seed(prefix)   set(MTE3_MTE2,0) / set(MTE3_MTE2,1)   ← 预置每槽「已腾空」

  ── 迭代 i ─────────────────────────────────────────────────────────
   ①  wait(MTE3_MTE2, slot)    等上一代读者(i-2)腾出本槽
         MTE2 : A[i] ──► ub[slot]
   ②  set (MTE2_MTE3, slot)    通知「数据就绪」
   ③  wait(MTE2_MTE3, slot)    等数据就绪
         MTE3 : ub[slot] ──► C[i]
   ④  set (MTE3_MTE2, slot)    通知「槽已腾空」 ──► 被 i+2 的 ① wait 消费
  ─────────────────────────────────────────────────────────────────────

  反向边（WAR，距离 2）：④ 与 ① 同 slot，slot 每 2 次迭代复用一次
  drain(suffix)  wait(MTE3_MTE2,0) / wait(MTE3_MTE2,1)  ← 回收末轮④令牌，与 seed 配对
```

> 对应测试：`test_sync_dependencies.py::test_ring_flags_use_the_data_clock`
> 验证动态 flag id 形如 `clock % 2 + base`；prefix/suffix 的常量 id 集合一致。
> `InsertBoundaryFlags`（`insert_sync.cc`）负责生成 kernel 首尾的 seed/drain。

版本数 `num_versions` 由 `OptimizeFlagVersions` 计算（反向最近距离），
`MakeFlagEventId` 用 `indexmod(iteration, num_versions) + base_id` 生成 id。

---

## 6. counter / write_counter：生成（generation）协议

### 6.1 为什么不能用词法时钟

§5 的 `iteration` 模式用循环变量 `i` 当 flag 时钟（`slot = i % num_versions`），
前提是**每次迭代生产者都写**。一旦生产者带条件（如 `if i % 3 == 0` 才写），
相邻两次发布之间隔着不确定个数的不写迭代，词法 `i` 就无法跟踪「当前写到第几代」。

`write_counter` 模式因此引入一个**标量生成计数器** `x_counter`（`local.var`），
由 `PrepareMultiBuffer` 在每次写之后插入 `x_counter += 1`（称为"advance"任务）。
真正的时钟是 `generation = x_counter[0]`，而不是 `i`。

```python
@T.prim_func
def main(A, C):
    with T.Kernel(1):
        x = T.alloc_shared((64,), "float32")
        T.annotate_buffer_versions({x: (2, "write_counter")})   # 2 个物理槽 + 生成计数器
        for i in T.Pipelined(7, num_stages=2):
            if i % 3 == 0:
                T.copy(A[i, :], x)     # 生产者：写 x（写周期结束后 x_counter += 1）
            T.copy(x, C[i, :])         # 消费者：读 x
```

### 6.2 通道与状态

InsertSync 为每个「生产者 lane → 消费者 lane」对建立一条 `Protocol`，
内含两个通道（`SyncPoint`）和一个状态变量：

| 对象 | 方向 | 距离 | 底层事件（本例同核 MTE2↔MTE3） | 含义 |
| --- | --- | --- | --- | --- |
| `ready`（发布） | 生产者 → 消费者 | `0` | `MTE2_MTE3` | 「第 gen 代数据已就绪」 |
| `free`（回收） | 消费者 → 生产者 | `num_versions` | `MTE3_MTE2` | 「第 gen 代槽位已消费，可复用」 |
| `acquired` | 消费者私有 | — | `local.var`（初值 `-1`） | 消费者已 acquire 的最新生成号 |

两个通道的物理 flag id 都按**数据时钟**取模：
`flag_id = base + generation % num_versions`（`MakeFlagEventId` + `BufferLoad(counter)`，见 `AddChannel`）。

> 与 §5 的唯一本质区别只在「时钟来源」：§5 用词法 `i`，本节用标量 `x_counter`。
> 其余（token 环形、首尾 seed/drain）完全同构。

### 6.3 三种 acquire 动作与 seed/drain

`BuildGenerationActions` 为每个消费者生成三类动作（`AcquisitionActionKind`）：

- **`kInitialize`**（protocol.begin）：`acquired = -1`。
- **`kAcquire`**（每处读之前，`MakeAcquire`）：

  ```
  if (acquired != generation) {
      wait(ready, generation)      // 等生产者发布这一代
      acquired = generation
  }
  ```
  用 `acquired != generation` 去重：同一代数据被多次读、或消费者也带 guard 时，
  不会重复 wait。

- **`kRetire`**（生产者 advance 之前，见 `MaterializeAction(kRetire)`）：

  ```
  if (generation >= 0) {
      if (acquired != generation) { wait(ready, generation); acquired = generation }
      set(free, generation)        // 释放上一代占用的槽
  }
  ```
  即：先保证本代已被 acquire，再把它占的槽标记为 free。

外加对称的**首尾**（与 §5 一致）：

- **seed**（protocol.begin）：`set(free, 0..N-1)` —— 预置所有槽可写。
- **drain**（protocol.end）：`wait(free, 0..N-1)` + 一次收尾 `kRetire`。

### 6.4 物化后的 IR（结合例子，`N=2`，`base` 由分配器给定）

```
// ① init
x_acquired = -1

// ② seed（free 全槽预置）
set_flag("MTE3_MTE2", base + 0)
set_flag("MTE3_MTE2", base + 1)

// ③ 消费者：读之前
if (x_acquired != x_counter) {
    wait_flag("MTE2_MTE3", base + x_counter % 2)
    x_acquired = x_counter
}
T.copy(x, C[i, :])

// ④ 生产者：写周期 advance 之前（guard = i % 3 == 0）
wait_flag("MTE3_MTE2", base + (x_counter + 1) % 2)    // 等下一个槽位腾空
if (x_counter >= 0) {
    if (x_acquired != x_counter) { wait_flag("MTE2_MTE3", base + x_counter % 2); x_acquired = x_counter }
    set_flag("MTE3_MTE2", base 5+ x_counter % 2)        // 释放本代槽位
}
T.copy(A[i, :], x)          // 生产者写
x_counter += 1              // advance

// ⑤ 写周期结束（epoch.end）
set_flag("MTE2_MTE3", base + x_counter % 2)            // 发布 ready

// ⑥ drain（protocol.end）
wait_flag("MTE3_MTE2", base + 0)
wait_flag("MTE3_MTE2", base + 1)
```

> 以上是逻辑展开；在软件流水线调度里，③/④/⑤ 被放到各自 site
> （读前 / advance 前 / epoch.end），并由生产者 guard 门控。

时序示意（token 流）：

```
  生成       gen=0          gen=1          gen=2          ...
  槽位       slot0          slot1          slot0          ...

  生产者    wait FREE[0]   wait FREE[1]   wait FREE[0]
                │              │              │
             写 slot0        写 slot1        写 slot0
              counter=1      counter=2      counter=3
              set READY[0]   set READY[1]   set READY[0]   ← 发布
                │              │              │
  消费者    wait READY[0]  wait READY[1]  wait READY[0]
             读 slot0        读 slot1        读 slot0
              set FREE[0]   set FREE[1]   set FREE[0]      ← 回收
                └──────────────┘  ← FREE[slot] 喂给下一轮同 slot 的 wait FREE

  值(gen) 与 槽(slot) 解耦：slot = gen % 2 固定循环，gen 由计数器单调递增。
```

### 6.5 与 §5 的同构，以及几处细节

- 结构上，本节是 §5 的「数据时钟」泛化：把 `i` 换成 `x_counter`、`slot` 换成 `gen % N`。
- 若消费者 lane == 生产者 lane 且是隐式顺序 pipe（S/V/M），则不需要 ready/free 通道，
  仅维护 `acquired`（`CollectGenerationChannels` 中的跳过分支）。
- 若存在回收通道但 flag 需要跨核复用，会在 protocol.end 加一条 S fence，
  阻止消费者在生产者 drain 之前重新 seed（`consumer.join`）。
- 没有回收通道、或生产者自己也读时，改用 `pipe_barrier` 兜底；
  跨核场景要求 Cube 作为中转，否则报错「write_counter does not support AIV0/AIV1」。

> 对应测试：`test_sync_protocols.py` 用一个小型模型穷举所有 pipe 交错，
> 校验「值、平衡、无死锁」；并核对 ready/free 各自 set/wait 的次数。
> `test_flag_allocation.py::test_distinct_counter_channels_keep_exclusive_rings_and_spill`
> 与 `test_write_clock_cross_core_flags_cover_every_consumer_bank` 覆盖跨核与溢出。

---

## 7. Flag 分配与复用（`FlagAllocator`）

分配器在硬件上限内给同步点分配物理 id，并复用生命周期不重叠的 flag：

- **同核**：上限 `kIntraCoreFlagLimit = 8`，域名为 `"core:HardEvent"`。
- **跨核**：上限 `kCrossCoreFlagLimit = 16`（每个 AIV bank 16 槽），域名为 `"Cube-Vector-X"`。
- 两个同步点能否复用同一组 flag，由 `CanReuse` 判定（同一 flag 类别 + 生命周期 `HasEdge` 证明一方严格在前）。

```python
# 5 条串行协议（每条 MTE2->V->MTE3 用到 3 组 flag）在 8 个槽内 -> 无需复用
# 9 条串行协议 -> 超过 8，触发复用（或 RewriteFlagToBuf 溢出）
```

> 对应测试：`test_flag_allocation.py::test_serial_protocols_fit_hardware_flag_budget`
> （`count=5` 适配、`count=9` 需要复用），以及 `test_cross_core_rings_fit_available_slots`。

---

## 8. 守卫剔除 / activation（同步只在 hazard 轮触发）

当生产/消费者的访问都带条件时，InsertSync 会用 guard 蕴含证明，
把同步**缩小到真正会发生冲突的轮次**，其它轮不插（或插在 guard 内）。

```python
for i in T.serial(8):
    if i % 2 == 0:
        T.copy(A[i, :], x)      # 生产者：只在偶数轮写
    if i % 2 == 0:
        T.copy(x, C[i, :])      # 消费者：只在偶数轮读
```

插入的 `set_flag` / `wait_flag` 会被包裹在 `i % 2 == 0` 的 guard 内，
即只有偶数轮才真正做 set/wait。

> 代码依据：`SyncInsertionSite.guard` / `activation` 的区分，以及
> `OptimizeSyncPoints` 中 `CoveringEligibility` / `ProvablyInactive` 的 hazard 判定。
> 对应测试：`test_sync_guard_culling.py`、`test_activation_covering.py`、
> `test_sync_protocols.py::test_guarded_read_history_and_sync_placement`。

---

## 9. AIV0 ↔ AIV1 直接同步被拒绝

Ascend AutoSchedule **不支持** AIV0 与 AIV1 之间直接同步
（既不让 flag 跨这两个子核，也不生成 AIC 中转）。这类依赖会 `LOG(FATAL)`：

```python
# AIV0 写 ub0，AIV1 读 ub0（split_subcores=True，无 Cube 顺序在其中）
```

报错信息：

```
Ascend AutoSchedule does not support synchronization between AIV0 and AIV1
```

> 代码依据：`SyncAnalyzer::ValidateSyncPoints`。
> 对应测试：`test_cross_core_flags.py::test_cross_aiv_memory_dependencies_require_supported_ordering`
> 等——只有在 Cube 作为中转（真实数据依赖）时才是合法的（`test_only_actual_cube_dependencies_can_order_aiv_accesses`）。

---

## 10. Overflow / 稀疏压缩：`RewriteFlagToBuf` 的配合

当同核 flag id 超过 8、或跨核环跨满 16 槽时，InsertSync 只负责**打标签**，
真正的「压缩稀疏越界 id / 溢出到 `get_buf`/`rls_buf`」由后续的
`RewriteFlagToBuf` Pass 完成。

> 对应测试：`test_flag_allocation.py::test_distinct_counter_channels_keep_exclusive_rings_and_spill`
> （两个排它 8 版本 release 环 → 溢出一环），以及
> `test_sparse_out_of_range_flags_are_compacted_without_spill`
> （越界稀疏 id 被改写为 `{0,1,2,3,4,5,6}`，无需溢出）。

---

## 11. 决策速查表

| 依赖边场景 | 判定函数 | 插入结果 |
| --- | --- | --- |
| 同 core + 同 pipe + S/V/M | `HasImplicitSamePipeOrder` | 无 |
| 同 core + 同 pipe + MTE/FIX | — | `pipe_barrier("PIPE_x")`（消费者开头） |
| 同 core + 跨 pipe | `UsesFlag` | `set_flag(event)` + `wait_flag(event)` |
| Cube ↔ AIV | `IsCrossCore` | `cross_core_set_flag(4,pipe)` + `cross_core_wait_flag(4,pipe)` |
| 跨迭代 RAW / WAR / WAW | `edge.distance ≥ 1` | 版本化 flag（`i % versions + base`）+ 首尾 seed/drain |
| write_counter 多缓冲区 | `info.write_counter` | READY / FREE 通道 + `acquired` 状态机 |
| hazard 可证明恒假 | `ProvablyInactive` | 该同步被剔除 |
| AIV0 ↔ AIV1 | `ValidateSyncPoints` | `LOG(FATAL)` 拒绝 |

---

## 12. 主流程回顾

`InsertKernelSync`（`insert_sync.cc` 末尾）按以下顺序编排：

1. `DecodeScheduledTIR` 解码，挂 `SyncInsertionExtraInfo`（暂存要插入的语句）。
2. 确定 core 集合（`vector_mask` + `kCoreCube`，或按 subcore 拆分）。
3. `SyncSiteRegistry::Collect` 收集所有同步插入点。
4. `SyncAnalyzer::Analyze` 做依赖分析、闭包饱和、同步点优化、flag 版本与复用配对。
5. `ValidateSyncPoints` 拒绝非法 AIV0↔AIV1。
6. `CollectExplicitCrossCoreFlags` 收集用户显式占用的 flag。
7. `FlagAllocator::Plan` 分配 / 复用物理 id。
8. `SyncMaterializer::Run` 物化同步语句 + `InsertBoundaryFlags` 生成环的首尾。
9. `EncodeScheduledTIR` 编码返回。

流程示意：

```
              scheduled TIR
                    │
                    ▼
   ①  DecodeScheduledTIR         解码，挂 SyncInsertionExtraInfo
                    │
                    ▼
   ②  确定 core 集合             vector_mask + kCoreCube / split subcores
                    │
                    ▼
   ③  SyncSiteRegistry::Collect  收集同步插入点（site）
                    │
                    ▼
   ④  SyncAnalyzer::Analyze      依赖分析·闭包·优化·flag 复用配对
                    │
                    ▼
   ⑤  ValidateSyncPoints         拒绝 AIV0↔AIV1
                    │
                    ▼
   ⑥  CollectExplicitCrossCoreFlags   收集用户显式占用
                    │
                    ▼
   ⑦  FlagAllocator::Plan        分配 / 复用物理 flag id
                    │
                    ▼
   ⑧  SyncMaterializer::Run      物化 set/wait/barrier + 环 seed/drain
                    │
                    ▼
   ⑨  EncodeScheduledTIR         编码
                    │
                    ▼
        带同步的 scheduled TIR
```

理解这套管线的关键是三层抽象：**物理位置（site）→ 纪元（domain/round）→ 顺序证明（闭包 + activation 蕴含）**，
三层共同保证在乱序流水线硬件上同步的**正确性**与**最小化**。