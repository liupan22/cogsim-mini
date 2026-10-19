"""迷你 Merlin：大规模仿真任务的编排与执行。

LLNL 用 Merlin（maestro DAG + celery/redis 队列）在 Sierra 上调度上亿次 HYDRA 仿真。
本模块是它的单机等比缩小版，保留三个关键性质：

  1. **任务化**  每个仿真是一个 (index, design, perturb) 任务，可独立执行；
  2. **并行**    通过进程池并行消费队列（Windows 下用 spawn，参数必须可 pickle）；
  3. **可追溯**  每次执行写 JSONL 账本 (ledger)，记录每个任务的 index（真实 Merlin
                 用它做断点续跑；这里因每次都按固定 seed 重跑以保证可复现，账本
                 只作为溯源记录）。

真实环境里把 `_run_one` 换成调用 HYDRA / PLUTO / FLASH 的命令行即可，
上层代码一行都不用改。
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .simulator import IMG, N_IMG, N_SCALAR, simulate


# ------------------------------------------------------------ 任务执行单元
def _run_one(task):
    """执行单个仿真任务。必须是模块级函数（Windows spawn 要求可 pickle）。"""
    idx, design, perturb, mode, seed = task
    rng = np.random.default_rng(seed)
    scalars, images = simulate(design, perturb, rng, mode=mode)
    return idx, scalars, images


def _load_ledger(path):
    """读取已完成任务的索引集合。"""
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["idx"])
    return done


# ------------------------------------------------------------ 编排入口
def run_ensemble(designs: np.ndarray,
                 perturbs: np.ndarray,
                 out_dir: str,
                 name: str = "ensemble",
                 mode: str = "sim",
                 seed0: int = 1000,
                 n_workers: int | None = None,
                 verbose: bool = True) -> dict:
    """批量执行仿真并落盘为 .npz，返回数据集路径字典。"""
    os.makedirs(out_dir, exist_ok=True)
    ledger_path = os.path.join(out_dir, f"{name}.ledger.jsonl")
    n = len(designs)

    done = _load_ledger(ledger_path)
    tasks = [(i, np.asarray(designs[i], dtype=np.float64),
              np.asarray(perturbs[i] if perturbs is not None else np.zeros(4)),
              mode, seed0 + i)
             for i in range(n) if i not in done]

    t0 = time.time()
    nw = min(n_workers or max(1, (os.cpu_count() or 2) - 1), 8)
    all_tasks = [(i, np.asarray(designs[i], dtype=np.float64),
                  np.asarray(perturbs[i] if perturbs is not None else np.zeros(4),
                             dtype=np.float64),
                  mode, seed0 + i)
                 for i in range(n)]
    if verbose:
        print(f"[workflow] {name}: {n} 个任务，已完成 {len(done)}，"
              f"待执行 {n - len(done)}，并行度 {nw}")

    results = [None] * n
    pending = all_tasks
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for idx, sc, im in ex.map(_run_one, pending, chunksize=32):
            results[idx] = (sc, im)

    # 账本只记录"本批次新增完成"的任务（已存在的条目不重复写）
    if pending:
        with open(ledger_path, "a", encoding="utf-8") as lf:
            for t in pending:
                lf.write(json.dumps({"idx": int(t[0])}) + "\n")

    scalars = np.stack([r[0] for r in results]).astype(np.float32)
    images = np.stack([r[1] for r in results]).astype(np.float32)
    assert scalars.shape == (n, N_SCALAR)
    assert images.shape == (n, N_IMG, IMG, IMG)

    npz = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(npz,
                        inputs=np.hstack([np.asarray(designs, dtype=np.float64),
                                          np.asarray(perturbs if perturbs is not None
                                                     else np.zeros((n, 4)), dtype=np.float64)]),
                        scalars=scalars, images=images)
    if verbose:
        print(f"[workflow] {name}: 完成，用时 {time.time()-t0:.1f}s -> {npz}")
    return {"inputs": npz, "ledger": ledger_path, "n": n}
