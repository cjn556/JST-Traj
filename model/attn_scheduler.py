# training/attn_scheduler.py
from __future__ import annotations
import os, csv, math
import torch
from torch_utils import training_stats

class AttnScheduler:
    """
    按 kimg 进度对所有 SparseSelfAttention2d 进行：
      - gate_scale 暖启：前 warmup_kimg 内从 0 -> 1
      - top-k 分阶段调度：none -> L/2 -> L/4（默认策略，可配置比例）
    并做简单的统计日志：平均 gate_scale、平均相对topk/L、窗口ws直方信息等。
    """
    def __init__(
        self,
        total_kimg: int,
        warmup_kimg: int = 100,           # 前100 kimg 线性从 0→1
        stage_ratio: tuple[float,float,float] = (0.2, 0.5, 0.3),  # 三阶段占比之和=1
        topk_mode: str = 'none-half-quarter',  # 预设: none -> L/2 -> L/4
        log_every: int = 10,              # 每多少 tick 记录一次统计
        csv_path: str | None = None       # 可选写入CSV
    ):
        self.total_kimg = total_kimg
        self.warmup_kimg = warmup_kimg
        a, b, c = stage_ratio
        s = a + b + c
        self.stage_ratio = (a/s, b/s, c/s)
        self.topk_mode = topk_mode
        self.log_every = log_every
        self.csv_path = csv_path
        self._csv_inited = False

    def _gate_scale(self, cur_kimg: float) -> float:
        if self.warmup_kimg <= 0:
            return 1.0
        return max(0.0, min(1.0, cur_kimg / float(self.warmup_kimg)))

    def _target_topk(self, L: int, progress: float) -> int | None:
        """
        progress ∈ [0,1] 相对总进度
        none-half-quarter：前 a 使用 full(None)，中 b 使用 L/2，后 c 使用 L/4
        """
        a, b, c = self.stage_ratio
        if progress <= a:
            return None   # 全量 softmax
        elif progress <= (a + b):
            return max(1, L // 2)
        else:
            return max(1, L // 4)

    @torch.no_grad()
    def step(self, G: torch.nn.Module, D: torch.nn.Module, cur_kimg: float, tick: int, run_dir: str | None = None):
        progress = min(1.0, max(0.0, cur_kimg / float(self.total_kimg)))
        gate = self._gate_scale(cur_kimg)

        # 扫描并设置
        modules = []
        for net in [G, D]:
            if net is None: continue
            for m in net.modules():
                clsn = m.__class__.__name__
                if clsn == 'SparseSelfAttention2d':
                    # 设置 gate_scale
                    m.set_gate_scale(gate)
                    # 计算窗口 token 数 L = ws*ws
                    ws = int(getattr(m, 'window_size', 8))
                    L = ws * ws
                    # topk
                    tk = self._target_topk(L, progress)
                    m.set_topk(tk)
                    modules.append((m, ws, L, tk))

        # 统计与日志
        if tick % self.log_every == 0 and len(modules) > 0:
            avg_rel_topk = sum([(0.0 if (tk is None) else float(tk)/float(L)) for _,_,L,tk in modules]) / len(modules)
            avg_ws = sum([ws for _,ws,_,_ in modules]) / len(modules)
            training_stats.report('attn/gate_scale', gate)
            training_stats.report('attn/rel_topk', avg_rel_topk)
            training_stats.report('attn/avg_ws', avg_ws)
            training_stats.report('attn/num_layers', len(modules))

            # 可选CSV
            if run_dir and self.csv_path:
                path = os.path.join(run_dir, self.csv_path)
                if not self._csv_inited:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'w', newline='') as f:
                        w = csv.writer(f)
                        w.writerow(['tick', 'cur_kimg', 'gate_scale', 'avg_rel_topk', 'avg_ws', 'num_layers'])
                    self._csv_inited = True
                with open(path, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([tick, cur_kimg, gate, avg_rel_topk, avg_ws, len(modules)])
