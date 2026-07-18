# move_playingcard_away Seed Collection — 问题记录与调试指南

**日期**：2026-07-06  
**目标**：为 move_playingcard_away 生成 PPO train seeds（success_seeds），写入 `train_seeds.json`

---

## 背景

`train_seeds.json` 中没有 `move_playingcard_away` 的条目。其他任务（beat_block_hammer、place_a2b_left 等）各有 ~1000 个 success seeds。

move_playingcard_away 的 PPO 配置（`moe_sft20k_move_playingcard_away_ppo100.yaml`）使用 `seeds_path: null`，env 会自动随机取 seed [10000, 200000)，因此 **PPO 可以不依赖预生成 seeds 直接运行**。Seed collection 是"锦上添花"，不是 PPO 先决条件。

---

## 方案：利用 episode_log_path 收集 success seeds

`robotwin_env.py` 支持 `episode_log_path` 配置，会把每个 episode 的 `{seed, success, instruction, ...}` 写入 JSONL 文件：

```python
# robotwin_env.py L105-116, L191-213
def _init_episode_log(self):
    episode_log_path = self.cfg.get("episode_log_path", None)
    if episode_log_path is None:
        return
    path = Path(episode_log_path)
    self._episode_log_path = path.with_name(f"{path.stem}.worker-{self.seed_offset}{suffix}")
    self._episode_log_path.parent.mkdir(parents=True, exist_ok=True)
    self._episode_log_path.write_text("")
```

输出格式（每行一条 JSON）：
```json
{"task": "move_playingcard_away", "worker": 0, "env_index": 3, "seed": 142857, "instruction": "...", "success": true, "success_step": 213, "total_steps": 400}
```

已有成功案例：`robotwin_adjust_bottle_ppo_openpi_pi05_eval.yaml` 在 **eval env** 段使用了该特性，路径格式为：
```yaml
episode_log_path: ${runner.logger.log_path}/robotwin_eval_adjust_bottle_episodes.jsonl
```

---

## 当前问题：CollectiveManager 崩溃

### 现象

4 次提交（Jobs 10201738、10201804、10201828、10201872），4 个不同节点，全部在启动约 8 分钟后崩溃，错误链：

```
(ActorGroup(rank=N)) ray.exceptions.ActorDiedError: The actor died unexpectedly
    class_name: CollectiveManager
    The actor is dead because it was killed by ray.kill
(RolloutGroup(rank=N)) Broadcast failed ... Connection closed by peer
Exiting main process due to a failure upon worker execution.
```

### 崩溃节点

| Job | 节点 | 结果 |
|-----|------|------|
| 10201738 | SH-IDCA1404-10-140-54-46 | ❌ Gloo broadcast 崩溃 |
| 10201804 | SH-IDCA1404-10-140-54-51 | ❌ Unsupported object type |
| 10201828 | SH-IDCA1404-10-140-54-1  | ❌ Gloo broadcast 崩溃 |
| 10201872 | SH-IDCA1404-10-140-54-1  | ❌ ActorDiedError |

**结论：节点无关，是配置问题。**

### 与正常运行的 PPO 任务的差异

| 参数 | seed collection config | 正常 PPO（如 place_a2b_left）|
|------|------------------------|------------------------------|
| `episode_log_path` | ✅ 有（train env） | ❌ 无 |
| `val_check_interval` | 999 | 50 |
| `save_interval` | 999 | 50 |
| `max_steps` | 8 | 500 |
| `seeds_path`（train） | null | train_seeds.json |

### 可能原因（待排查）

1. **`episode_log_path` 路径解析问题**：Hydra 插值 `${runner.logger.log_path}` 在 train env 段能否正确解析？参考成功案例（adjust_bottle）将路径改为绝对路径或移到 eval env 段。
2. **`val_check_interval: 999` 的副作用**：runner 的初始化逻辑是否依赖 val_check_interval 在合理范围内？
3. **ActorGroup 在极短 `max_steps`（8步）下的初始化时序问题**：runner 可能在 actor 全部 ready 前就触发了某个流程。

---

## 调试建议（新终端）

### 快速验证：去掉 episode_log_path，跑通 1 step

```yaml
# 修改 moe_sft20k_collect_seeds_move_playingcard_away.yaml：
# 1. 删除 episode_log_path 行
# 2. max_steps: 1
# 3. val_check_interval: 1
# 4. save_interval: 999
```

如果去掉 `episode_log_path` 后能跑通，则问题在 episode_log_path 配置。

### 修复方向 A：改用绝对路径

```yaml
env:
  train:
    episode_log_path: "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/results/seed_collection/episodes.jsonl"
```

### 修复方向 B：参考 adjust_bottle 成功案例

adjust_bottle 的 `episode_log_path` 在 **eval env** 段，不在 train env 段。
将 `episode_log_path` 移到 eval 段（需要 `only_eval: True`），用 eval 数据收集 success seeds。

### 修复方向 C：在正常 PPO 配置上追加 episode_log_path

不写新配置，在已知可以跑通的 `moe_sft20k_move_playingcard_away_ppo100.yaml` 基础上，通过 Hydra override 传入：

```bash
# 在 run_embodiment.sh 之后追加 override
python examples/embodiment/train_embodied_agent.py \
    --config-name moe_sft20k_move_playingcard_away_ppo100 \
    runner.max_steps=8 \
    runner.val_check_interval=1 \
    runner.save_interval=999 \
    "env.train.episode_log_path=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/results/seed_collection/episodes.jsonl"
```

---

## 相关文件路径

| 文件 | 路径 |
|------|------|
| Seed collection config | `examples/embodiment/config/moe_sft20k_collect_seeds_move_playingcard_away.yaml` |
| Seed collection sbatch | `examples/embodiment/slurm/moe_sft20k_collect_seeds_move_playingcard_away.sbatch` |
| PPO config（可直接提交）| `examples/embodiment/config/moe_sft20k_move_playingcard_away_ppo100.yaml` |
| PPO sbatch | `examples/embodiment/slurm/moe_sft20k_move_playingcard_away_ppo100.sbatch` |
| train_seeds.json | `rlinf/envs/robotwin/seeds/train_seeds.json` |
| 成功案例 config | `examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05_eval.yaml` |
| Slurm logs | `slurm_logs/moe_sft20k_collect_seeds_move_playingcard_away-*.out` |

---

## Slurm 集群使用

### 基本命令

```bash
# 查看自己的任务（必须用 muyao）
squeue -u muyao

# 提交任务
sbatch examples/embodiment/slurm/<name>.sbatch

# 取消任务
scancel <JOBID>

# 查看任务 log（实时）
tail -f slurm_logs/<name>-<JOBID>.out

# 查看任务运行节点
squeue -u muyao -o "%.10i %.20j %.8T %.10M %.20R"
```

### 提交 PPO 的标准流程

```bash
cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

# 提交
sbatch examples/embodiment/slurm/moe_sft20k_<task>_ppo100.sbatch

# 确认分配到节点
squeue -u muyao

# 10-15 分钟后检查 log
tail -50 slurm_logs/moe_sft20k_<task>_ppo100-<JOBID>.out

# 检查 TensorBoard 有无 Global Step
ls -lt logs/$(ls -t logs/ | grep <task> | head -1)/tensorboard/
```

### 崩溃判断与处理

| 现象 | 可能原因 | 处理 |
|------|---------|------|
| `Unsupported object type` / `Gloo broadcast failed` | 节点 collective 通信故障或 OOM | 加节点到 `--exclude`，重提 |
| `Gloo broadcast timeout（10800s）` | 节点网络故障 | 加节点到 `--exclude`，重提 |
| `Global Step` 停止更新，但 TB mtime 仍在变 | Ray logging actor 死亡，训练继续 | 正常，不干预 |
| `Global Step` 停止，TB mtime 也停（>30min）| 真正卡住 | scancel + 重提 |
| 训练指标（policy_loss 等）全为 0.0 | `filter_rewards: True` + `rewards_lower_bound` 过大 | 改 `filter_rewards: False`，重提 |

### 排除已知坏节点

所有 PPO sbatch 的 `--exclude` 行末尾追加新坏节点：

```bash
# 已知坏节点（截至 2026-07-06）
SH-IDCA1404-10-140-54-46  # Gloo 崩溃
SH-IDCA1404-10-140-54-51  # Unsupported object type
# + 历史坏节点见各 sbatch 文件
```

### TensorBoard 查看

```bash
# 在集群上启动（指定端口避免冲突）
tensorboard --logdir logs/<log_dir>/tensorboard --port 16006 --host 0.0.0.0

# 本地 SSH 转发
ssh -L 16006:localhost:16006 <cluster>
# 浏览器访问 http://localhost:16006
```

### 常用路径

```
REPO_PATH:    /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
ROBOTWIN:     /mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf
SFT-20k 权重: /mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/
              pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/
Seeds 目录:   rlinf/envs/robotwin/seeds/
PPO configs:  examples/embodiment/config/moe_sft20k_<task>_ppo100.yaml
PPO sbatch:   examples/embodiment/slurm/moe_sft20k_<task>_ppo100.sbatch
Slurm logs:   slurm_logs/
Train logs:   logs/
```
