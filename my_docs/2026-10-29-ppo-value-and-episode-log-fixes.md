# PPO 与轨迹记录修复（2026-10-29）

正式仓库：`/home/tanhe/sjy/RLinf-robotwin`。

独立验证副本、补丁和部署备份：`/data2/sjy/experiments/ppo_valuefix_20261029/`。

## 实际修复

1. **空指令日志**：`RoboTwinEnv` 现在从 `raw_obs['instruction']` 缓存真实文本；这与模型收到的 `task_descriptions` 来自同一个 observation。缓存与 manifest 的 `current_instructions` 分离，避免训练 reset 被错误固定为上一条指令。JSONL 增加 `instruction_source=observation`。若仿真实际返回空白或非字符串指令，在构造模型输入前报错并给出环境索引和 seed。
2. **重复 episode 日志**：同一环境从 reset 到下一次 reset 只写首次 done。局部 reset 只清除对应环境的去重标记。每条记录增加 `episode_id`，并保留首次终止时的真实步数。标准 `step` 与 `chunk_step` 均写日志。
3. **恢复/重新实例化时日志被清空**：创建日志改为追加保留，新增 `session_id` 区分不同进程会话。episode 的唯一键是 session、worker、env_index、episode_id，不能只用 seed，因为同一 seed 可以重复采样。
4. **rollout 与训练的 value 不一致**：非 joint 训练模式使用相同 `denoise_inds` 对应的 value，不再将 5 个步骤平均后与单一步骤比较。joint 模式保留去噪步骤平均；评测模式仍支持 `denoise_inds=-1`。
5. **explained variance 统计失效**：新增可相加的 FP64 sufficient statistics，跨微批次和 GPU 汇总后计算 EV、MAE、MSE、value/return 均值与标准差、有效样本数。单独记录 `ev_defined`；只有整个汇总样本不足或 returns 常量时 EV 才明确未定义，不用 0 伪装收敛。
6. **增加 rollout critic 诊断**：在完整 rollout 有效样本上计算旧 value 对 GAE returns 的指标。这个目标含 bootstrap，不是独立测试集上的 Monte Carlo 回报，不能单凭它宣布价值函数收敛。
7. **value clipping 比例错误**：按 clamp 之前的 value 变化判断超限，并只统计有效样本。
8. **可复现评测随机数**：可通过 `rollout.eval_seed` 指定策略噪声种子，按 worker rank 派生随机数；退出评测时恢复训练的 Python、NumPy、Torch RNG。只有显式设置此项才改变旧评测行为。仍应使用相同 env 数、batch 布局、manifest 和模型设置做配对；物理仿真并不因此保证位级一致。
9. **更新前一致性检查**：新增可选 `actor.verify_rollout_consistency=true`。在第一次 optimizer update 前检查实际 rollout 与重算的 value/logprob，并记录差异；超出容差时明确失败。

## 验证

使用 CPU（`CUDA_VISIBLE_DEVICES=`）运行以下测试，不启动 Ray、仿真或训练：

```bash
cd /home/tanhe/sjy/RLinf-robotwin
CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD .venv/bin/python -m pytest -q \
  tests/unit_tests/test_ppo_valuefix.py \
  tests/unit_tests/test_robotwin_multitask_env.py
```

测试包含实际模型方法的受控去噪 value 重算、joint/非 joint 模式、EV 跨分片聚合及空样本、value clipping、RNG 重放与恢复、单/多任务指令、局部 reset 去重、标准 step、空输入拒绝和日志追加。

另外检查修改文件的 Ruff lint/format，以及 actor/rollout 模块导入。完整 GPU rollout 和 PPO smoke test 尚未运行：用户已要求停止当前和后续训练，测试只使用 CPU。

## 历史结果与实验状态

- 旧权重、原始 JSONL、TensorBoard 文件均保留。旧空指令行无法可靠还原当时真实随机指令，不回填猜测文本。
- 旧训练日志重复写行不能直接当作独立 episode；先前审计通过 400 步终结记录核对为 3200 条 episode，不能用 11520 行计算成功率。
- 修复不改变旧 checkpoint 的内容，也不证明现有 PPO50 的成功率已经改善。
- `beat_block_hammer` 50-step 权重保留；`place_phone_stand` 在完成 8 step 后按用户要求停止，未形成 checkpoint。
- 原队列和 3 小时自动巡检均保持暂停，没有启动复测或新实验。
- 本次没有修改原有学习率、update_epoch 或 critic_warmup_steps。预热、降低更新幅度等属于后续需要对照验证的实验参数。

最终验证：正式仓库 41 项 CPU 测试通过；Ruff 检查通过；actor/rollout/metrics 导入通过。
