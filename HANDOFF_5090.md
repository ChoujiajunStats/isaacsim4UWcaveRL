# 5090 接手说明 — 2026-09-09

## 先读结论

代码仓库：<https://github.com/ChoujiajunStats/isaacsim4UWcaveRL>，**私有**，分支 `master`。
训练/诊断代码提交：`894660377a275935907863071b251311f765376a`。
4060 上本轮训练、诊断和评估已经结束，没有留下正在训练的任务。

**没有已验收的导航 policy。** PPO 原续训、FlashSAC 原续训和本轮 PPO 修复模型，
标准确定性完整路线评估均为 **0/90**。不要把日志中的 `PASS`、队列 `complete`、
课程升级或训练成功率当作策略验收通过，也不要因为修复模型更新就把它当成最佳模型。

接手目标仍是：一个共享的视觉导航策略，在不同洞穴中从指定入口安全到达指定出口。
当前 policy 输出机体系 xyz 速度、yaw 角速度和两个灯光动作，不输出全局路径/路标图。
Actor 输入为 1553 维：低分辨率双目 RGBD、IMU、压力深度、上一动作，以及相对出口向量/距离。
出口是已知导航目标，目前使用 ground-truth localization；这还不是未知出口探索、VIO 或实机验证。
地图、路线、洞穴 ID 不进入 actor。PPO 是 CNN+GRU；FlashSAC 是原生前馈视觉 MLP，不能当成纯算法公平对比。

另一台机器已有 Isaac Sim 和 OceanSim，**不要覆盖它们**。本仓库当前使用 Isaac Lab、
项目自建水动力和 Isaac RTX 相机；尚未接上 OceanSim。先复现这条链路，后续接 OceanSim
属于独立集成工作，不能把它混入当前结果比较。

## 已完成的实验

| 实验 | 新增 transitions | 完整成功 | 碰撞 | 越界 |
| --- | ---: | ---: | ---: | ---: |
| PPO 607 → 932 | 2,002,944 | 0/90 | 60/90 | 30/90 |
| FlashSAC 4096 → 24960 vector steps | 2,002,944 | 0/90 | 16/90 | 74/90 |
| PPO 932 weights-only 修复 → 95 | 589,824 | 0/90 | 60/90 | 30/90 |

完整评估：48 workers，每洞 30 回合，真实入口，300 s，固定均衡配额，确定性动作。
SPL 均为 0。通过门槛是各洞 success ≥80%、collision ≤15%、SPL ≥0.5。
`out_of_bounds` 包括距离参考路线大于 2.25 m，不等同于实际碰撞；不要悄悄放宽阈值来制造提升。

修复配方：保留 PPO 权重/归一化器、重置 optimizer/课程；探索 std 重置 0.4；
学习率 1e-4 fixed，entropy 0.001；50% 出口附近复习，20 回合/80%/×1.25 的较慢升级。
训练约 579 s，最终 frontier 为 easy/medium/hard = 6.25/6.25/5 m，仍未学稳。
8 m 复测：确定性 0/18，随机 2/18；完整 easy/medium/hard 平均走了 18.70/5.19/5.95 m。
详细结果在 [NAVIGATION_REPAIR.md](docs/NAVIGATION_REPAIR.md)。

冻结模型诊断的有用线索：

- PPO 932 的 8 m、保存 frontier、真实入口三种起点，三种动作方式都失败。
  不能仅归因于评估时关掉探索噪声。
- FlashSAC 原生确定性动作 `tanh(mu)` 在 8 m 和保存 frontier 都是 0/18。
  原生随机动作两组各 1/18；训练历史约 31.4% 成功不是最终模型在固定起点的成功率。
- 实验性 `E[tanh(mu + sigma*noise)]` 在 easy 保存 frontier 上是 6/6，其他两洞仍失败；
  easy 的 frontier 已是全参考路线长度。8 m 起点只有 easy 2/6。这只是小样本线索，
  **尚未用真实入口、30 回合/洞确认，也未替换默认推理规则**。

## 数据迁移：只 git clone 不够

以下文件已在 4060 本机生成，**未上传 GitHub，也未传到 5090**：

```text
/home/kenton/Desktop/IsaacSim4uwRL/outputs/handoff_5090/
  2026-09-09_nav_4060.tar.gz
  2026-09-09_nav_4060.tar.gz.sha256
  2026-09-09_nav_4060.tar.gz.manifest.json
```

压缩包 955,442,189 bytes（约 911 MiB），83 个数据文件，SHA-256：

```text
74bcfdf2a6f06bac8c84184903d5bfb02f6dc75675751f227c9a7309bf906219
```

请通过你自己的内网/SCP/移动存储复制这三个文件。不能在没有 5090 地址的情况下声称已传输。
包内有 PPO 932 和修复 95、FlashSAC 24960 的原生完整 checkpoint（含约 1.54 GiB replay）、
参数、评估、诊断、队列状态及 189 MiB 原始洞穴树。每个文件已重新从压缩包读取并核对 SHA-256。
不包含环境、GitHub/SSH 凭证、其他下载目录或依赖安装；不要复制本机登录凭证。
洞穴压缩包没有提供许可证，保持用户两台机器间的私有迁移，不公开发布。

在 5090 先 clone，再按注释复制文件，之后校验和解包：

```bash
git clone git@github.com:ChoujiajunStats/isaacsim4UWcaveRL.git
cd isaacsim4UWcaveRL
git switch master
mkdir -p outputs/handoff_5090
# 此处暂停，把三个迁移文件复制到当前仓库的 outputs/handoff_5090/，再继续。
(cd outputs/handoff_5090 && sha256sum -c 2026-09-09_nav_4060.tar.gz.sha256)
tar --keep-old-files -xzf outputs/handoff_5090/2026-09-09_nav_4060.tar.gz
export ISAAC_UNDERWATER_ASSET_ROOT="$PWD/transfer/assets"
```

使用有此私有仓库访问权的 5090 GitHub 身份。`--keep-old-files` 遇到已有数据会报错，
不要用强制覆盖解决；先检查冲突或用新 clone。`logs/`、`outputs/`、`transfer/` 不进 Git。
不要复用 4060 生成的绝对路径 USD 缓存；原始 OBJ/纹理已在包内，应在 5090 重新转换。

## 环境检查与首次复现

4060 实际验证版本：Python 3.11.16、Isaac Sim 5.1.0、Isaac Lab v2.3.2、
torch 2.7.0+cu128、RSL-RL 3.1.2、gymnasium 1.2.1、tensordict 0.14.1。
这是来源机记录，**不是要求卸载/降级 5090 已部署的环境**。先核对目标机版本；
不兼容时使用隔离的项目环境，保留 OceanSim 部署。5090 的版本、驱动和实际性能尚未验证。

本仓库队列脚本约定 Python 在 `.venv/bin/python`，训练入口还需要 `.deps/IsaacLab` 源码。
可复用已验证的项目环境/匹配的 Isaac Lab 源码；不要盲目运行 `setup_isaaclab.sh`，它会安装指定栈。
确认依赖齐备后，把项目安装到选定环境：

```bash
./.venv/bin/python -m pip install --no-deps -e ./source/isaac_underwater
./scripts/check_installation.sh
```

若缺少依赖源码，以下操作只下载源码，不安装/升级现有模拟器或 torch。
目标目录须不存在；已有源码先核对版本，不覆盖：

```bash
mkdir -p .deps
git clone --branch v2.3.2 --depth 1 --filter=blob:none --sparse \
  https://github.com/isaac-sim/IsaacLab.git .deps/IsaacLab
git -C .deps/IsaacLab sparse-checkout set source scripts apps tools
git clone --filter=blob:none --sparse https://github.com/Holiday-Robot/FlashSAC.git .deps/FlashSAC
git -C .deps/FlashSAC sparse-checkout set flash_rl
git -C .deps/FlashSAC checkout 87edc9061150ae9e962dd84e6544e27a1554b3ab
```

不要对 FlashSAC 执行 `uv sync`；桥接保留现有 torch，用 typing-only shim 避免无必要的 JAX 依赖。
先过 CPU 合约，再跑单个模拟器检查：

```bash
./.venv/bin/python scripts/test_cave_dataset.py
./.venv/bin/python -m unittest discover -s scripts -p 'test_navigation*.py'
./.venv/bin/python scripts/test_bounded_actions.py
./.venv/bin/python scripts/test_training_supervisor.py
./.venv/bin/python scripts/test_flash_sac.py
./.venv/bin/python scripts/convert_cave_asset.py \
  --config worlds/caves_difficulty_v01.yaml --profile train_all --headless --device cuda:0
./.venv/bin/python scripts/smoke_cave_navigation.py --num_envs 6 --headless --device cuda:0
./.venv/bin/python scripts/smoke_batched_visual_observation.py \
  --headless --device cuda:0 --output outputs/handoff_batched_equivalence.json
```

来源机 43 个 CPU 单测、视觉 policy 合约、44 个资产校验和合约已过；目标机仍须重跑。
`batched_visual_observation_enabled` 默认开启；仅加速 nominal/ground-truth 路径，
先前实际 GPU 测过 49 帧含自动 reset，policy/critic 与原路径零容差一致。

## 模型定位与有限续训命令

下列命令均在 repo 根目录、上述 asset-root 已导出的 shell 中运行：

```bash
PPO_BASE_RUN=2026-09-09_01-30-35_dual4060_20260909_013015_ppo
PPO_REPAIR_RUN=2026-09-09_12-56-59_exit_rehearsal_20260909_125652
FLASH_CHECKPOINT=logs/flash_sac/underwater_cave_multinav/2026-09-09_02-04-09_dual4060_20260909_013015_flash/step_0024960
```

先验证修复模型的标准完整评估，预期是复现失败，不是部署：

```bash
./.venv/bin/python scripts/evaluate_ppo.py \
  --task Isaac-Underwater-Cave-Navigation-v0 \
  --checkpoint "logs/rsl_rl/underwater_cave_multinav/$PPO_REPAIR_RUN/model_95.pt" \
  --cave_dataset_profile train_all --num_envs 48 --episodes_per_scene 30 \
  --episode_length_s 300 --seed 42 --headless --device cuda:0 \
  --output outputs/5090_ppo_repair_full.json
./.venv/bin/python scripts/check_navigation_metrics.py \
  outputs/5090_ppo_repair_full.json --expected-profile train_all
```

最后的 checker 非零退出是预期的策略验收失败；区分它与模拟器/程序崩溃。
新 GPU 不承诺 bit-identical 轨迹；大幅差异要先查配置、资产和软件栈。

如只测试修复 checkpoint 的 optimizer/课程续训，先做 **2 updates**：

```bash
./.venv/bin/python scripts/train.py \
  --task Isaac-Underwater-Cave-Navigation-v0 --cave_dataset_profile train_all \
  --navigation_curriculum --num_envs 96 --max_iterations 2 \
  --resume --load_run "$PPO_REPAIR_RUN" --checkpoint model_95.pt \
  --run_name 5090_repair_resume_smoke --seed 42 --headless --enable_cameras --device cuda:0 \
  agent.navigation_weights_only=False agent.navigation_reset_noise_std=0.0 \
  agent.algorithm.learning_rate=0.0001 agent.algorithm.schedule=fixed agent.algorithm.entropy_coef=0.001 \
  env.navigation_curriculum_window=20 env.navigation_curriculum_success_threshold=0.8 \
  env.navigation_curriculum_growth=1.25 env.navigation_rehearsal_probability=0.5 \
  env.navigation_rehearsal_min_distance_m=2.0 env.navigation_rehearsal_max_distance_m=12.0
```

这些参数是严格恢复所需，不是推荐继续砸大预算。不要把原修复启动时的
`navigation_weights_only=True` 再带进正常续训，否则会再次清掉 optimizer/课程。
PPO 932 用旧课程默认值（10 回合、70%、×1.5、无 rehearsal），不要把两套恢复配置混用。

FlashSAC 的有限恢复 smoke：

```bash
./.venv/bin/python scripts/train_flash_sac.py \
  --checkpoint "$FLASH_CHECKPOINT" --num_envs 96 --train_steps 128 \
  --run_name 5090_flash_resume_smoke --save_replay --eval_episodes_per_scene 0
```

改变 FlashSAC workers/batch/replay 必须显式 `--resize_resume`；buffer 可扩容，不可悄悄缩小。
适配器会按 ring 的真实顺序恢复 replay，重置新 workers 的局部 reward-return 状态；
这是 optimizer/replay/课程恢复、模拟器新回合，不是逐 bit 续接。PPO 也不保存正在进行的物理回合。
FlashSAC `--train_steps` 是额外 vector steps，transitions = steps × workers；不要和 PPO updates 混淆。

## 下一位 agent 优先做什么

1. 完成文件迁移、依赖核对、资产转换、smoke，再确认入口/出口/碰撞条件一致。
2. 优先确认 FlashSAC bounded-mean 的 easy 信号。保持 checkpoint 冻结，真实入口、
   每洞 30 个 workers 各一回合，对比标准确定性和 bounded-mean：

   ```bash
   ./.venv/bin/python scripts/diagnose_navigation_actions.py --algorithm flash \
     --checkpoint "$FLASH_CHECKPOINT" --num_envs 90 --scopes full \
     --modes deterministic bounded_mean --seed 42 --output outputs/5090_flash_action_full.json
   ```

   这是显式的推理规则实验，报告不能冒充标准 SAC 验收；还要追加独立种子。
3. 分析短程冷启动、终点附近控制和训练分布问题。当前“课程遗忘/观测别名/动作限幅”
   是候选原因，不是已证实的唯一根因；本轮 rehearsal + 低噪声配方未解决。
   可考虑更完整的路线起点覆盖、速度/记忆可观测性检查，或单独标注的 teacher warm-start 实验。
4. 5090 扩容先短测 96/192/384 等整除 3 的 workers，记录 transitions/s、显存、
   主机 available RAM 和温度，不以 GPU 利用率最高选配置。高面数碰撞与 CPU/内存也可能限制吞吐。
   现有 4060 队列默认 VRAM guard 是 7650 MiB；在 5090 上按实测总显存重新设置
   `scripts/supervise_training.py --max_gpu_memory_mb ...`，保留余量，不照搬硬编码上限。
   一次只跑一个重型模拟器；不要为了训练结束其他用户进程。
5. 几何泛化要另做 leave-one-cave-out 或增加新场景。当前只有三洞、全部参与训练，
   不能宣称多数未知洞穴泛化。不要在这一轮同时改算法、观察接口、OceanSim 和验收定义。

本机 push 使用了经验证的 GitHub SSH 443 替代通道（22 端口连接被关闭），未禁用 host-key 校验。
若 5090 常规 SSH 能用，不需要继承这个网络绕行设置。
