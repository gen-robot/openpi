# RoboDojo 到 Pi0.5：数据、训练与推理链路

本文整理我们刚才讨论的几个问题：dev-openpi 为什么能训练 RoboDojo 数据、数据在每一层是什么格式、adapter 做了什么，以及训练和推理怎样保持一致。

文件位置和行号对应当前 `dev-openpi-xjy-robodojo` 版本。

## 1. 先看完整主线

```text
RoboDojo 原始数据
  ├─ PICO DAgger：JSON/JSONL + 三路 MP4
  └─ 官方数据：HDF5
        │
        ▼ 转换器
LeRobot 数据集
  单帧 state[14] + action[14] + 三路图像 + task
        │
        ▼ DataLoader
当前 observation + 未来 50 步 action chunk[50,14]
        │
        ▼ Repack + ARX X5 adapter
OpenPI 通用 policy 输入
        │
        ▼ Delta + Normalize + model transforms
Pi0.5 模型张量
  state[32] + actions[50,32] + 224×224 图像 + 文本 token
        │
        ▼
训练 → checkpoint → 推理 → RoboDojo action chunk[50,14]
```

每层解决的问题不同：

- **LeRobot**统一磁盘存储；
- **DataLoader**把单帧 action 组成未来动作序列；
- **Repack**整理字段名；
- **ARX X5 adapter**翻译机器人接口；
- **Delta/Normalize**处理动作语义和数值尺度；
- **model transforms**生成 Pi0.5 真正使用的张量。

## 2. 原始 state/action 是什么

当前 ARX X5 的 14 维顺序是：

```text
0～5    左臂 6 个关节
6      左夹爪
7～12  右臂 6 个关节
13     右夹爪
```

机械臂关节使用弧度。当前 RoboDojo X5 中，夹爪接近 `1` 表示打开，接近 `0` 表示闭合。

RoboDojo 的 `observation.state` 具体来自：

- 12 个机械臂维度：Isaac Sim 当前实际关节位置；
- 2 个夹爪维度：最近一次处理后的夹爪控制参考 `prev_control`。

对应代码在 RoboDojo：

- `env/robot_manager/robot_manager.py:172-186`：读取机械臂实际关节；
- `env/observation_manager/obs_manager.py:144-188`：组装 state，并用 `prev_control` 覆盖夹爪字段。

所以这里的 state 是一个明确的混合契约：**机械臂是实际状态，夹爪是控制参考**。

## 3. 原始数据怎样变成 LeRobot

### 3.1 DAgger 数据

转换器只选择已经 Accept、成功、系统有效且确实发生过接管的 episode。筛选函数是 [`is_selected`](../examples/robodojo/convert_dagger_to_lerobot.py#L204)。

DAgger 原始记录没有单独保存一列可直接训练的 expert action，因此当前转换规则是：

```text
observation.state[t] = state[t]
action[t]            = state[t+1]
image[t]             = 当前控制周期对应的视频帧
```

最后一帧没有 `state[t+1]`，所以丢掉。实现见 [`convert_episode`](../examples/robodojo/convert_dagger_to_lerobot.py#L326)。

### 3.2 官方 HDF5 数据

官方 HDF5 已经保存了 `state` 和 `action`。转换器会验证 `action[t] == state[t+1]`，然后原样写入源 `state[t]` 和 `action[t]`，同样丢掉最后一行。

关键函数：

- [`robot_vector`](../examples/robodojo/convert_hdf5_to_lerobot.py#L115)：按固定顺序拼出 14 维；
- [`inspect_episode`](../examples/robodojo/convert_hdf5_to_lerobot.py#L152)：检查 state/action、图像和 fps；
- [`convert_episode`](../examples/robodojo/convert_hdf5_to_lerobot.py#L234)：写入 LeRobot。

两种数据最终标签关系相同，但来源不同：**DAgger 用下一帧 state 构造标签；HDF5 保留并验证官方 action。**

### 3.3 LeRobot 中的一帧

转换后的目录主要是：

```text
<repo_id>/
├── data/       # Parquet：state、action 和索引
├── videos/     # 三路相机视频
└── meta/       # task、episode 和数据集信息
```

一个逻辑时间点是：

```python
{
    "observation.state": float32[14],
    "action": float32[14],
    "observation.images.cam_high": image,
    "observation.images.cam_left_wrist": image,
    "observation.images.cam_right_wrist": image,
    "task_index": int,
}
```

LeRobot 到这里统一了“怎么存、怎么按 episode 读取”，但没有决定 14 维的机器人含义、夹爪方向或 action 是否使用 delta；这些由后面的 DataConfig 和 adapter 负责。

如果配置中写入 `official_repo,dagger_repo`，[`create_torch_dataset`](../src/openpi/training/data_loader.py#L245) 会分别加载，再由 [`MultiDataset`](../src/openpi/training/data_loader.py#L123) 合并。当前采样比例近似由各 repo 的有效帧数决定，并非自动各占一半。

## 4. DataLoader 怎样组成 action chunk

LeRobot 磁盘上每行只有一个 `action[14]`。当前训练配置使用：

```python
Pi0Config(pi05=True, action_horizon=50)
```

因此 DataLoader 以当前帧 `t` 为起点读取：

```text
state[t]
三路 image[t]
task
action[t], action[t+1], ..., action[t+49]
```

得到：

```python
{
    "observation.state": float32[14],
    "action": float32[50,14],
    "observation.images.*": image,
    "prompt": str,
}
```

也就是说，`[50,14]` action chunk 是 DataLoader 根据时间索引临时取出的窗口，不是转换器提前存成一个大数组。

相关位置：

- [`create_torch_dataset`](../src/openpi/training/data_loader.py#L245)：根据 fps 和 `action_horizon` 建立时间查询；
- [`PromptFromLeRobotTask`](../src/openpi/transforms.py#L309)：把 `task_index` 还原成 prompt；
- [`config.py:783-793`](../src/openpi/training/config.py#L783)：当前 RoboDojo Pi0.5 训练配置。

## 5. Repack 和 ARX X5 adapter 分别做什么

### 5.1 Repack：只整理字段

[`RepackTransform`](../src/openpi/transforms.py#L79) 把 LeRobot 字段整理成：

```python
{
    "state": float32[14],
    "actions": float32[50,14],
    "images": {
        "cam_high": image,
        "cam_left_wrist": image,
        "cam_right_wrist": image,
    },
    "prompt": str,
}
```

具体映射在 [`config.py:414-431`](../src/openpi/training/config.py#L414)。Repack 不改变数值，也不做 delta、归一化或图像缩放。

### 5.2 ARX adapter：翻译机器人接口

[`RoboDojoArxX5Inputs.__call__`](../src/openpi/policies/robodojo_arx_x5_policy.py#L81) 负责：

- 检查 state/action 最后一维是 14；
- 将 CHW 或 HWC 图像统一成 HWC `uint8`；
- 检查三路相机并映射到 OpenPI 的相机槽位。

```text
cam_high        → base_0_rgb
cam_left_wrist  → left_wrist_0_rgb
cam_right_wrist → right_wrist_0_rgb
```

输出变成 OpenPI policy 接口：

```python
{
    "image": {三个 OpenPI 相机槽位},
    "image_mask": {三路均有效},
    "state": float32[14],
    "actions": float32[50,14],  # 只有训练时存在
    "prompt": str,
}
```

训练时有 `actions`，实时推理时没有；同一个 input adapter 通过 [`if "actions" in data`](../src/openpi/policies/robodojo_arx_x5_policy.py#L112) 兼容两种情况。

推理输出端的 [`RoboDojoArxX5Outputs`](../src/openpi/policies/robodojo_arx_x5_policy.py#L122) 把模型输出从 32 维截回 14 维。

## 6. Delta、norm stats 和模型变换

### 6.1 DeltaActions

当前 mask 的含义是：

```text
左右机械臂 12 维：使用 delta
左右夹爪 2 维：保持 absolute
```

计算方式是：

```text
机械臂：delta_action[k] = absolute_action[k] - current_state[t]
夹爪：保持 absolute_action[k]
```

50 个未来动作都相对同一个当前 state `t`，不是 `action[k] - action[k-1]`。配置在 [`config.py:439-445`](../src/openpi/training/config.py#L439)，计算函数是 [`DeltaActions`](../src/openpi/transforms.py#L203)。

### 6.2 Norm stats

统计顺序是：

```text
LeRobot → Repack → ARX adapter → DeltaActions → 统计 state/actions
```

代码在 [`scripts/compute_norm_stats.py:24-43`](../scripts/compute_norm_stats.py#L24)。所以 stats 对应的仍是：

```text
state   [14]
actions [50,14]：12 维机械臂 delta + 2 维绝对夹爪
```

当前 Pi0.5 使用 q01/q99 的 quantile normalization。本次用“官方 100 条 + DAgger 56 条”训练，因此应由这个数据组合共同计算新 stats；保存 checkpoint 时，stats 会进入：

```text
<checkpoint-step>/assets/<asset_id>/norm_stats.json
```

推理必须使用同一份 stats。

### 6.3 Model transforms

[`ModelTransformFactory`](../src/openpi/training/config.py#L109) 最后执行：

```text
图像           → 224×224
prompt         → 最多 200 个 token
state[14]      → 补零为 state[32]
actions[50,14] → 补零为 actions[50,32]
```

所以模型实际看到：

```text
images   [B,224,224,3]
state    [B,32]
actions  [B,50,32]
tokens   [B,200]
```

## 7. DataConfig、TrainConfig 和训练循环

### DataConfig：规定每条样本怎样加工

[`LeRobotRoboDojoArxX5DataConfig.create`](../src/openpi/training/config.py#L407) 提供 Repack、ARX adapter、delta、norm stats 和 model transforms；[`transform_dataset`](../src/openpi/training/data_loader.py#L407) 按顺序执行：

```text
Repack → ARX adapter → Delta → Normalize → model transforms
```

### TrainConfig：规定一次训练实验

[`TrainConfig`](../src/openpi/training/config.py#L641) 指定：

```text
模型和 action_horizon
DataConfig 和 LeRobot repo
初始 checkpoint
batch size、训练步数、seed
checkpoint 输出位置
```

训练入口 [`scripts/train.py:221`](../scripts/train.py#L221) 创建 DataLoader；[`train_step`](../scripts/train.py#L136) 最终执行：

```text
model.compute_loss(observation, actions)
→ 求梯度
→ optimizer 更新参数
```

因此 dev-openpi 原本已经提供模型和训练循环。我们为 RoboDojo 增加的主要内容是：

1. DAgger/HDF5 转换器；
2. ARX X5 input/output adapter；
3. ARX X5 DataConfig；
4. 具体 TrainConfig。

## 8. 推理时为什么也要 adapter

实时推理不经过 LeRobot，但输入仍是 RoboDojo 的 `state/images/prompt`。模型需要 OpenPI/Pi0.5 格式，所以推理链路是：

```text
RoboDojo 实时 observation
→ ARX X5 input adapter
→ Normalize
→ Pi0.5 model transforms
→ model.sample_actions
→ Unnormalize
→ delta 加回当前 state
→ ARX X5 output adapter
→ RoboDojo action chunk[50,14]
```

dev 仓库的 [`create_trained_policy`](../src/openpi/policies/policy_config.py#L16) 组装这条链路，[`Policy.infer`](../src/openpi/policies/policy.py#L67) 调用模型推理。

所谓训练和推理使用“相同 adapter”，指这些语义必须一致：

- 14 维顺序和夹爪约定；
- 相机映射；
- delta mask；
- norm stats；
- action horizon 和 model action dim。

它不表示推理还要创建 LeRobot 数据集；Repack 也是训练侧整理 LeRobot 字段所需的步骤。

## 9. checkpoint 放回 5090 的边界

服务器训练完成后，可以把完整 step 目录复制到 XPolicyLab 的 checkpoint 目录。但当前还不能只复制文件就认定适配完成：

- 服务器 dev 使用 `RoboDojoArxX5Inputs/Outputs`，不做 ALOHA 关节翻转和夹爪几何转换；
- 5090 当前 `XPolicyLab/Pi_05/deploy.yml` 仍选择 ALOHA DataConfig，其 `AlohaInputs` 默认会做这些变换。

对应位置：

- `/data/xiaojinyang/RoboDojo/XPolicyLab/policy/Pi_05/deploy.yml:21-23`；
- `/data/xiaojinyang/RoboDojo/XPolicyLab/policy/Pi_05/model.py:104-114`；
- `/data/xiaojinyang/RoboDojo/XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/aloha_policy.py:30-107`。

因此新 checkpoint 部署前，还要让 XPolicyLab 选择与服务器训练一致的 ARX X5 adapter/DataConfig，并从新 checkpoint 加载对应 norm stats。这个修改只解决 checkpoint 的正确消费方式，不需要重写 RoboDojo rollout。

## 10. 一句话记住每一层

```text
转换器：原始 RoboDojo → LeRobot
DataLoader：单 action → 50 步 action chunk
Repack：整理字段名
ARX adapter：RoboDojo 接口 → OpenPI 接口
Delta/Norm：动作语义和数值尺度
Model transforms：14 维数据 → Pi0.5 的 32 维张量
推理：按对应的逆变换输出 RoboDojo action chunk
```
