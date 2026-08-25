# OpenPI 八月份之后训练的数据集使用情况

统计范围：`/mnt/public/gaojiahuan/openpi/wandb/run-202608*`，即 2026-08-01 之后在本目录留下本地 W&B 记录的训练。数据集清单以每次运行保存的 `config.yaml` 或 `output.log` 为准，而不是以当前 `src/openpi/training/config.py` 回推。统计时间截至 2026-08-17。

## 历次训练记录

| 启动时间 | W&B run ID | config | exp_name / 变体 | 数据集组 | 日志末次 step |
|---|---|---|---|---:|---:|
| 08-02 17:06 | `1zvl5xdm` | `bagging_sku_sm2sm` | `bagging_sku_0802_dm10dh50df50po20_meta` | A | 55,400 |
| 08-02 20:48 | `tis01bp9` | `bagging_sku_sm2sm` | `bagging_sku_0802_filtered_dm10dh50df50po20_meta` | A | 90,000 |
| 08-04 13:51 | `y3tjzclc` | `bagging_sku_sm2sm` | `bagging_sku_0804_dm10dh50df50po20_meta` | B0 | 未进入 step |
| 08-04 15:16 | `rbwux7wi` | `bagging_sku_sm2sm` | `bagging_sku_0804_dm10dh50df50po20_meta` | B | 110,000 |
| 08-04 16:15 | `54fya2qx` | `bagging_sku_2_sm2sm` | `bagging_sku_0804v2_dm10dh50df50po20_meta` | C | 100,300 |
| 08-04 17:55 | `u6lyqo8i` | `bagging_sku_3_sm2sm` | `bagging_sku_0804v3_dm10dh50df50po20_meta` | D | 99,200 |
| 08-07 23:57 | `oy4ywp5e` | `bagging_sku_delta_sm2sm` | `bagging_sku_0807_delta_dm10dh50df50po20_meta` | E | 未进入 step |
| 08-08 01:07 | `4wqepgb3` | `bagging_sku_sm2sm` | `bagging_sku_0807_dm10dh50df50po20_meta` | E | 85,800 |
| 08-09 13:14 | `20b85h7y` | `bagging_sku_delta_sm2sm` | `bagging_sku_0807_delta_dm10dh50df50po20_meta` | E | 27,200 |
| 08-10 10:47 | `ngmy4fgq` | `bagging_sku_delta_sm2sm` | `bagging_sku_0810_delta_dm10dh50df50po20_meta` | E | 未进入 step |
| 08-10 13:15 | `whefcdtw` | `bagging_sku_delta_sm2sm` | `bagging_sku_0810_delta_dm10dh50df50po20_meta` | E | 122,200 |
| 08-11 21:03 | `83kot2j2` | `bagging_sku_0811_sm2sm` | `bagging_sku_0811_dm10dh50df50po20_meta` | F | 97,500 |
| 08-13 11:02 | `w998qgtx` | `bagging_sku_0812_sm2sm` | `bagging_sku_0812_dm10dh50df50po20_meta` | G | 120,400 |
| 08-13 18:35 | `ovv73ajc` | `bagging_sku_0811_pi05_sm2sm` | `bagging_sku_0811_dm10dh50df50po20_pi05_meta` | F | 400 |
| 08-13 19:03 | `39bfwiq1` | `bagging_sku_0811_pi05_sm2sm` | `bagging_sku_0811_dm10dh50df50po20_pi05_meta` | F | 未进入 step |
| 08-13 20:47 | `osylwk1m` | `bagging_sku_0811_pi05_sm2sm` | `bagging_sku_0811_dm10dh50df50po20_pi05_meta` | F | 103,300 |
| 08-14 15:10 | `y3vhjebj` | `bagging_sku_0814_sm2sm` | 由运行参数及 asset 确认 | H | 89,900 |
| 08-17 11:09 | `oy1990z3` | `bagging_sku_0817_sm2sm` | masked | I | 4,700 |
| 08-17 13:22 | `9jwsln6h` | `bagging_sku_0817_nomask_sm2sm` | no-mask | I | 未记录 step |

“日志末次 step”是本地日志中最后出现的训练 step，只表示日志记录到的位置，不等同于训练正常完成。

## 数据集组汇总

| 组 | 使用次数 | 数据集条目数 | 日志可核实 episode 总数 | 数据集变化/用途 |
|---:|---:|---:|---:|---|
| A | 2 | 29 | 日志未完整记录 | 早期混合：SKU 数据 + 旧 4SKU 数据 |
| B0 | 1 | 26 | 未加载成功 | 08-04 首次配置；存在漏逗号，见异常说明 |
| B | 1 | 27 | 日志未完整记录 | 大部分旧 4SKU 数据替换为 `*_filter_static` |
| C | 1 | 29 | 日志未完整记录 | B 基础上补回 `0702b`、`0703` |
| D | 1 | 24 | 日志未完整记录 | 精简部分旧 4SKU 数据，并启用 position offset |
| E | 5 | 12 | 2,069（含重复项） | 只保留 SKU 数据；normal/delta 共用同一数据集组 |
| F | 4 | 14 | 2,223（含重复项） | E 的多批数据改用 trimmed，并加入 `0805_dagger`、`0810`；pi0/pi0.5 共用 |
| G | 1 | 14 | 2,223（含重复项） | 与 F 同批次，但使用未 trimmed 的原数据 |
| H | 1 | 15 | 2,163 | 在 G 上加入 `0813`、`0811_1`，并去掉重复的 `0715` |
| I | 2 | 10 | 1,156 | 进一步精简旧数据，加入 `0814`；masked/no-mask 对照 |

## 各组数据集明细

| 组 | 数据集 |
|---:|---|
| A | `bagging_sku_zzc_0730`, `bagging_sku_zzc_0724`, `bagging_sku_fty_0717`, `bagging_sku_fty_0715`, `bagging_sku_fty_0702b`, `bagging_sku_fty_0703`, `bagging_sku_zyx_0625`, `bagging_sku_fty_0706+0707`, `bagging_sku_fty_0708`, `bagging_4sku_ny_0501_filtered`, `bagging_4sku_xpc_0430`, `bagging_4sku_xpc_0501`, `bagging_4sku_xpc_0502`, `bagging_4sku_xpc_0505`, `bagging_4sku_xpc_0506`, `bagging_4sku_xpc_0507`, `bagging_4sku_zyx_0421`, `bagging_4sku_zyx_0423`, `bagging_4sku_zyx_0425`, `bagging_4sku_zyx_0509`, `bagging_4sku_zyx_0512`, `bagging_4sku_zyx_0515`, `bagging_4sku_zyx_0518`, `bagging_4sku_zyx_0519`, `bagging_4sku_zyx_0522`, `bagging_4sku_zyx_0526_dagger`, `bagging_4sku_zyx_0528_dagger`, `bagging_sku_fty_0713_dagger_filtered`, `bagging_sku_fty_0714` |
| B | `bagging_4sku_zyx_0528_dagger_filter_static`, `bagging_4sku_ny_0501_filtered`, `bagging_4sku_xpc_0430_filter_static`, `bagging_4sku_xpc_0501_filter_static`, `bagging_sku_fty_0706+0707`, `bagging_4sku_xpc_0502_filter_static`, `bagging_sku_fty_0708`, `bagging_4sku_xpc_0505_filter_static`, `bagging_sku_fty_0713_dagger_filtered`, `bagging_4sku_xpc_0506_filter_static`, `bagging_sku_fty_0714`, `bagging_4sku_xpc_0507_filter_static`, `bagging_sku_fty_0715`, `bagging_4sku_zyx_0421`, `bagging_sku_fty_0717`, `bagging_4sku_zyx_0423_filter_static`, `bagging_sku_zyx_0625`, `bagging_4sku_zyx_0425_filter_static`, `bagging_sku_zzc_0724`, `bagging_4sku_zyx_0509_filter_static`, `bagging_sku_zzc_0730`, `bagging_4sku_zyx_0512_dagger_filter_static`, `bagging_4sku_zyx_0515_filter_static`, `bagging_4sku_zyx_0518_filter_static`, `bagging_4sku_zyx_0519_filter_static`, `bagging_4sku_zyx_0522_filter_static`, `bagging_4sku_zyx_0526_dagger_filter_static` |
| C | B + `bagging_sku_fty_0702b`, `bagging_sku_fty_0703` |
| D | C 去掉 `bagging_4sku_xpc_0430_filter_static`, `bagging_4sku_zyx_0421`, `bagging_4sku_zyx_0423_filter_static`, `bagging_4sku_zyx_0425_filter_static`, `bagging_sku_zzc_0730` |
| E | `bagging_sku_fty_0702b`, `bagging_sku_fty_0703`, `bagging_sku_fty_0706+0707`, `bagging_sku_fty_0708`, `bagging_sku_fty_0715`（重复 2 次）, `bagging_sku_fty_0713_dagger_filtered`, `bagging_sku_fty_0714`, `bagging_sku_fty_0717`, `bagging_sku_zzc_0724`, `bagging_sku_zzc_0730`, `bagging_sku_zyx_0625` |
| F | `bagging_sku_fty_0702b_trimmed`, `bagging_sku_fty_0703_trimmed`, `bagging_sku_fty_0706+0707`, `bagging_sku_fty_0708_trimmed`, `bagging_sku_fty_0715_trimmed`（重复 2 次）, `bagging_sku_fty_0713_dagger_filtered`, `bagging_sku_fty_0714_trimmed`, `bagging_sku_fty_0717_trimmed`, `bagging_sku_zzc_0724_trimmed`, `bagging_sku_zzc_0730_trimmed`, `bagging_sku_zyx_0625`, `bagging_sku_zzc_0805_dagger`, `bagging_sku_zzc_0810` |
| G | F 中所有 `*_trimmed` 换成对应未 trimmed 数据；其余不变，`0715` 仍重复 2 次 |
| H | `bagging_sku_fty_0702b`, `bagging_sku_fty_0703`, `bagging_sku_fty_0706+0707`, `bagging_sku_fty_0708`, `bagging_sku_fty_0713_dagger_filtered`, `bagging_sku_fty_0714`, `bagging_sku_fty_0715`, `bagging_sku_fty_0717`, `bagging_sku_zzc_0724`, `bagging_sku_zzc_0730`, `bagging_sku_zyx_0625`, `bagging_sku_zzc_0805_dagger`, `bagging_sku_zzc_0810`, `bagging_sku_fty_0813`, `bagging_sku_zzc_0811_1` |
| I | `bagging_sku_fty_0706+0707`, `bagging_sku_fty_0708`, `bagging_sku_fty_0714`, `bagging_sku_fty_0715`, `bagging_sku_fty_0717`, `bagging_sku_zzc_0724`, `bagging_sku_zzc_0805_dagger`, `bagging_sku_fty_0814`, `bagging_sku_fty_0813`, `bagging_sku_zzc_0811_1` |

## 异常与注意事项

- 08-04 13:51 的 B0 配置把 `bagging_sku_fty_0713_dagger_filtered` 和 `bagging_4sku_xpc_0506_filter_static` 拼成了一个名称，日志未出现训练 step；随后 15:16 的 B 运行已修正。
- E、F、G 的 `repo_id` 都把 `0715` 数据集写了两遍。数据加载日志也确实加载了两遍，所以表中的 episode 总数按实际条目相加；去重后分别为 1,899、2,053、2,053 episodes。
- 08-17 两条记录创建时间很近，最后一条在统计时尚未写出完整 `config.yaml` 和训练 step；数据集清单来自其 `output.log`，config 名来自 `wandb-metadata.json`。
