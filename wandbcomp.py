import csv
import json
from typing import Any

import wandb


RUN_A = "striding-algo/openpi/trcdjp4r"
RUN_B = "striding-algo/openpi/4m0rlmp8"


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """将嵌套字典展开为 model.action_horizon 这类键。"""
    result = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).startswith("_"):
                continue

            new_key = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(value, new_key))
    else:
        result[prefix] = data

    return result


def normalize(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except Exception:
        return str(value)


api = wandb.Api()

run_a = api.run(RUN_A)
run_b = api.run(RUN_B)

config_a = flatten(dict(run_a.config))
config_b = flatten(dict(run_b.config))

all_keys = sorted(set(config_a) | set(config_b))

rows = []
for key in all_keys:
    in_a = key in config_a
    in_b = key in config_b

    value_a = normalize(config_a[key]) if in_a else "<不存在>"
    value_b = normalize(config_b[key]) if in_b else "<不存在>"

    if not in_a:
        status = "仅 4m0rlmp8"
    elif not in_b:
        status = "仅 7zjew0dh"
    elif value_a != value_b:
        status = "不同"
    else:
        status = "相同"

    rows.append(
        {
            "参数": key,
            "7zjew0dh": value_a,
            "4m0rlmp8": value_b,
            "状态": status,
        }
    )

different_rows = [row for row in rows if row["状态"] != "相同"]

print(f"7zjew0dh 参数数量: {len(config_a)}")
print(f"4m0rlmp8 参数数量: {len(config_b)}")
print(f"差异数量: {len(different_rows)}")
print()

for row in different_rows:
    print(f"[{row['状态']}] {row['参数']}")
    print(f"  7zjew0dh : {row['7zjew0dh']}")
    print(f"  4m0rlmp8 : {row['4m0rlmp8']}")
    print()

with open("wandb_config_diff.csv", "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["参数", "7zjew0dh", "4m0rlmp8", "状态"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("完整结果已保存到 wandb_config_diff.csv")
