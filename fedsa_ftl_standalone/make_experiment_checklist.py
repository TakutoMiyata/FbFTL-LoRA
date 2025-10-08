#!/usr/bin/env python3
import os
import re
import pandas as pd

# 検索対象のルートディレクトリ
config_root = "configs/experiment_configs_bit"

# 正規表現パターン
pattern = re.compile(
    r"(?P<model>bit_s_r(50|101)x1)_alpha(?P<alpha>[0-9.]+)_f(?P<f>[0-9.]+)(?:_r(?P<r>[0-9]+))?(?:_eps(?P<eps>[0-9]+))?_(?P<method>fedavg|fedsa_lora|fedsa_lora_dp)\.yaml"
)

rows = []
for root, dirs, files in os.walk(config_root):
    for fname in files:
        if fname.endswith(".yaml"):
            m = pattern.match(fname)
            if not m:
                print(f"[!] Skip unmatched: {fname}")
                continue
            data = m.groupdict()
            method = data["method"]
            model = data["model"]
            alpha = float(data["alpha"])
            f = float(data["f"])
            r = int(data["r"]) if data["r"] else None
            eps = int(data["eps"]) if data["eps"] else None
            filepath = os.path.join(root, fname)
            rows.append({
                "Method": method,
                "Model": model,
                "α": alpha,
                "f": f,
                "r": r,
                "ε": eps,
                "Config Filename": fname,
                "Config Path": filepath,
                "実行済み": "☐",  # チェックボックス代わり
                "精度記録メモ": ""
            })

# DataFrame 作成
df = pd.DataFrame(rows)
df.insert(0, "No.", range(1, len(df)+1))
df = df.sort_values(by=["Method", "Model", "α", "f", "r", "ε"]).reset_index(drop=True)

# Excel 出力
output_path = "experiment_checklist.xlsx"
df.to_excel(output_path, index=False)

print(f"[+] Excel checklist saved to: {output_path}")