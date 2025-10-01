# ハイパーパラメータスイープガイド

## 概要

全てのハイパーパラメータの組み合わせで実験を自動的に実行するスクリプトです。

### 探索するハイパーパラメータ

- **data.alpha**: [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (11種類)
- **training.lr**: [0.001, 0.00001, 0.000001] (3種類)
- **lora.dropout**: [0.1, 0.2, 0.3] (3種類)

**合計**: 11 × 3 × 3 = **99通りの実験**

## 使い方

### 方法1: 逐次実行（推奨：メモリ制約がある場合）

```bash
python run_hyperparameter_sweep.py
```

**特徴**:
- 1つずつ順番に実験を実行
- GPU メモリの使用量が少ない
- 実行時間が長い（99実験 × 実験時間）
- 途中経過がリアルタイムで表示される

### 方法2: 並列実行（推奨：複数GPU利用可能な場合）

```bash
python run_hyperparameter_sweep_parallel.py
```

**特徴**:
- 複数の実験を並列実行
- デフォルトで2並列（`NUM_PARALLEL_JOBS`で調整可能）
- 実行時間が大幅に短縮（最大で並列数倍速）
- 複数GPUまたは大容量GPUメモリが必要

**並列数の調整**:

スクリプトを編集して`NUM_PARALLEL_JOBS`を変更：

```python
# 並列数を変更（GPUメモリに応じて調整）
NUM_PARALLEL_JOBS = 2  # 例: 2並列
NUM_PARALLEL_JOBS = 4  # 例: 4並列（大容量GPUの場合）
```

## 出力結果

### ディレクトリ構造

```
experiments/hyperparameter_sweep/
└── 20251001_123456/                    # タイムスタンプ
    ├── sweep_summary.json              # 全実験の概要
    ├── results_summary.csv             # 結果一覧（CSV形式）
    ├── alpha0.1_lr0.001_dropout0.1/   # 個別実験1
    │   ├── config.yaml
    │   ├── training.log
    │   └── final_results_*.json
    ├── alpha0.1_lr0.001_dropout0.2/   # 個別実験2
    │   └── ...
    └── ...
```

### 結果の確認

#### 1. JSON形式の概要

```bash
cat experiments/hyperparameter_sweep/TIMESTAMP/sweep_summary.json
```

内容:
- 全実験のステータス（成功/失敗）
- 各実験のハイパーパラメータ
- 実行時間
- 結果ファイルのパス

#### 2. CSV形式の結果一覧

```bash
cat experiments/hyperparameter_sweep/TIMESTAMP/results_summary.csv
```

列:
- `experiment_id`: 実験番号
- `alpha`: データ分布パラメータ
- `lr`: 学習率
- `dropout`: LoRAドロップアウト率
- `status`: 実験ステータス
- `duration_hours`: 実行時間（時間）
- `best_test_accuracy`: 最高テスト精度
- `final_avg_accuracy`: 最終平均精度

#### 3. トップ10の結果

スクリプト終了時に自動的に表示されます：

```
🏆 Top 10 Best Results:
  alpha    lr  dropout  best_test_accuracy
    5.0 0.001      0.2               75.32
    3.0 0.001      0.1               74.89
   ...
```

## 実行時の注意点

### 実行時間の見積もり

- **1実験あたりの時間**: 約1-2時間（100ラウンドの場合）
- **逐次実行**: 99実験 × 1.5時間 = 約150時間（6日）
- **2並列実行**: 99実験 ÷ 2 × 1.5時間 = 約75時間（3日）
- **4並列実行**: 99実験 ÷ 4 × 1.5時間 = 約38時間（1.5日）

### GPU メモリ要件

- **1実験あたり**: 約4-8GB（モデルとバッチサイズによる）
- **逐次実行**: 1GPU（4-8GB）で可能
- **2並列実行**: 2GPU または 1GPU（16GB以上）
- **4並列実行**: 4GPU または 2GPU（16GB以上）

### 中断と再開

#### 中断方法

```bash
Ctrl+C  # プログラムを中断
```

#### 再開方法

現在、自動再開機能はありません。以下の手順で手動再開：

1. `sweep_summary.json`から成功した実験を確認
2. スクリプトを編集して、未実行の組み合わせのみを設定
3. 再実行

## ハイパーパラメータのカスタマイズ

スクリプトを編集して探索範囲を変更できます：

```python
# run_hyperparameter_sweep.py を編集

# 探索範囲を変更
ALPHA_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]  # 5種類に削減
LR_VALUES = [0.001, 0.0001]                 # 2種類に削減
DROPOUT_VALUES = [0.1, 0.2]                 # 2種類に削減

# この場合、合計 5 × 2 × 2 = 20実験
```

## ベース設定の変更

別の設定ファイルを使用する場合：

```python
# スクリプトを編集
BASE_CONFIG = "configs/experiment_configs_iid/IID-FedSA-LoRA.yaml"
```

または、コマンドライン引数を追加する（将来的な拡張）。

## トラブルシューティング

### メモリ不足エラー

```
CUDA out of memory
```

**解決策**:
1. 並列数を減らす（`NUM_PARALLEL_JOBS`を1に設定）
2. バッチサイズを減らす（ベース設定の`data.batch_size`を32に削減）
3. より大きなGPUを使用

### 実験が失敗する

1. ログファイルを確認：
   ```bash
   tail -n 50 experiments/hyperparameter_sweep/TIMESTAMP/alphaX_lrY_dropoutZ/training.log
   ```

2. 個別に実験を実行してデバッグ：
   ```bash
   python quickstart_resnet.py --config experiments/hyperparameter_sweep/TIMESTAMP/alphaX_lrY_dropoutZ/config.yaml
   ```

### プロセスの監視

別のターミナルで進行状況を監視：

```bash
# GPU使用率の監視
watch -n 1 nvidia-smi

# 完了した実験数を確認
find experiments/hyperparameter_sweep/TIMESTAMP -name "final_results_*.json" | wc -l
```

## 結果の分析

### Pythonでの分析例

```python
import pandas as pd
import matplotlib.pyplot as plt

# 結果の読み込み
df = pd.read_csv('experiments/hyperparameter_sweep/TIMESTAMP/results_summary.csv')

# 成功した実験のみ
df_success = df[df['status'] == 'SUCCESS']

# 最良のハイパーパラメータを見つける
best_row = df_success.loc[df_success['best_test_accuracy'].idxmax()]
print(f"Best config: alpha={best_row['alpha']}, lr={best_row['lr']}, dropout={best_row['dropout']}")
print(f"Best accuracy: {best_row['best_test_accuracy']:.2f}%")

# ヒートマップの作成（alpha vs lr、dropout固定）
for dropout in df_success['dropout'].unique():
    subset = df_success[df_success['dropout'] == dropout]
    pivot = subset.pivot_table(values='best_test_accuracy', 
                               index='alpha', 
                               columns='lr')
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.values, cmap='viridis', aspect='auto')
    plt.colorbar(label='Best Test Accuracy (%)')
    plt.xlabel('Learning Rate')
    plt.ylabel('Alpha')
    plt.title(f'Accuracy Heatmap (Dropout={dropout})')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(f'heatmap_dropout_{dropout}.png')
    plt.close()
```

## 参考

- ベース設定: `configs/experiment_configs_non_iid/non-IID-FedSA-LoRA.yaml`
- メイン実行スクリプト: `quickstart_resnet.py`
- 結果プロット: `plot_results.py`（既存のスクリプト）
