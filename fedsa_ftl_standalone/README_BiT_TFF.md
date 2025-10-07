# BiT (Big Transfer) + TFF CIFAR-100 Non-IID Federated Learning

このドキュメントでは、TensorFlow Federatedの階層的LDA non-IIDデータセットを使用したBiTモデルの連合転移学習の実行方法を説明します。

## 概要

- **モデル**: BiT (Big Transfer) - S/M variants with ResNet-50/101
- **データセット**: TFF CIFAR-100（2段階階層的LDA non-IID分割）
- **実験設定**: 論文準拠
  - 200 rounds
  - 25 round毎に評価
  - 5または10 training clients
  - 各クライアント: ~100 train samples + ~100 test samples
  - **重要**: 訓練とテストで同じクライアントIDを使用（Non-IID一貫性）

## セットアップ

### 1. 依存関係のインストール

```bash
cd fedsa_ftl_standalone
pip install -r requirements.txt
```

主な追加依存関係:
- `tensorflow>=2.13.0`
- `tensorflow-federated>=0.60.0`
- `timm>=0.9.0` (BiTモデル用)

### 2. 環境変数の設定（オプション）

Slack通知を有効にする場合は`.env`ファイルを作成:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

## 使用方法

### 基本的な実行

```bash
python quickstart_bit_tff.py
```

デフォルト設定:
- Model: `bit_m_r50x1` (BiT-M ResNet-50×1)
- Training clients: 10
- Test clients: 30
- Rounds: 200
- Evaluation frequency: 25 rounds

### モデルバリアントの選択

BiTの4つのバリアントから選択可能:

```bash
# BiT-S ResNet-50×1 (ImageNet-21k pretrained)
python quickstart_bit_tff.py --model bit_s_r50x1

# BiT-M ResNet-50×1 (ImageNet-21k + ImageNet-1k fine-tuned) [推奨]
python quickstart_bit_tff.py --model bit_m_r50x1

# BiT-S ResNet-101×1
python quickstart_bit_tff.py --model bit_s_r101x1

# BiT-M ResNet-101×1
python quickstart_bit_tff.py --model bit_m_r101x1
```

### クライアント数の変更

論文では5または10クライアントを使用:

```bash
# 5 training clients
python quickstart_bit_tff.py --clients 5

# 10 training clients (デフォルト)
python quickstart_bit_tff.py --clients 10
```

### ラウンド数の調整

```bash
# 100 rounds (テスト用)
python quickstart_bit_tff.py --rounds 100

# 200 rounds (論文設定)
python quickstart_bit_tff.py --rounds 200
```

### カスタム設定ファイルの使用

```bash
python quickstart_bit_tff.py --config configs/experiment_configs_non_iid/bit_tff_cifar100.yaml
```

## 設定ファイルのカスタマイズ

`configs/experiment_configs_non_iid/bit_tff_cifar100.yaml`を編集:

### モデル設定

```yaml
model:
  model_name: bit_m_r50x1  # bit_s_r50x1, bit_m_r50x1, bit_s_r101x1, bit_m_r101x1
  num_classes: 100
  pretrained: true
  freeze_backbone: true
  lora:
    enabled: true
    r: 8  # LoRA rank
    alpha: 16
    dropout: 0.1
```

### 実験設定

```yaml
federated:
  num_rounds: 200  # 総ラウンド数
  num_clients: 10  # Training clients (5 or 10)
  client_fraction: 1.0  # 各ラウンドで選択するクライアント割合

evaluation:
  eval_freq: 25  # 評価間隔（論文では25）

data:
  num_test_clients: 30  # テストクライアント数（固定）
  batch_size: 64
```

### Differential Privacyの有効化

```yaml
federated:
  aggregation_method: fedsa_shareA_dp  # DP-FedSA

privacy:
  enable_privacy: true
  epsilon: 8.0
  delta: 1.0e-05
  max_grad_norm: 0.5
  noise_multiplier: 1.0
  target: lora_A  # LoRA A行列のみにDPを適用
```

## 実験結果

結果は以下のディレクトリに保存されます:

```
experiments/quickstart_bit_tff/
└── fedsa/  # または fedsa_shareA_dp
    └── 20250107_143022/  # タイムスタンプ
        ├── training_results_0107BiT.json
        ├── results_summary_0107BiT.csv
        ├── final_results_0107BiT.json
        └── checkpoint_round_25_0107BiT.json
        └── checkpoint_round_50_0107BiT.json
        └── ...
```

### 結果ファイルの内容

- `training_results_*.json`: 各ラウンドの詳細な訓練結果
- `results_summary_*.csv`: Pandas/matplotlibで可視化しやすいCSV形式
- `final_results_*.json`: 最終的な実験サマリー
- `checkpoint_round_*_*.json`: 定期的なチェックポイント

### CSVフォーマット

```csv
round,train_accuracy,test_accuracy,per_round_communication_mb,is_best,epsilon
1,25.4,23.1,45.2,False,
25,45.8,42.3,45.2,True,
50,58.3,55.7,45.2,True,
...
```

## TFFデータセットについて

### 階層的LDA Non-IID分割

TFF CIFAR-100は以下の特徴を持つnon-IIDデータセット:

1. **2段階階層構造**: CIFAR-100の20スーパークラスと100サブクラス
2. **LDA (Latent Dirichlet Allocation)**: クライアント間でクラス分布が不均一
3. **自動分割**: `tff.simulation.datasets.cifar100.load_data()`で自動取得

### データ統計

- Training clients: 500 (内5-10を使用)
- **各クライアント**: 訓練100サンプル + テスト100サンプル
- Classes: 100
- Input size: 224×224 (BiT用にリサイズ)
- **重要**: 訓練とテストで同じクライアントIDを使用することで、Non-IID特性を維持

## BiTモデルについて

### モデルバリアント

| Model | Pretrain Dataset | Parameters | 推奨用途 |
|-------|-----------------|-----------|---------|
| BiT-S-R50×1 | ImageNet-21k | 23.5M | 軽量・高速 |
| BiT-M-R50×1 | ImageNet-21k→1k | 23.5M | **推奨** |
| BiT-S-R101×1 | ImageNet-21k | 42.5M | 高精度 |
| BiT-M-R101×1 | ImageNet-21k→1k | 42.5M | 最高精度 |

### LoRA適用箇所

- **LoRA A**: 全クライアント共有（FedAvgで集約）
- **LoRA B**: 各クライアントローカル（パーソナライゼーション）
- **Classifier head**: 各クライアントローカル

## トラブルシューティング

### TensorFlow/TFFのインストールエラー

```bash
# CPU版（推奨）
pip install tensorflow-cpu tensorflow-federated

# GPU版（CUDA環境のみ）
pip install tensorflow tensorflow-federated
```

### メモリ不足エラー

batch_sizeを削減:

```yaml
data:
  batch_size: 32  # デフォルト: 64
```

### BiTモデルのダウンロードエラー

timmのキャッシュをクリア:

```bash
rm -rf ~/.cache/torch/hub/checkpoints/
```

## 実験例

### 論文準拠の実験

```bash
# BiT-M ResNet-50×1, 10 clients, 200 rounds
python quickstart_bit_tff.py \
  --model bit_m_r50x1 \
  --clients 10 \
  --rounds 200

# BiT-S ResNet-101×1, 5 clients, 200 rounds
python quickstart_bit_tff.py \
  --model bit_s_r101x1 \
  --clients 5 \
  --rounds 200
```

### DP-FedSA実験

設定ファイルを編集してDP有効化:

```yaml
federated:
  aggregation_method: fedsa_shareA_dp

privacy:
  enable_privacy: true
  epsilon: 8.0
```

実行:

```bash
python quickstart_bit_tff.py --config configs/experiment_configs_non_iid/bit_tff_cifar100_dp.yaml
```

## 参考資料

- [Big Transfer (BiT) Paper](https://arxiv.org/abs/1912.11370)
- [TensorFlow Federated Documentation](https://www.tensorflow.org/federated)
- [timm Library](https://github.com/huggingface/pytorch-image-models)
- [FedSA-LoRA Paper](https://arxiv.org/abs/2310.15384)

## ライセンス

このコードはMITライセンスの下で提供されています。
