# CIFAR-10 実験ガイド

## 概要

CIFAR-10データセット（10クラス分類）での連合学習実験が可能です。

## CIFAR-10とCIFAR-100の違い

| 項目 | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| クラス数 | 10 | 100 |
| 画像数 | 60,000 | 60,000 |
| 訓練データ | 50,000 | 50,000 |
| テストデータ | 10,000 | 10,000 |
| クラス例 | airplane, car, bird, cat... | 細粒度カテゴリ |
| 難易度 | 簡単（精度80-95%） | 難しい（精度60-80%） |

## CIFAR-10用設定ファイル

以下の設定ファイルが作成されています：

### Non-IID設定
- `configs/experiment_configs_non_iid/non-IID-FedAvg-CIFAR10.yaml`
  - FedAvgベースライン（LoRAなし）
  
- `configs/experiment_configs_non_iid/non-IID-FedSA-LoRA-CIFAR10.yaml`
  - FedSA-LoRA（提案手法）

### IID設定
- `configs/experiment_configs_iid/IID-FedSA-LoRA-CIFAR10.yaml`
  - IID分布でのFedSA-LoRA

## 実行方法

### 1. FedAvgベースライン（CIFAR-10、Non-IID）

```bash
python quickstart_resnet.py --config configs/experiment_configs_non_iid/non-IID-FedAvg-CIFAR10.yaml
```

### 2. FedSA-LoRA（CIFAR-10、Non-IID）

```bash
python quickstart_resnet.py --config configs/experiment_configs_non_iid/non-IID-FedSA-LoRA-CIFAR10.yaml
```

### 3. FedSA-LoRA（CIFAR-10、IID）

```bash
python quickstart_resnet.py --config configs/experiment_configs_iid/IID-FedSA-LoRA-CIFAR10.yaml
```

### 4. ラウンド数の変更

```bash
# 200ラウンド実行
python quickstart_resnet.py --config configs/experiment_configs_non_iid/non-IID-FedSA-LoRA-CIFAR10.yaml --round 200

# 50ラウンド実行（クイックテスト）
python quickstart_resnet.py --config configs/experiment_configs_non_iid/non-IID-FedSA-LoRA-CIFAR10.yaml --round 50
```

## データの自動ダウンロード

初回実行時、CIFAR-10データセットが自動的にダウンロードされます：

```
./data/cifar-10-batches-py/
```

サイズ: 約170MB

## 期待される精度

### CIFAR-10（簡単）
- **中央集権学習**: 90-95%
- **FedAvg (IID)**: 85-90%
- **FedAvg (Non-IID)**: 75-85%
- **FedSA-LoRA (Non-IID)**: 80-88%

### CIFAR-100（難しい）
- **中央集権学習**: 70-80%
- **FedAvg (IID)**: 65-75%
- **FedAvg (Non-IID)**: 55-65%
- **FedSA-LoRA (Non-IID)**: 60-70%

## 設定の変更ポイント

### データセット切り替え

YAMLファイルの以下の部分を変更：

```yaml
data:
  dataset_name: cifar10  # cifar10 または cifar100
  
model:
  num_classes: 10  # cifar10: 10, cifar100: 100
```

### Non-IID の強さ調整

```yaml
data:
  alpha: 0.1   # 小さいほど強いNon-IID（0.1-10.0）
  # 0.1: 強いNon-IID（各クライアントは少数クラスのみ）
  # 1.0: 中程度のNon-IID
  # 10.0: 弱いNon-IID
  # 1000: IID相当
```

## CIFAR-10 vs CIFAR-100 比較実験

両方のデータセットで同じ設定で実験して比較：

```bash
# CIFAR-10での実験
python quickstart_resnet.py --config configs/experiment_configs_non_iid/non-IID-FedSA-LoRA-CIFAR10.yaml

# CIFAR-100での実験
python quickstart_resnet.py --config configs/experiment_configs_non_iid/non-IID-FedSA-LoRA.yaml
```

## トラブルシューティング

### データが見つからないエラー

```bash
# CIFAR-10を手動ダウンロード
python -c "import torchvision; torchvision.datasets.CIFAR10(root='./data', train=True, download=True)"
```

### メモリ不足

```yaml
# バッチサイズを削減
data:
  batch_size: 32  # デフォルト: 64
```

### 学習が収束しない

```yaml
# 学習率を調整
training:
  lr: 0.0005  # デフォルト: 0.001（小さくする）
  
# またはラウンド数を増やす
federated:
  num_rounds: 150  # デフォルト: 100
```

## 実験の進捗確認

```bash
# GPU使用状況
nvidia-smi

# ログ確認
tail -f experiments/quickstart_resnet_iid_cifar10/TIMESTAMP/training.log

# 結果確認
cat experiments/quickstart_resnet_iid_cifar10/TIMESTAMP/final_results_*.json
```

## クラスリスト

### CIFAR-10 クラス（10個）
1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

### CIFAR-100 クラス（100個）
20のスーパークラスに分類された100の細粒度クラス

## まとめ

✅ **CIFAR-10の利点**:
- 学習が速い（収束が早い）
- 精度が高い
- デバッグしやすい

✅ **CIFAR-100の利点**:
- より現実的なタスク
- 手法の性能差が明確に出る
- 研究で広く使われる

**推奨**: 最初はCIFAR-10でシステムをテストし、その後CIFAR-100で本格評価。
