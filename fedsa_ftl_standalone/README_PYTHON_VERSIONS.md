# Python バージョンと依存関係の互換性

## 概要

このプロジェクトには2つの conda 環境が必要です：

| 環境名 | Python | 用途 | スクリプト |
|--------|--------|------|-----------|
| `FedSA-LoRA-DP` | 3.11 | 標準的な実験 | `quickstart_resnet.py`, `quickstart_resnet_fedavg.py` など |
| `FedSA-LoRA-DP-py310` | 3.10 | TensorFlow Federated | `quickstart_bit_tff.py` |

## なぜ2つの環境が必要か？

**TensorFlow Federated 0.64.0 は Python 3.11 と互換性がありません。**
- Python 3.11 の `typing` モジュールの変更により、TFF がクラッシュします
- TFF を使用するスクリプトは Python 3.10 環境が必要です

## セットアップ

### オプション1: 自動セットアップ（推奨）

```bash
# Python 3.10 環境を自動作成
./setup_py310_env.sh
```

### オプション2: 手動セットアップ

```bash
# 1. Python 3.10 環境を作成
conda create -n FedSA-LoRA-DP-py310 python=3.10 -y

# 2. 環境をアクティブ化
conda activate FedSA-LoRA-DP-py310

# 3. 依存関係をインストール
pip install -r requirements.txt
```

## 使用方法

### TensorFlow Federated を使用する場合

```bash
conda activate FedSA-LoRA-DP-py310
python quickstart_bit_tff.py --config configs/experiment_configs_non_iid/bit_tff_cifar100.yaml
```

### 標準的な実験の場合

```bash
conda activate FedSA-LoRA-DP  # または FedSA-LoRA-DP-py310
python quickstart_resnet.py --config configs/experiment_configs_non_iid/non-IID-FedSA-LoRA.yaml
```

## 重要な依存関係のバージョン

### Python 3.10 環境 (FedSA-LoRA-DP-py310)

| パッケージ | バージョン | 理由 |
|-----------|-----------|------|
| Python | 3.10 | TensorFlow Federated の互換性 |
| transformers | 4.30.0 | PyTorch 2.1 との互換性 |
| opacus | 1.4.0 | PyTorch 2.0-2.1 との互換性 |
| tensorflow | 2.14.0 | TensorFlow Federated 0.64.0 との互換性 |
| tensorflow-federated | 0.64.0 | Python 3.10 と互換性がある最新版 |

### Python 3.11 環境 (FedSA-LoRA-DP)

| パッケージ | バージョン | 理由 |
|-----------|-----------|------|
| Python | 3.11 | 最新の安定版 |
| transformers | 4.30.0 | PyTorch 2.0 との互換性 |
| opacus | 1.4.0 | PyTorch 2.0 との互換性 |

## トラブルシューティング

### エラー: `AttributeError: module 'torch.nn' has no attribute 'RMSNorm'`

**原因**: Opacus のバージョンが新しすぎて、PyTorch 2.0-2.1 と互換性がありません。

**解決策**:
```bash
pip uninstall -y opacus
pip install opacus==1.4.0
```

### エラー: `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`

**原因**: Transformers のバージョンが新しすぎて、PyTorch 2.1 と互換性がありません。

**解決策**:
```bash
pip install transformers==4.30.0
```

### エラー: TensorFlow Federated が Python 3.11 でクラッシュ

**原因**: TensorFlow Federated 0.64.0 は Python 3.11 と互換性がありません。

**解決策**: Python 3.10 環境を使用してください：
```bash
conda activate FedSA-LoRA-DP-py310
```

## まとめ

- **TFF を使う**: `FedSA-LoRA-DP-py310` (Python 3.10) を使用
- **TFF を使わない**: どちらの環境でも動作しますが、`FedSA-LoRA-DP` (Python 3.11) を推奨
- 両方の環境で `quickstart_resnet.py` などの標準スクリプトは動作します
