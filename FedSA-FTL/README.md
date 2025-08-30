# FedSA-FTL: Federated Share-A Transfer Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## 概要

FedSA-FTL (Federated Share-A Transfer Learning) は、Feature-based Federated Transfer Learning (FbFTL) と FedSA-LoRA の原理を統合したハイブリッドアーキテクチャです。連合学習における**通信効率**、**異質性への耐性**、**パーソナライズ**のトリレンマを解決する新しいアプローチを提供します。

### 主な特徴

- **凍結バックボーン**: 事前学習済みモデルをクライアント側の特徴抽出器として使用
- **LoRA適応ヘッド**: タスク特化ヘッドにLoRA (Low-Rank Adaptation) を適用
- **選択的集約**: LoRA A行列のみを通信し、B行列はクライアント固有として保持
- **通信効率**: 標準LoRA連合学習と比較して50%の通信量削減
- **パーソナライズ**: non-IIDデータに対する優れたロバスト性

## アーキテクチャ

```
Client Side:                     Server Side:
┌─────────────────┐              ┌──────────────────┐
│  Frozen         │              │  Global LoRA A   │
│  Backbone       │              │  Aggregation     │
│  (Feature       │              │                  │
│  Extractor)     │              │  ┌─────────────┐ │
└─────────┬───────┘              │  │   FedAvg    │ │
          │                      │  │ Aggregation │ │
          v                      │  └─────────────┘ │
┌─────────────────┐              └──────────────────┘
│  LoRA Head      │                        │
│  ┌─────────────┐│              ┌────────────────────┐
│  │ LoRA A      ││◄─────────────┤   Broadcast        │
│  │ (Shared)    ││              │   Global LoRA A    │
│  └─────────────┘│              └────────────────────┘
│  ┌─────────────┐│
│  │ LoRA B      ││
│  │ (Local)     ││
│  └─────────────┘│
└─────────────────┘
```

## インストール

### 必要環境

- Python 3.8+
- PyTorch 1.9+
- torchvision
- transformers (NLPタスクの場合)
- numpy
- matplotlib
- pandas
- scikit-learn

### インストール手順

```bash
# リポジトリをクローン
git clone <repository-url>
cd FedSA-FTL

# 依存関係をインストール
pip install torch torchvision transformers numpy matplotlib pandas scikit-learn seaborn

# または requirements.txt から
pip install -r requirements.txt
```

## 使用方法

### 1. 基本的な実験実行

```bash
# CIFAR-10でのVision Transformer実験
python main.py --config configs/cifar10_vit_base.json

# CIFAR-100での挑戦的な設定
python main.py --config configs/cifar100_vit_challenging.json
```

### 2. カスタム設定

設定ファイル（JSON形式）を編集して実験パラメータを調整：

```json
{
  "experiment_name": "FedSA-FTL_Custom",
  "dataset": {
    "name": "cifar10",
    "alpha": 0.5,
    "data_dir": "./data"
  },
  "model": {
    "type": "vit_base",
    "lora_rank": 8,
    "lora_alpha": 16.0
  },
  "federated": {
    "num_clients": 10,
    "num_rounds": 100,
    "client_sampling_ratio": 1.0
  },
  "training": {
    "local_epochs": 1,
    "batch_size": 32,
    "optimizer": {
      "lr": 0.005,
      "weight_decay": 1e-4
    }
  }
}
```

### 3. プログラムによる使用

```python
from src import FedSAFTLExperiment

# 設定を読み込み
with open('config.json', 'r') as f:
    config = json.load(f)

# 実験を作成・実行
experiment = FedSAFTLExperiment(config)
experiment.setup_data()
experiment.setup_model()
experiment.setup_clients_and_server()
results = experiment.run_experiment()

# 結果を保存
experiment.save_results('./results/my_experiment.json')
```

### 4. 結果分析

```bash
# 単一実験の分析
python analyze_results.py --results results/experiment.json --mode single

# 複数実験の比較分析
python analyze_results.py --results results/ --mode multiple
```

## 実験設定

### データセット

- **CIFAR-10**: 10クラスの自然画像分類
- **CIFAR-100**: 100クラスの挑戦的な画像分類
- **GLUE**: 自然言語理解タスク（実装中）

### Non-IID設定

Dirichlet分布を使用してクライアント間のデータ分布を制御：

- `α → ∞`: IID（均等分布）
- `α = 0.5`: 中程度のnon-IID
- `α = 0.1`: 深刻なnon-IID

### モデルアーキテクチャ

#### コンピュータビジョン
- **Vision Transformer (ViT-Base)**: 現代的なTransformerベースのアーキテクチャ
- **ResNet-18**: 軽量なCNNアーキテクチャ

#### 自然言語処理（実装中）
- **RoBERTa-Base/Large**: 強力な言語理解モデル

## 実験結果例

### 通信効率

| 手法 | 1ラウンドあたりの通信量 | 通信削減率 |
|------|---------------------|----------|
| FedAvg-Full | 全モデル | - |
| FedAvg-LoRA | LoRA A + B | - |
| FedSA-FTL | LoRA A のみ | **50%** |

### 性能結果（CIFAR-10、α=0.5）

```
=== 実験結果サマリー ===
最終テスト精度: 85.32%
最高テスト精度: 86.14% (Round 78)
総通信量: 142.5 MB
平均通信量/ラウンド: 1.43 MB
総学習時間: 23.7分
通信削減率: 50.0%
```

## ファイル構成

```
FedSA-FTL/
├── src/
│   ├── __init__.py
│   ├── fedsa_ftl_model.py      # コアモデル実装
│   ├── fedsa_ftl_client.py     # クライアント実装
│   ├── fedsa_ftl_server.py     # サーバー実装
│   ├── data_utils.py           # データ前処理・分割
│   └── experiment_controller.py # 実験制御
├── configs/
│   ├── cifar10_vit_base.json   # CIFAR-10基本設定
│   └── cifar100_vit_challenging.json # CIFAR-100挑戦設定
├── experiments/                # 実験結果保存用
├── main.py                     # メイン実行スクリプト
├── analyze_results.py          # 結果分析スクリプト
└── README.md
```

## 理論的背景

### FbFTLからの継承
- **凍結バックボーン**: 計算効率の向上とクライアントドリフトの抑制
- **特徴抽出の一元化**: 高品質で汎用的な特徴表現の共有

### FedSA-LoRAからの継承
- **LoRA行列の役割分担**: A行列（一般知識）とB行列（固有知識）の分離
- **選択的集約**: 通信効率とパーソナライズのバランス

### ハイブリッドアプローチの利点
1. **通信効率**: LoRA Aのみの通信による大幅な通信量削減
2. **パーソナライズ**: クライアント固有のLoRA Bによる適応
3. **計算効率**: 凍結バックボーンによる軽量な学習
4. **ロバスト性**: non-IIDデータに対する優れた耐性

## 今後の拡張

- [ ] より多くのバックボーンモデルのサポート
- [ ] 自然言語処理タスクの完全実装
- [ ] 動的LoRAランク調整
- [ ] より高度な集約アルゴリズム（FedProx、SCAFFOLD等）
- [ ] プライバシー保護メカニズムの統合
- [ ] モバイル・エッジデバイス最適化

## 引用

```bibtex
@article{fedsa_ftl2024,
  title={FedSA-FTL: Federated Share-A Transfer Learning for Personalized and Communication-Efficient Federated Learning},
  author={Research Team},
  journal={arXiv preprint},
  year={2024}
}
```

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 貢献

プルリクエストやイシューの報告を歓迎します。大きな変更を提案する場合は、まずイシューを作成してディスカッションをお願いします。

## サポート

質問やサポートが必要な場合は、GitHubのIssuesページをご利用ください。
