# Custom DP Implementation (Memory Efficient)

このバージョンは、Opacusの`PrivacyEngine`を使わずに差分プライバシーを実装したメモリ効率的な版です。

## 特徴

✅ **メモリ効率的**: Opacusのgrad_sampleバッファを使わない
✅ **正確なε計算**: Opacusの`RDPAccountant`のみを使用してεを正確に計算
✅ **A行列のみにDP**: B行列と分類器は通常のSGD
✅ **詳細なプライバシー分析**: ε、ノイズの標準偏差などを表示

## メモリ消費の比較

| 方法 | メモリ使用量 | grad_sampleバッファ |
|------|-------------|-------------------|
| `fedsa_shareA_dp` (Opacus) | 非常に大きい | あり（巨大） |
| `fedsa_custom_dp` (Custom) | 小さい | なし |
| `fedsa` (No DP) | 最小 | なし |

## 使い方

### 1. ファイル構成

- `quickstart_bit_tff_custom_dp.py` - カスタムDP実装版（新規）
- `configs/bit_custom_dp_example.yaml` - サンプル設定ファイル

### 2. 設定ファイル

重要な設定項目：

```yaml
federated:
  aggregation_method: fedsa_custom_dp  # ← これが重要！

privacy:
  enable_privacy: true
  epsilon: 8.0                 # プライバシー予算
  delta: 1.0e-05              # δパラメータ
  max_grad_norm: 0.5          # 勾配クリッピング閾値
  noise_multiplier: 1.0       # ノイズスケール
```

### 3. 実行方法

```bash
# カスタムDP版を実行
python quickstart_bit_tff_custom_dp.py --config configs/bit_custom_dp_example.yaml
```

## プライバシーパラメータの説明

### ε (Epsilon) - プライバシー予算
- 小さいほどプライバシー保護が強い
- 推奨値: 1.0 ~ 10.0
- 例: ε=8.0 は一般的な設定

### δ (Delta)
- 通常は 1e-5 程度
- データセットサイズの逆数より小さく設定

### max_grad_norm
- 勾配クリッピングの閾値
- 小さいほどプライバシー保護が強いが、精度が下がる
- 推奨値: 0.1 ~ 1.0

### noise_multiplier
- ノイズの強さを制御
- 実際のノイズ標準偏差 = `noise_multiplier × max_grad_norm`
- 例: `1.0 × 0.5 = 0.5`

## 出力される情報

訓練中、各ラウンドで以下の情報が表示されます：

```
=== Privacy Analysis (Custom DP-SGD (A-only, memory efficient)) ===
Current ε (avg): 2.4531, ε (max): 2.4531
Privacy Budget: ε≤8.0
δ: 1e-05
Noise multiplier: 1.0
Max grad norm: 0.5
Actual noise std: 0.5000
=============================================
```

### 表示内容の説明

- **Current ε (avg/max)**: 現在のプライバシー消費量
- **Privacy Budget**: 目標ε
- **Noise multiplier**: 設定されたノイズ倍率
- **Max grad norm**: 勾配クリッピング閾値
- **Actual noise std**: 実際に追加されるノイズの標準偏差

## Google Colabでの使用

### インストール

```python
# 必要なライブラリ
!pip install torch torchvision
!pip install transformers peft timm
!pip install opacus==1.4.0  # RDPAccountantのみ使用（軽量）
!pip install pyyaml tqdm pandas matplotlib
```

### 実行例

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Feature-based-Federated-Transfer-Learning-main/fedsa_ftl_standalone

!python quickstart_bit_tff_custom_dp.py \
  --config configs/bit_custom_dp_example.yaml \
  --rounds 30 \
  --clients 3
```

### Colab用の推奨設定

```yaml
data:
  num_clients: 3-5        # クライアント数を減らす
  batch_size: 16-32       # バッチサイズを減らす

federated:
  num_rounds: 30-50       # ラウンド数を減らす

model:
  lora:
    r: 4                  # LoRAランクを減らす
```

## トラブルシューティング

### メモリ不足エラー

1. `batch_size`を減らす（32 → 16）
2. `num_clients`を減らす（10 → 3）
3. LoRAの`r`を減らす（8 → 4）
4. `model_name`を小さいモデルに変更（`bit_s_r50x1`）

### εが想定より大きい

1. `noise_multiplier`を増やす（1.0 → 1.5）
2. `max_grad_norm`を減らす（0.5 → 0.3）
3. `epochs`を減らす（5 → 3）

## 実装の仕組み

### DPOptimizer (dp_utils.py)

1. **Per-sample勾配クリッピング**
   - 各サンプルごとに勾配のL2ノルムを計算
   - `max_grad_norm`を超えたらクリッピング

2. **ノイズ追加**
   - ガウシアンノイズを勾配に追加
   - `N(0, (noise_multiplier × max_grad_norm)²)`

3. **ε計算**
   - Opacusの`RDPAccountant`を使用（正確）
   - またはカスタム近似式（Opacusなし）

### メモリ効率の理由

- Opacusの`PrivacyEngine`は、各パラメータに`grad_sample`属性を保存
- バッチサイズ×パラメータ数の巨大なテンソルが必要
- カスタム実装では、マイクロバッチごとに処理して累積
- `grad_sample`バッファ不要 → メモリ節約

## 論文執筆時の注意

εの値は**Opacus RDPAccountant**を使用した値を報告してください：

```python
privacy_analysis = {
    'opacus_epsilon': 2.45,  # ← これを使う
    'custom_epsilon': 2.60,  # 近似値（参考）
}
```

出力JSONファイルの`privacy_analysis.full_analysis`に詳細が保存されます。
