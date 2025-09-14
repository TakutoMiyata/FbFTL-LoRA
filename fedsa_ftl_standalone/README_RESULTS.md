# ViT 結果保存とSlack通知機能

`quickstart_vit.py`に結果保存機能と10ラウンドごとのSlack通知機能を追加しました。

## 新機能

### 1. 実行結果の自動保存
- **JSON形式**: ラウンドごとの詳細な結果を保存
- **CSV形式**: プロット用の要約データを保存
- **チェックポイント**: 指定間隔でのスナップショット保存

### 2. Slack通知機能
- **開始通知**: 学習開始時の設定情報
- **進捗通知**: 10ラウンドごとの性能更新
- **完了通知**: 最終結果とサマリー
- **エラー通知**: 実行時エラーの即座な報告

### 3. 結果可視化
- **自動プロット生成**: 学習曲線、通信コスト、性能分布
- **ダッシュボード**: 包括的な結果サマリー

## セットアップ

### 依存関係のインストール
```bash
pip install -r requirements.txt
```

### Slack通知の設定（オプション）
1. Slack Workspaceでwebhook URLを取得
2. 環境変数に設定:
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

## 使用方法

### 基本実行
```bash
python quickstart_vit.py --config configs/cifar100_vit_base.yaml
```

### Slack通知のテスト
```bash
python test_slack_notification.py
```

### 結果のプロット生成
```bash
# 自動で全てのプロット生成
python plot_results.py experiments/fedsa_ftl_cifar100_vit/20240912_123456/

# 特定のプロットのみ生成
python plot_results.py experiments/fedsa_ftl_cifar100_vit/20240912_123456/ --plots accuracy communication

# カスタム出力ディレクトリ
python plot_results.py experiments/fedsa_ftl_cifar100_vit/20240912_123456/ --output-dir plots/
```

## 保存される結果ファイル

### 実験ディレクトリ構造
```
experiments/fedsa_ftl_cifar100_vit/YYYYMMDD_HHMMSS/
├── final_results.json          # 完全な実行結果
├── training_results.json       # リアルタイム更新結果
├── results_summary.csv         # プロット用データ
├── checkpoint_round_N.json     # チェックポイントファイル
├── accuracy_curves.png         # 学習曲線
├── communication_cost.png      # 通信コスト分析
├── client_performance.png      # クライアント性能分布
└── training_dashboard.png      # 総合ダッシュボード
```

### JSON結果フォーマット
```json
{
  "config": {...},              // 実行設定
  "start_time": "2024-09-12T12:34:56",
  "end_time": "2024-09-12T14:56:78",
  "training_duration_seconds": 8542,
  "rounds": [
    {
      "round": 1,
      "timestamp": "2024-09-12T12:35:12",
      "selected_clients": [0, 1, 2],
      "avg_train_accuracy": 23.4,
      "avg_test_accuracy": 21.8,
      "individual_train_accuracies": [23.1, 23.9, 23.2],
      "individual_test_accuracies": [21.5, 22.3, 21.6],
      "communication_cost_mb": 2.4,
      "is_best_round": true
    },
    ...
  ],
  "summary": {
    "best_test_accuracy": 75.8,
    "best_round": 42,
    "total_rounds": 50,
    "total_communication_mb": 120.5,
    "final_avg_accuracy": 72.1,
    "training_duration_hours": 2.37,
    "model_name": "vit_small",
    "dataset": "cifar100"
  }
}
```

### CSV結果フォーマット
| round | train_accuracy | test_accuracy | communication_mb | is_best |
|-------|---------------|---------------|------------------|---------|
| 1     | 23.4          | 21.8          | 2.4              | True    |
| 2     | 25.1          | 23.2          | 2.4              | True    |
| ...   | ...           | ...           | ...              | ...     |

## Slack通知の内容

### 10ラウンドごとの進捗通知
- 現在のラウンド数と進捗率
- 学習精度とテスト精度
- 最高精度の記録
- 通信コスト
- プライバシー予算（DP有効時）

### 完了通知
- 最終結果サマリー
- 学習時間
- モデル性能評価
- 結果ファイルの場所

## プロット機能

### 生成されるプロット
1. **学習曲線**: 学習精度とテスト精度の推移
2. **通信コスト**: ラウンドあたりと累積コスト
3. **クライアント性能**: 個別クライアントの性能分布
4. **ダッシュボード**: 包括的な結果可視化

### カスタマイズ
- スタイル: seaborn style
- 解像度: 300 DPI
- フォーマット: PNG
- 自動保存: 実験ディレクトリに保存

## トラブルシューティング

### Slack通知が送信されない
1. SLACK_WEBHOOK_URLが正しく設定されているか確認
2. `test_slack_notification.py`でテスト実行
3. webhookのURL形式を確認

### プロット生成エラー
1. seabornとmatplotlibが最新版かチェック
2. 結果ファイルが存在するか確認
3. 出力ディレクトリの書き込み権限を確認

### 結果ファイルが見つからない
1. 実験ディレクトリのパスを確認
2. 実行が正常に完了しているか確認
3. `training_results.json`の代わりに`final_results.json`を使用

## 高度な使用例

### バッチ処理での結果分析
```bash
# 複数の実験結果を一括プロット
for exp_dir in experiments/*/20240912_*/; do
    python plot_results.py "$exp_dir" --plots dashboard
done
```

### 結果比較スクリプト
```python
import json
import pandas as pd

# 複数実験の結果を比較
experiments = ['exp1', 'exp2', 'exp3']
comparison_data = []

for exp in experiments:
    with open(f'experiments/{exp}/final_results.json') as f:
        result = json.load(f)
        comparison_data.append({
            'experiment': exp,
            'best_accuracy': result['summary']['best_test_accuracy'],
            'training_time': result['summary']['training_duration_hours']
        })

df = pd.DataFrame(comparison_data)
print(df)
```

この機能により、ViTの学習結果を包括的に記録・分析し、リアルタイムで進捗を把握できるようになります。
