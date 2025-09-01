# 機械学習完了通知機能

学習関連コードとは独立した、機械学習完了時のSlack通知機能です。

## 🎯 概要

- **目的**: 機械学習の訓練完了時のみSlackに通知
- **独立性**: 学習コードから完全に分離
- **シンプル**: 最小限の機能のみ実装

## 📁 ファイル構成

```
fedsa_ftl_standalone/
├── notify_completion.py    # 完了通知専用モジュール（独立）
├── README_NOTIFICATION.md  # このファイル
└── main.py                # 学習完了時にnotify_completion.pyを呼び出し
```

## 🚀 使用方法

### 1. Slack Webhook URLの設定

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### 2. 自動通知（推奨）

学習が完了すると自動的に通知が送信されます：

```bash
# 学習実行（完了時に自動通知）
python main.py --config configs/cifar10_vit_private.yaml
```

### 3. 手動通知

#### 結果ファイルから通知
```bash
# results.jsonファイルから詳細な通知を送信
python notify_completion.py results experiments/example/20250901_120000/results.json

# 実験名を指定
python notify_completion.py results experiments/example/20250901_120000/results.json --name "Custom Experiment"
```

#### シンプルメッセージ
```bash
# 簡単なメッセージを送信
python notify_completion.py message "Training completed successfully!"

# 実験名付き
python notify_completion.py message "Accuracy: 85.3%" --name "CIFAR-10 Experiment"
```

#### テスト通知
```bash
# 通知テスト
python notify_completion.py test

# 実験名付きテスト
python notify_completion.py test --name "Test Notification"
```

## 📱 通知内容

### 詳細通知（results.jsonから）
```
🎉 Machine Learning Training Complete

Experiment: fedsa_ftl_cifar10_vit
Status: Excellent
Duration: 2h 15m 30s
Completed: 2025-01-15 14:30

Final Accuracy: 85.75% ± 3.21%
Best Accuracy: 87.90%
Rounds: 50
Communication: 1.23 MB

🔒 Privacy Budget Spent: ε = 15.2
📁 Results: experiments/example/results.json
```

### シンプル通知
```
🎯 Training Notification

Experiment: My Training
Message: Training completed successfully!
Sent at 2025-01-15 14:30:15
```

## ⚙️ カスタマイズ

### ステータス判定の変更
`notify_completion.py`の以下の部分を編集：

```python
# Determine status
if final_accuracy >= 80:      # 80%以上で「Excellent」
    status_emoji = "🎉"
    status_text = "Excellent"
elif final_accuracy >= 70:    # 70-79%で「Good」
    status_emoji = "✅"
    status_text = "Good"
# ...
```

### 通知フォーマットの変更
`send_completion_notification`関数内の`blocks`配列を編集してSlackの表示内容をカスタマイズできます。

## 🔧 エラー対処

### 通知が送信されない場合

1. **Webhook URL確認**
   ```bash
   echo $SLACK_WEBHOOK_URL
   ```

2. **テスト実行**
   ```bash
   python notify_completion.py test
   ```

3. **結果ファイル確認**
   ```bash
   # results.jsonの存在と内容確認
   cat experiments/your_experiment/results.json
   ```

### 一般的なエラー

- `SLACK_WEBHOOK_URL not set`: 環境変数が設定されていない
- `Results file not found`: 指定したresults.jsonが存在しない
- `Invalid JSON`: results.jsonの形式が正しくない
- `Failed to send notification`: ネットワークエラーまたは無効なWebhook URL

## 🔒 セキュリティ

- Webhook URLは機密情報です
- 環境変数で管理してください
- GitリポジトリにWebhook URLをコミットしないでください

## 📝 独立性の利点

1. **保守性**: 学習コードと通知コードが分離
2. **再利用性**: 他のプロジェクトでも使用可能
3. **テスト性**: 通知機能を単独でテスト可能
4. **オプション性**: 通知機能なしでも学習可能