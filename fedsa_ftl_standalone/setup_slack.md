# Slack通知のセットアップガイド

FedSA-FTLの学習完了をSlackで通知する機能の設定方法

## 1. Slack Webhook URLの取得

### ステップ1: Slack Appの作成
1. https://api.slack.com/apps にアクセス
2. 「Create New App」→「From scratch」を選択
3. App名を入力（例: FedSA-FTL Notifier）
4. ワークスペースを選択

### ステップ2: Incoming Webhookの有効化
1. 「Incoming Webhooks」を選択
2. 「Activate Incoming Webhooks」をOnに設定
3. 「Add New Webhook to Workspace」をクリック
4. 通知を送信したいチャンネルを選択
5. 「Allow」をクリック

### ステップ3: Webhook URLをコピー
生成されたWebhook URLをコピーしてください（`https://hooks.slack.com/services/...` の形式）

## 2. 環境変数の設定

### 方法1: 環境変数で設定（推奨）
```bash
# bashの場合
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# fishの場合
set -x SLACK_WEBHOOK_URL "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Windows（Command Prompt）の場合
set SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Windows（PowerShell）の場合
$env:SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### 方法2: 設定ファイルで直接指定
```yaml
notification:
  enable_slack: true
  slack_webhook_url: "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

## 3. 通知の有効化

### 既存の設定ファイルを使用する場合
```yaml
# configs/cifar10_vit_private.yaml などに追加
notification:
  enable_slack: true  # Slack通知を有効化
```

### コマンドラインでの実行
```bash
# 環境変数を設定後、通常通り実行
python main.py --config configs/cifar10_vit_private.yaml

# または
python quickstart.py
```

## 4. 通知内容

### 学習開始時の通知
- 🚀 実験名、開始時刻
- クライアント数、ラウンド数
- プライバシー設定（有効な場合）

### 学習完了時の通知
- 🎉/✅/📊/⚠️ 結果に応じたステータス
- 最終精度、最高精度
- 学習時間、通信量
- プライバシー予算消費量（有効な場合）
- 結果保存場所

### エラー発生時の通知
- ❌ エラーメッセージ
- 実験名、発生時刻

## 5. テスト

通知機能をテストするには：

```python
# test_slack.py
from src.notification_utils import SlackNotifier
import os

# Webhook URLを設定
webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
notifier = SlackNotifier(webhook_url)

# テストメッセージを送信
if notifier.enabled:
    success = notifier.send_message("🧪 FedSA-FTL Slack通知テスト")
    print(f"通知送信: {'成功' if success else '失敗'}")
else:
    print("Slack通知が無効です")
```

## 6. トラブルシューティング

### 通知が送信されない場合
1. Webhook URLが正しく設定されているか確認
2. ネットワーク接続を確認
3. Slackワークスペースでアプリの権限を確認

### 設定の確認
```bash
# 環境変数の確認
echo $SLACK_WEBHOOK_URL

# または
python -c "import os; print(os.environ.get('SLACK_WEBHOOK_URL', 'Not set'))"
```

## 7. セキュリティ上の注意

- Webhook URLは秘密情報です。Gitリポジトリにコミットしないでください
- 環境変数での管理を推奨します
- 必要に応じてWebhook URLを再生成してください

## 8. カスタマイズ

通知内容をカスタマイズしたい場合は、`src/notification_utils.py`の`SlackNotifier`クラスを修正してください。

例：
- 通知頻度の調整
- 追加のメトリクスの表示
- 通知フォーマットの変更