# Slack通知の設定方法

FedSA-FTLの訓練進捗をSlackに自動通知する機能の設定方法です。

## 🔧 設定手順

### 1. **Slack Webhook URLの取得**

#### **Slack App作成**
1. https://api.slack.com/apps にアクセス
2. **"Create New App"** をクリック
3. **"From scratch"** を選択
4. アプリ名（例：`FedSA-FTL Notifier`）とワークスペースを選択

#### **Incoming Webhookの有効化**
1. 左メニューから **"Incoming Webhooks"** をクリック
2. **"Activate Incoming Webhooks"** をオンに切り替え
3. **"Add New Webhook to Workspace"** をクリック
4. 通知を送信したいチャンネルを選択
5. **"Allow"** をクリック

#### **Webhook URLのコピー**
1. 生成されたWebhook URLをコピー
   ```
   https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
   ```

### 2. **環境変数の設定**

#### **方法1: 環境変数として設定**
```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

#### **方法2: .envファイルを使用**
プロジェクトルートに`.env`ファイルを作成：
```bash
echo "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL" > .env
```

### 3. **通知のテスト**

```bash
# テスト通知を送信
python test_slack.py
```

成功すると以下のメッセージが表示され、Slackにテスト通知が送信されます：
```
✅ Slack notification test successful!
```

## 📊 通知の種類

### **1. 訓練開始通知**
```
🚀 FedSA-FTL Training Started
Experiment: fedsa_ftl_cifar10_vgg16
Clients: 10 | Rounds: 100
```

### **2. 進捗通知（10ラウンドごと）**
```
📊 Training Progress - Round 20/100
Progress: 20.0% ██████░░░░░░░░░░░░░░
Train Accuracy: 75.3%
Test Accuracy: 68.7%
Best So Far: 68.7%
Communication: 1.59 MB
🔒 Privacy Budget: ε=2.00
```

### **3. 訓練完了通知**
```
🎉 FedSA-FTL Training Complete
Final Accuracy: 85.2% ± 2.1% (Excellent)
Best Accuracy: 86.1%
Total Rounds: 100
Communication: 159.0 MB
Duration: 2.5 hours
```

## ⚙️ 設定のカスタマイズ

### **通知頻度の変更**
`configs/*.yaml`ファイルで調整：
```yaml
federated:
  checkpoint_freq: 5  # 5ラウンドごとに通知（デフォルト: 10）
```

### **通知の無効化**
環境変数を設定しなければ、通知は無効になります：
```
📴 Slack notifications disabled
```

### **エラー通知**
訓練中にエラーが発生した場合も自動通知されます：
```
❌ FedSA-FTL Training Error
Error: CUDA out of memory
Experiment: fedsa_ftl_cifar100_vgg16
```

## 🔒 セキュリティ

- Webhook URLは機密情報です。コードにハードコーディングしないでください
- 環境変数や`.env`ファイルを使用してください
- `.env`ファイルを`.gitignore`に追加してください

## 🚀 実行例

```bash
# 環境変数設定
export SLACK_WEBHOOK_URL="your_webhook_url_here"

# 通知付きで訓練開始
python main.py --config configs/cifar10_vgg16_base.yaml
```

訓練開始時、10ラウンドごと、完了時に自動的にSlack通知が送信されます！