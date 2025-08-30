#!/bin/bash

# FedSA-FTL セットアップスクリプト
# このスクリプトはconda環境を作成し、必要な依存関係をインストールします

set -e  # エラーが発生したら停止

echo "🚀 FedSA-FTL セットアップを開始します..."

# conda環境が既に存在するかチェック
if conda env list | grep -q "fedsa-ftl"; then
    echo "⚠️  fedsa-ftl環境が既に存在します。削除して再作成しますか? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "🗑️  既存の環境を削除中..."
        conda env remove -n fedsa-ftl -y
    else
        echo "❌ セットアップを中止しました。"
        exit 1
    fi
fi

# environment.ymlから環境を作成
echo "📦 conda環境を作成中..."
conda env create -f environment.yml

echo "✅ 環境作成が完了しました！"

# 環境をアクティベートして追加パッケージをインストール
echo "🔧 追加パッケージをインストール中..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fedsa-ftl

# pipで追加パッケージをインストール
pip install -r requirements.txt

echo ""
echo "🎉 セットアップが完了しました！"
echo ""
echo "次のステップ:"
echo "1. 環境をアクティベート: conda activate fedsa-ftl"
echo "2. 実装をテスト: python verify_implementation.py"
echo "3. 実験を実行: python main.py --config configs/cifar10_vit_base.json"
echo ""
echo "環境を非アクティブ化: conda deactivate"
