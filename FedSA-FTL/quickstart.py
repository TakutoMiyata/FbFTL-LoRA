"""
FedSA-FTL クイックスタートスクリプト
conda環境での実行をテストします
"""

import sys
import torch
import transformers
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_environment():
    """環境設定をテスト"""
    print("🔍 環境テストを開始します...")
    
    # 基本パッケージのテスト
    print(f"✅ Python: {sys.version}")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ Transformers: {transformers.__version__}")
    
    # CUDA利用可能性をテスト
    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.version.cuda} (GPU: {torch.cuda.get_device_name()})")
    else:
        print("⚠️  CUDA: 利用不可 (CPUで実行されます)")
    
    # インポートテスト
    try:
        from fedsa_ftl_model import FedSAFTLModel
        from fedsa_ftl_client import FedSAFTLClient
        from fedsa_ftl_server import FedSAFTLServer
        print("✅ FedSA-FTL モジュールのインポート成功")
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    
    # 簡単なモデル作成テスト
    try:
        print("🧪 モデル作成テスト...")
        model = FedSAFTLModel.create_vision_model(
            model_name="google/vit-base-patch16-224",
            num_classes=10,
            lora_rank=16
        )
        print(f"✅ モデル作成成功: {type(model).__name__}")
        
        # 簡単な推論テスト
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ 推論テスト成功: 出力形状 {output.shape}")
        
    except Exception as e:
        print(f"❌ モデルテストエラー: {e}")
        return False
    
    print("🎉 すべてのテストが成功しました！")
    return True

def show_usage():
    """使用方法を表示"""
    print("\n" + "="*60)
    print("📚 FedSA-FTL 使用方法")
    print("="*60)
    print()
    print("1. 基本的な実験実行:")
    print("   python main.py --config configs/cifar10_vit_base.json")
    print()
    print("2. カスタム設定:")
    print("   python main.py --config configs/cifar100_vit_large.json --num_clients 20")
    print()
    print("3. 結果の分析:")
    print("   python analyze_results.py --experiment_dir results/latest")
    print()
    print("4. 設定ファイルの詳細:")
    print("   - configs/: 実験設定ファイル")
    print("   - src/: コアモジュール")
    print("   - results/: 実験結果保存先")
    print()

if __name__ == "__main__":
    print("🌟 FedSA-FTL クイックスタート")
    print("="*50)
    
    if test_environment():
        show_usage()
    else:
        print("\n❌ 環境に問題があります。setup_conda_env.shを再実行してください。")
