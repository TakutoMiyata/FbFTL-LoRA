#!/usr/bin/env python3
"""
Test script for Slack notifications
Run this to verify that Slack notifications are working correctly.

Usage:
    export SLACK_WEBHOOK_URL="your_webhook_url_here"
    python test_slack_notification.py
"""

import os
import sys
from datetime import datetime

# Load environment variables from .env file
def load_env_file(env_path='.env'):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        print(f"✅ Environment variables loaded from {env_path}")
        return True
    else:
        print(f"⚠️  No .env file found at {env_path}")
        return False

# Load .env file at startup
load_env_file()

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from notification_utils import SlackNotifier


def test_slack_notifications():
    """Test all types of Slack notifications"""
    
    # Check if webhook URL is configured
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if not webhook_url:
        print("❌ SLACK_WEBHOOK_URL environment variable not set")
        print("Please set it with: export SLACK_WEBHOOK_URL='your_webhook_url_here'")
        return False
    
    # Initialize notifier
    slack_notifier = SlackNotifier(webhook_url)
    
    if not slack_notifier.enabled:
        print("❌ Slack notifier could not be initialized")
        return False
    
    print("✅ Slack notifier initialized successfully")
    print("Testing different types of notifications...\n")
    
    # Test configuration for notifications
    test_config = {
        'experiment': {
            'name': 'ViT QuickStart Test',
            'output_dir': 'experiments/test'
        },
        'federated': {
            'num_clients': 3,
            'num_rounds': 50
        },
        'privacy': {
            'enable_privacy': False
        }
    }
    
    # Test 1: Training start notification
    print("🚀 Testing training start notification...")
    success = slack_notifier.send_training_start(test_config)
    print(f"   {'✅ Success' if success else '❌ Failed'}")
    
    # Test 2: Progress update notification
    print("📊 Testing progress update notification...")
    round_stats = {
        'train_accuracy': 65.5,
        'test_accuracy': 62.3,
        'communication_cost_mb': 2.4
    }
    server_summary = {
        'best_test_accuracy': 68.9
    }
    success = slack_notifier.send_progress_update(
        test_config, 19, round_stats, server_summary  # Round 20
    )
    print(f"   {'✅ Success' if success else '❌ Failed'}")
    
    # Test 3: Training completion notification
    print("🎉 Testing training completion notification...")
    summary = {
        'final_avg_accuracy': 72.1,
        'final_std_accuracy': 3.2,
        'best_test_accuracy': 75.8,
        'total_rounds': 50,
        'total_communication_mb': 120.5
    }
    training_duration = 3600 * 2.5  # 2.5 hours
    success = slack_notifier.send_training_complete(
        test_config, summary, training_duration
    )
    print(f"   {'✅ Success' if success else '❌ Failed'}")
    
    # Test 4: Error notification
    print("❌ Testing error notification...")
    error_message = "Test error: This is a simulated error for testing purposes"
    success = slack_notifier.send_error_notification(test_config, error_message)
    print(f"   {'✅ Success' if success else '❌ Failed'}")
    
    # Test 5: Simple message
    print("💬 Testing simple message...")
    success = slack_notifier.send_message("🧪 Slack notification test completed successfully!")
    print(f"   {'✅ Success' if success else '❌ Failed'}")
    
    print("\n✅ All notification tests completed!")
    print("Check your Slack channel to see the test messages.")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Slack Notification Test")
    print("=" * 60)
    
    success = test_slack_notifications()
    
    if success:
        print("\n🎉 Test completed! Check your Slack channel for test messages.")
    else:
        print("\n❌ Test failed. Please check your SLACK_WEBHOOK_URL configuration.")
    
    print("\nTo use Slack notifications in training:")
    print("1. Set SLACK_WEBHOOK_URL environment variable")
    print("2. Run quickstart_vit.py normally")
    print("3. Notifications will be sent every 10 rounds")
