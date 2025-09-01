"""
Test script for Slack notifications
"""

import os
from src.notification_utils import SlackNotifier


def test_slack_notifications():
    """Test Slack notification functionality"""
    print("Testing Slack Notifications")
    print("=" * 50)
    
    # Get webhook URL from environment variable
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    
    if not webhook_url:
        print("❌ SLACK_WEBHOOK_URL environment variable not set")
        print("\nTo set it:")
        print("export SLACK_WEBHOOK_URL=\"https://hooks.slack.com/services/YOUR/WEBHOOK/URL\"")
        return False
    
    # Initialize notifier
    notifier = SlackNotifier(webhook_url)
    
    if not notifier.enabled:
        print("❌ Slack notifier is not enabled")
        return False
    
    print("✅ Slack notifier initialized successfully")
    print(f"Webhook URL: {webhook_url[:50]}...")
    
    # Test simple message
    print("\n1. Testing simple message...")
    success = notifier.send_message("🧪 FedSA-FTL Slack notification test")
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test training start notification
    print("\n2. Testing training start notification...")
    test_config = {
        'experiment': {'name': 'test_experiment'},
        'federated': {'num_clients': 10, 'num_rounds': 50},
        'privacy': {'enable_privacy': True, 'epsilon': 2.0}
    }
    success = notifier.send_training_start(test_config)
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test training completion notification
    print("\n3. Testing training completion notification...")
    test_summary = {
        'final_avg_accuracy': 75.5,
        'final_std_accuracy': 3.2,
        'best_test_accuracy': 78.9,
        'total_rounds': 50,
        'total_communication_mb': 1.5,
        'privacy': {
            'privacy_enabled': True,
            'total_epsilon_spent': 15.2
        }
    }
    success = notifier.send_training_complete(test_config, test_summary, 3600)  # 1 hour
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test error notification
    print("\n4. Testing error notification...")
    success = notifier.send_error_notification(test_config, "Test error: Connection timeout")
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test milestone notification
    print("\n5. Testing milestone notification...")
    success = notifier.send_checkpoint_notification(25, 92.5)
    print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
    
    print("\n" + "=" * 50)
    print("Slack notification test completed!")
    print("Check your Slack channel for the test messages.")
    
    return True


if __name__ == "__main__":
    test_slack_notifications()