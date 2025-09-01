"""
Notification utilities for FedSA-FTL
Sends training completion notifications to Slack
"""

import json
import os
from typing import Dict, Optional
import requests
from datetime import datetime


class SlackNotifier:
    """
    Slack notification sender for training events
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Slack notifier
        
        Args:
            webhook_url: Slack webhook URL (can also be set via environment variable)
        """
        # Try to get webhook URL from parameter, environment variable, or config file
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self.enabled = self.webhook_url is not None
        
        if not self.enabled:
            print("⚠️  Slack notifications disabled (no webhook URL provided)")
    
    def send_message(self, message: str, blocks: Optional[list] = None) -> bool:
        """
        Send a message to Slack
        
        Args:
            message: Text message to send
            blocks: Optional Slack blocks for rich formatting
        
        Returns:
            True if message was sent successfully
        """
        if not self.enabled:
            return False
        
        payload = {"text": message}
        if blocks:
            payload["blocks"] = blocks
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send Slack notification: {e}")
            return False
    
    def send_training_start(self, config: Dict) -> bool:
        """
        Send training start notification
        
        Args:
            config: Training configuration
        
        Returns:
            True if message was sent successfully
        """
        experiment_name = config.get('experiment', {}).get('name', 'Unknown')
        num_clients = config.get('federated', {}).get('num_clients', 0)
        num_rounds = config.get('federated', {}).get('num_rounds', 0)
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚀 FedSA-FTL Training Started"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Experiment:*\n{experiment_name}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Clients:*\n{num_clients}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Rounds:*\n{num_rounds}"
                    }
                ]
            }
        ]
        
        # Add privacy info if enabled
        if config.get('privacy', {}).get('enable_privacy', False):
            epsilon = config['privacy'].get('epsilon', 'N/A')
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🔒 *Privacy:* Enabled (ε={epsilon})"
                }
            })
        
        return self.send_message(
            f"Training started: {experiment_name}",
            blocks=blocks
        )
    
    def send_training_complete(self, config: Dict, summary: Dict, duration_seconds: float) -> bool:
        """
        Send training completion notification
        
        Args:
            config: Training configuration
            summary: Training summary statistics
            duration_seconds: Training duration in seconds
        
        Returns:
            True if message was sent successfully
        """
        experiment_name = config.get('experiment', {}).get('name', 'Unknown')
        
        # Format duration
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{hours}h {minutes}m {seconds}s"
        
        # Extract key metrics
        final_accuracy = summary.get('final_avg_accuracy', 0)
        final_std = summary.get('final_std_accuracy', 0)
        best_accuracy = summary.get('best_test_accuracy', 0)
        total_comm_mb = summary.get('total_communication_mb', 0)
        
        # Determine status emoji based on accuracy
        if final_accuracy >= 80:
            status_emoji = "🎉"
            status_text = "Excellent"
        elif final_accuracy >= 70:
            status_emoji = "✅"
            status_text = "Good"
        elif final_accuracy >= 60:
            status_emoji = "📊"
            status_text = "Moderate"
        else:
            status_emoji = "⚠️"
            status_text = "Low"
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} FedSA-FTL Training Complete"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Experiment:*\n{experiment_name}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Duration:*\n{duration_str}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n{status_text}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*📈 Performance Metrics*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Final Accuracy:*\n{final_accuracy:.2f}% ± {final_std:.2f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Best Accuracy:*\n{best_accuracy:.2f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Total Rounds:*\n{summary.get('total_rounds', 0)}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Communication:*\n{total_comm_mb:.2f} MB"
                    }
                ]
            }
        ]
        
        # Add privacy spent if enabled
        if summary.get('privacy', {}).get('privacy_enabled', False):
            epsilon_spent = summary['privacy'].get('total_epsilon_spent', 'N/A')
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🔒 *Privacy Budget Spent:* ε={epsilon_spent:.2f}"
                }
            })
        
        # Add output directory
        output_dir = config.get('experiment', {}).get('output_dir', 'N/A')
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"📁 Results saved to: `{output_dir}`"
                }
            ]
        })
        
        return self.send_message(
            f"Training complete: {experiment_name} - Accuracy: {final_accuracy:.2f}% ({status_text})",
            blocks=blocks
        )
    
    def send_error_notification(self, config: Dict, error_message: str) -> bool:
        """
        Send error notification
        
        Args:
            config: Training configuration
            error_message: Error message
        
        Returns:
            True if message was sent successfully
        """
        experiment_name = config.get('experiment', {}).get('name', 'Unknown')
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "❌ FedSA-FTL Training Failed"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Experiment:*\n{experiment_name}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error:*\n```{error_message[:500]}```"  # Limit error message length
                }
            }
        ]
        
        return self.send_message(
            f"Training failed: {experiment_name}",
            blocks=blocks
        )
    
    def send_checkpoint_notification(self, round_num: int, accuracy: float) -> bool:
        """
        Send checkpoint notification for significant milestones
        
        Args:
            round_num: Current round number
            accuracy: Current test accuracy
        
        Returns:
            True if message was sent successfully
        """
        if accuracy >= 90:  # Only notify for high accuracy milestones
            return self.send_message(
                f"🎯 Milestone reached! Round {round_num}: {accuracy:.2f}% accuracy"
            )
        return False


def setup_slack_notifier(config: Dict) -> Optional[SlackNotifier]:
    """
    Setup Slack notifier from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        SlackNotifier instance or None if disabled
    """
    notification_config = config.get('notification', {})
    
    if not notification_config.get('enable_slack', False):
        return None
    
    webhook_url = notification_config.get('slack_webhook_url')
    if not webhook_url:
        # Try to get from environment variable
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    
    if not webhook_url:
        print("Warning: Slack notifications enabled but no webhook URL provided")
        return None
    
    return SlackNotifier(webhook_url)