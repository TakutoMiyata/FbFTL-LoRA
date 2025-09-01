"""
Simple completion notification for machine learning training
Standalone module for sending Slack notifications when training completes
"""

import json
import os
import requests
from datetime import datetime
from typing import Dict, Optional
import argparse


class CompletionNotifier:
    """
    Simple notification sender for training completion
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize completion notifier
        
        Args:
            webhook_url: Slack webhook URL (can also be set via environment variable)
        """
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self.enabled = self.webhook_url is not None
        
        if not self.enabled:
            print("⚠️  Slack notifications disabled (SLACK_WEBHOOK_URL not set)")
    
    def send_completion_notification(self, results_file: str, experiment_name: str = None) -> bool:
        """
        Send training completion notification based on results file
        
        Args:
            results_file: Path to results.json file
            experiment_name: Override experiment name
        
        Returns:
            True if notification was sent successfully
        """
        if not self.enabled:
            print("Slack notifications not enabled")
            return False
        
        try:
            # Load results
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract key information
            config = results.get('config', {})
            summary = results.get('summary', {})
            duration = results.get('training_duration_seconds', 0)
            
            # Get experiment name
            exp_name = experiment_name or config.get('experiment', {}).get('name', 'Unknown')
            
            # Extract metrics
            final_accuracy = summary.get('final_avg_accuracy', 0)
            final_std = summary.get('final_std_accuracy', 0)
            best_accuracy = summary.get('best_test_accuracy', 0)
            total_rounds = summary.get('total_rounds', 0)
            total_comm_mb = summary.get('total_communication_mb', 0)
            
            # Format duration
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            duration_str = f"{hours}h {minutes}m {seconds}s"
            
            # Determine status
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
            
            # Create notification message
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{status_emoji} Machine Learning Training Complete"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Experiment:*\n{exp_name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Status:*\n{status_text}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Duration:*\n{duration_str}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Completed:*\n{datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        }
                    ]
                },
                {
                    "type": "divider"
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
                            "text": f"*Rounds:*\n{total_rounds}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Communication:*\n{total_comm_mb:.2f} MB"
                        }
                    ]
                }
            ]
            
            # Add privacy information if available
            privacy_info = summary.get('privacy')
            if privacy_info and privacy_info.get('privacy_enabled'):
                epsilon_spent = privacy_info.get('total_epsilon_spent', 0)
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"🔒 *Privacy Budget Spent:* ε = {epsilon_spent:.2f}"
                    }
                })
            
            # Add results file path
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"📁 Results: `{results_file}`"
                    }
                ]
            })
            
            # Send notification
            payload = {
                "text": f"Training complete: {exp_name} - {final_accuracy:.2f}% ({status_text})",
                "blocks": blocks
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            success = response.status_code == 200
            if success:
                print(f"✅ Completion notification sent to Slack")
            else:
                print(f"❌ Failed to send notification (status: {response.status_code})")
            
            return success
            
        except FileNotFoundError:
            print(f"❌ Results file not found: {results_file}")
            return False
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in results file: {results_file}")
            return False
        except Exception as e:
            print(f"❌ Error sending notification: {e}")
            return False
    
    def send_simple_message(self, message: str, experiment_name: str = "Unknown") -> bool:
        """
        Send a simple completion message
        
        Args:
            message: Message to send
            experiment_name: Name of experiment
        
        Returns:
            True if message was sent successfully
        """
        if not self.enabled:
            return False
        
        try:
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🎯 Training Notification"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Experiment:* {experiment_name}\n*Message:* {message}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                }
            ]
            
            payload = {
                "text": f"{experiment_name}: {message}",
                "blocks": blocks
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending simple message: {e}")
            return False


def notify_from_results(results_file: str, experiment_name: str = None):
    """
    Send notification based on results file
    
    Args:
        results_file: Path to results.json file
        experiment_name: Optional experiment name override
    """
    notifier = CompletionNotifier()
    notifier.send_completion_notification(results_file, experiment_name)


def notify_message(message: str, experiment_name: str = "Training"):
    """
    Send a simple notification message
    
    Args:
        message: Message to send
        experiment_name: Experiment name
    """
    notifier = CompletionNotifier()
    notifier.send_simple_message(message, experiment_name)


def main():
    """Command line interface for sending notifications"""
    parser = argparse.ArgumentParser(description='Send training completion notifications to Slack')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Results-based notification
    results_parser = subparsers.add_parser('results', help='Send notification from results file')
    results_parser.add_argument('results_file', help='Path to results.json file')
    results_parser.add_argument('--name', help='Override experiment name')
    
    # Simple message notification
    message_parser = subparsers.add_parser('message', help='Send simple message')
    message_parser.add_argument('message', help='Message to send')
    message_parser.add_argument('--name', default='Training', help='Experiment name')
    
    # Test notification
    test_parser = subparsers.add_parser('test', help='Send test notification')
    test_parser.add_argument('--name', default='Test', help='Experiment name')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check webhook URL
    if not os.environ.get('SLACK_WEBHOOK_URL'):
        print("❌ SLACK_WEBHOOK_URL environment variable not set")
        print("\nTo set it:")
        print("export SLACK_WEBHOOK_URL=\"https://hooks.slack.com/services/YOUR/WEBHOOK/URL\"")
        return
    
    if args.command == 'results':
        notify_from_results(args.results_file, args.name)
    elif args.command == 'message':
        notify_message(args.message, args.name)
    elif args.command == 'test':
        notifier = CompletionNotifier()
        success = notifier.send_simple_message(
            "Test notification from completion notifier", 
            args.name
        )
        print(f"Test notification: {'✅ Success' if success else '❌ Failed'}")


if __name__ == "__main__":
    main()