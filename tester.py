# realtime_chat_tester.py
"""
Real-Time Chat Testing Interface for Stage 2 Conversation Scoring
Simulates live chat conversations with real-time scoring updates
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scorer import EnhancedStage2ConversationScorer

class RealTimeChatTester:
    """Interactive chat testing interface"""
    
    def __init__(self):
        self.scorer = EnhancedStage2ConversationScorer()
        self.current_lead_id = None
        self.conversation_active = False
        
        # Colors for terminal output
        self.colors = {
            'header': '\033[95m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'red': '\033[91m',
            'bold': '\033[1m',
            'underline': '\033[4m',
            'end': '\033[0m'
        }
    
    def print_colored(self, text: str, color: str = 'end'):
        """Print colored text to terminal"""
        print(f"{self.colors.get(color, '')}{text}{self.colors['end']}")
    
    def print_header(self):
        """Print application header"""
        self.print_colored("=" * 80, 'cyan')
        self.print_colored("         REAL-TIME STAGE 2 CONVERSATION SCORING TESTER", 'bold')
        self.print_colored("=" * 80, 'cyan')
        print()
    
    def print_score_dashboard(self, result: Dict):
        """Print real-time scoring dashboard"""
        print("\n" + "-" * 60)
        self.print_colored("📊 REAL-TIME SCORING DASHBOARD", 'header')
        print("-" * 60)
        
        # Score information
        current_score = result.get('current_score', 0)
        stage1_score = result.get('stage1_score', 0)
        ml_score = result.get('ml_predicted_score', 0)
        score_change = result.get('score_change', 0)
        tier = result.get('tier', 'unknown').upper()
        
        # Color code the tier
        tier_color = 'red'
        if tier == 'HOT':
            tier_color = 'red'
        elif tier == 'WARM':
            tier_color = 'yellow'
        elif tier == 'COLD':
            tier_color = 'blue'
        
        print(f"Lead ID: {result.get('lead_id', 'N/A')}")
        print(f"Current Score: {current_score:.3f}")
        print(f"Stage 1 Score: {stage1_score:.3f}")
        print(f"ML Predicted: {ml_score:.3f}")
        print(f"Score Change: {score_change:+.3f}")
        self.print_colored(f"Lead Tier: {tier}", tier_color)
        
        # Message analysis
        analysis = result.get('message_analysis', {})
        if analysis:
            print(f"\nMessage Analysis:")
            print(f"  Sentiment: {analysis.get('sentiment_score', 0):.3f} ({analysis.get('sentiment_category', 'neutral')})")
            print(f"  Engagement: {analysis.get('engagement_score', 0):.3f}")
            print(f"  Message Length: {analysis.get('message_length', 0)} chars")
            print(f"  Questions Asked: {analysis.get('question_count', 0)}")
            
            signals = analysis.get('detected_signals', [])
            if signals:
                self.print_colored(f"  Detected Signals: {', '.join(signals)}", 'green')
        
        # Routing decision
        routing = result.get('routing_decision', {})
        if routing and routing.get('action') != 'continue':
            self.print_colored(f"\n🚨 ROUTING ACTION: {routing.get('action', '').upper()}", 'red')
            print(f"   Reason: {routing.get('reason', '')}")
            print(f"   Urgency: {routing.get('urgency', '')}")
            print(f"   Confidence: {routing.get('confidence', 0):.2f}")
        
        # Conversation metrics
        metrics = result.get('conversation_metrics', {})
        if metrics:
            print(f"\nConversation Metrics:")
            print(f"  Duration: {metrics.get('duration_minutes', 0):.1f} minutes")
            print(f"  Total Messages: {metrics.get('message_count', 0)}")
            print(f"  Engagement Rate: {metrics.get('engagement_rate', 0):.2f} msg/min")
            print(f"  Quality: {metrics.get('conversation_quality', 'unknown').title()}")
            print(f"  Trend: {metrics.get('engagement_trend', 'stable').title()}")
            
            # Signal breakdown
            signals = metrics.get('signal_breakdown', {})
            if any(signals.values()):
                print(f"  Signal Counts: +{signals.get('positive_signals', 0)}, " +
                      f"-{signals.get('negative_signals', 0)}, " +
                      f"booking:{signals.get('booking_indicators', 0)}, " +
                      f"urgency:{signals.get('urgency_indicators', 0)}")
        
        print("-" * 60)
    
    def start_new_conversation(self):
        """Start a new conversation for testing"""
        print()
        self.print_colored("🆕 STARTING NEW CONVERSATION", 'header')
        
        # Get conversation details
        while True:
            lead_id = input("Enter Lead ID (or 'random' for auto-generated): ").strip()
            if lead_id:
                if lead_id.lower() == 'random':
                    lead_id = f"TEST_{datetime.now().strftime('%H%M%S')}"
                break
            else:
                print("Please enter a valid Lead ID")
        
        while True:
            try:
                stage1_score = float(input("Enter Stage 1 Score (0.0 - 1.0): ").strip())
                if 0 <= stage1_score <= 1:
                    break
                else:
                    print("Score must be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")
        
        # Optional Stage 1 data
        stage1_data = {}
        budget_input = input("Enter customer budget (optional, press Enter to skip): ").strip()
        if budget_input.isdigit():
            stage1_data['budget'] = int(budget_input)
        
        location_input = input("Enter preferred location (optional): ").strip()
        if location_input:
            stage1_data['location'] = location_input
        
        # Start the conversation
        result = self.scorer.start_stage2_scoring(lead_id, stage1_score, stage1_data)
        
        if 'error' not in result:
            self.current_lead_id = lead_id
            self.conversation_active = True
            
            self.print_colored(f"✅ Conversation started for {lead_id}", 'green')
            print(f"Initial Score: {stage1_score}")
            print(f"Lead Tier: {result['tier'].upper()}")
            
            if result['routing_decision']['action'] != 'continue':
                self.print_colored(f"Initial Routing: {result['routing_decision']['action']}", 'yellow')
        else:
            self.print_colored(f"❌ Error: {result['error']}", 'red')
    
    def process_message(self):
        """Process a new message in the active conversation"""
        if not self.conversation_active or not self.current_lead_id:
            self.print_colored("❌ No active conversation. Start a new one first.", 'red')
            return
        
        print()
        self.print_colored("💬 SEND MESSAGE", 'header')
        
        # Get message input
        message = input("Customer message: ").strip()
        if not message:
            print("Empty message, skipping...")
            return
        
        # Process the message
        print("\nProcessing message...")
        start_time = time.time()
        
        result = self.scorer.process_conversation_message(
            self.current_lead_id, 
            message, 
            'lead'
        )
        
        processing_time = time.time() - start_time
        
        if 'error' not in result:
            self.print_score_dashboard(result)
            print(f"\n⚡ Processing Time: {processing_time:.3f} seconds")
        else:
            self.print_colored(f"❌ Error: {result['error']}", 'red')
    
    def add_agent_response(self):
        """Add an agent response to the conversation"""
        if not self.conversation_active or not self.current_lead_id:
            self.print_colored("❌ No active conversation. Start a new one first.", 'red')
            return
        
        print()
        self.print_colored("🤖 AGENT RESPONSE", 'header')
        
        agent_message = input("Agent response: ").strip()
        if not agent_message:
            print("Empty message, skipping...")
            return
        
        # Add agent message (won't affect scoring)
        result = self.scorer.process_conversation_message(
            self.current_lead_id,
            agent_message,
            'agent'
        )
        
        if 'error' not in result:
            self.print_colored("✅ Agent response recorded", 'green')
        else:
            self.print_colored(f"❌ Error: {result['error']}", 'red')
    
    def show_conversation_summary(self):
        """Show detailed conversation summary"""
        if not self.conversation_active or not self.current_lead_id:
            self.print_colored("❌ No active conversation.", 'red')
            return
        
        # Get active conversations to find current one
        active_conversations = self.scorer.get_active_conversations()
        current_conv = None
        
        for conv in active_conversations:
            if conv['lead_id'] == self.current_lead_id:
                current_conv = conv
                break
        
        if not current_conv:
            self.print_colored("❌ Conversation not found.", 'red')
            return
        
        print()
        self.print_colored("📋 DETAILED CONVERSATION SUMMARY", 'header')
        print("=" * 60)
        
        print(f"Lead ID: {current_conv['lead_id']}")
        print(f"Duration: {current_conv['duration_minutes']:.1f} minutes")
        print(f"Score Journey: {current_conv['stage1_score']:.3f} → {current_conv['current_score']:.3f}")
        print(f"ML Predicted: {current_conv['ml_predicted_score']:.3f}")
        print(f"Current Tier: {current_conv['tier'].upper()}")
        print(f"Messages: {current_conv['message_count']}")
        print(f"Engagement Rate: {current_conv['engagement_rate']:.2f} msg/min")
        print(f"Quality: {current_conv['conversation_quality'].title()}")
        print(f"Trend: {current_conv['engagement_trend'].title()}")
        
        # Signal breakdown
        signals = current_conv['signal_breakdown']
        print(f"\nSignal Analysis:")
        print(f"  Positive Signals: {signals['positive_signals']}")
        print(f"  Negative Signals: {signals['negative_signals']}")
        print(f"  Booking Indicators: {signals['booking_indicators']}")
        print(f"  Commitment Indicators: {signals['commitment_indicators']}")
        print(f"  Urgency Indicators: {signals['urgency_indicators']}")
        print(f"  Questions Asked: {signals['qualification_questions']}")
        print(f"  Budget Disclosed: {'Yes' if current_conv['budget_disclosed'] else 'No'}")
        
        print(f"\nSentiment: {current_conv['avg_sentiment']:.3f}")
        print(f"Engagement: {current_conv['avg_engagement']:.3f}")
    
    def end_conversation(self):
        """End the current conversation"""
        if not self.conversation_active or not self.current_lead_id:
            self.print_colored("❌ No active conversation to end.", 'red')
            return
        
        print()
        self.print_colored("🏁 END CONVERSATION", 'header')
        
        # Ask for outcome
        print("Select outcome:")
        outcomes = [
            "completed", "viewing_scheduled", "lease_signed", 
            "lost_to_competitor", "budget_mismatch", "requirements_not_met",
            "timeout_inactive", "customer_not_interested"
        ]
        
        for i, outcome in enumerate(outcomes, 1):
            print(f"{i}. {outcome}")
        
        while True:
            try:
                choice = int(input("Enter outcome number: ").strip())
                if 1 <= choice <= len(outcomes):
                    selected_outcome = outcomes[choice - 1]
                    break
                else:
                    print("Invalid choice")
            except ValueError:
                print("Please enter a number")
        
        # End the conversation
        summary = self.scorer.end_conversation(self.current_lead_id, selected_outcome)
        
        if 'error' not in summary:
            print("\n" + "=" * 60)
            self.print_colored("📊 FINAL CONVERSATION SUMMARY", 'header')
            print("=" * 60)
            
            print(f"Lead ID: {summary['lead_id']}")
            print(f"Duration: {summary['session_duration_minutes']:.1f} minutes")
            print(f"Score Journey: {summary['stage1_score']:.3f} → {summary['final_score']:.3f}")
            print(f"Score Improvement: {summary['score_improvement']:+.3f} ({summary['percentage_improvement']:+.1f}%)")
            print(f"ML Predicted Final: {summary['ml_predicted_score']:.3f}")
            print(f"Final Tier: {summary['final_tier'].upper()}")
            print(f"Outcome: {summary['outcome']}")
            
            # Reset state
            self.current_lead_id = None
            self.conversation_active = False
            
            self.print_colored("✅ Conversation ended and saved to database", 'green')
        else:
            self.print_colored(f"❌ Error: {summary['error']}", 'red')
    
    def show_all_active_conversations(self):
        """Show all active conversations"""
        active = self.scorer.get_active_conversations()
        
        if not active:
            print("No active conversations")
            return
        
        print()
        self.print_colored("👥 ALL ACTIVE CONVERSATIONS", 'header')
        print("=" * 80)
        
        for i, conv in enumerate(active, 1):
            tier_color = {'hot': 'red', 'warm': 'yellow', 'cold': 'blue', 'inactive': 'red'}.get(conv['tier'], 'end')
            
            print(f"{i}. Lead: {conv['lead_id']}")
            print(f"   Score: {conv['stage1_score']:.3f} → {conv['current_score']:.3f}")
            self.print_colored(f"   Tier: {conv['tier'].upper()}", tier_color)
            print(f"   Duration: {conv['duration_minutes']:.1f} min, Messages: {conv['message_count']}")
            print(f"   Quality: {conv['conversation_quality'].title()}")
            print()
    
    def run_interactive_mode(self):
        """Run the interactive testing interface"""
        self.print_header()
        
        while True:
            print()
            self.print_colored("🔧 TESTING OPTIONS", 'cyan')
            print("1. Start New Conversation")
            print("2. Send Customer Message")
            print("3. Add Agent Response")
            print("4. Show Conversation Summary")
            print("5. End Current Conversation")
            print("6. Show All Active Conversations")
            print("7. Quick Test Conversation")
            print("8. Exit")
            
            if self.conversation_active:
                self.print_colored(f"Active: {self.current_lead_id}", 'green')
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                self.start_new_conversation()
            elif choice == '2':
                self.process_message()
            elif choice == '3':
                self.add_agent_response()
            elif choice == '4':
                self.show_conversation_summary()
            elif choice == '5':
                self.end_conversation()
            elif choice == '6':
                self.show_all_active_conversations()
            elif choice == '7':
                self.run_quick_test()
            elif choice == '8':
                print("\nExiting...")
                if self.conversation_active:
                    print("Note: Active conversation will remain in system")
                break
            else:
                print("Invalid choice, please try again")
    
    def run_quick_test(self):
        """Run a quick test conversation with predefined messages"""
        print()
        self.print_colored("⚡ QUICK TEST CONVERSATION", 'header')
        
        # Start conversation
        test_lead_id = f"QUICKTEST_{datetime.now().strftime('%H%M%S')}"
        result = self.scorer.start_stage2_scoring(test_lead_id, 0.6, {'budget': 1200})
        
        if 'error' in result:
            self.print_colored(f"❌ Error starting conversation: {result['error']}", 'red')
            return
        
        self.current_lead_id = test_lead_id
        self.conversation_active = True
        
        # Predefined test messages
        test_messages = [
            "Hi, I'm interested in the property listing I saw online",
            "The location looks perfect for my needs. What's the availability?",
            "I need to move in urgently, within the next two weeks",
            "My budget is around 1500 per month, is that workable?",
            "This looks exactly like what I've been searching for!",
            "Can we schedule a viewing as soon as possible?",
            "I'm really serious about this, when can we meet?"
        ]
        
        print(f"\nRunning quick test with {len(test_messages)} messages...")
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n--- Message {i} ---")
            print(f"Customer: {message}")
            
            result = self.scorer.process_conversation_message(test_lead_id, message, 'lead')
            
            if 'error' not in result:
                # Show abbreviated dashboard
                score = result['current_score']
                change = result['score_change']
                tier = result['tier']
                routing = result['routing_decision']['action']
                
                print(f"Score: {score:.3f} ({change:+.3f}) | Tier: {tier.upper()} | Action: {routing}")
                
                analysis = result.get('message_analysis', {})
                signals = analysis.get('detected_signals', [])
                if signals:
                    print(f"Signals: {', '.join(signals)}")
            
            time.sleep(0.5)  # Brief pause for readability
        
        # Show final summary
        print()
        self.print_colored("📊 QUICK TEST COMPLETE", 'green')
        self.show_conversation_summary()
        
        # Ask if user wants to continue with this conversation or end it
        continue_choice = input("\nContinue with this conversation? (y/n): ").strip().lower()
        if continue_choice != 'y':
            self.end_conversation()

def main():
    """Main function to run the real-time chat tester"""
    try:
        tester = RealTimeChatTester()
        tester.run_interactive_mode()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()