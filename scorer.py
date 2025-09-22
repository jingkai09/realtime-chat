# scorer.py
"""
Enhanced Stage 2 Conversation Scoring System
Integrates ML models, conversation analysis, and real-time scoring
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Import your existing modules
from models import Stage2MLModels
from analyzer import ConversationAnalyzer

@dataclass
class ConversationState:
    """Tracks the state of an active conversation"""
    lead_id: str
    stage1_score: float
    current_score: float
    ml_predicted_score: float
    start_time: datetime
    last_activity: datetime
    message_count: int
    messages: List[Dict]
    analyses: List[Dict]
    stage1_data: Dict
    routing_history: List[Dict]

class EnhancedStage2ConversationScorer:
    """Enhanced Stage 2 conversation scoring with ML integration"""
    
    def __init__(self, db_path="stage2_conversations.db", enable_logging=True):
        self.db_path = db_path
        self.active_conversations = {}
        
        # Initialize components
        self.ml_models = Stage2MLModels()
        self.analyzer = ConversationAnalyzer()
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        
        # Initialize database
        self._init_database()
        
        # Load configuration
        self.config = self._load_config()
        
        self.logger.info("Enhanced Stage 2 Conversation Scorer initialized")
    
    def _load_config(self) -> Dict:
        """Load scoring configuration based on Stage 1 insights"""
        return {
            'thresholds': {
                'hot_lead': 0.75,
                'warm_lead': 0.55,
                'cold_lead': 0.35,
                'inactive_threshold': 0.25
            },
            'score_weights': {
                'ml_prediction': 0.4,
                'rule_based': 0.35,
                'stage1_base': 0.25
            },
            'signal_weights': {
                'urgency': 0.20,
                'booking_intent': 0.18,
                'commitment_indicators': 0.15,
                'positive_signals': 0.12,
                'qualification_questions': 0.10,
                'budget_disclosure': 0.08,
                'objections': -0.15,
                'comparison_shopping': -0.10
            },
            'time_decay': {
                'decay_rate': 0.02,
                'max_inactive_hours': 24,
                'rapid_decay_threshold': 4
            },
            'engagement_multipliers': {
                'high_frequency': 1.15,
                'recent_activity': 1.10,
                'long_messages': 1.05,
                'multiple_questions': 1.08
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for conversation tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id TEXT UNIQUE,
                stage1_score REAL,
                final_score REAL,
                ml_predicted_score REAL,
                start_time TEXT,
                end_time TEXT,
                duration_minutes REAL,
                message_count INTEGER,
                outcome TEXT,
                stage1_data TEXT,
                conversation_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id TEXT,
                message_text TEXT,
                sender TEXT,
                timestamp TEXT,
                analysis_data TEXT,
                score_before REAL,
                score_after REAL,
                routing_decision TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS routing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id TEXT,
                event_type TEXT,
                from_handler TEXT,
                to_handler TEXT,
                trigger_reason TEXT,
                score_at_event REAL,
                confidence REAL,
                timestamp TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info("Database initialized successfully")
    
    def start_stage2_scoring(self, lead_id: str, stage1_score: float, stage1_data: Dict = None) -> Dict:
        """Initialize Stage 2 scoring for a new conversation"""
        
        if lead_id in self.active_conversations:
            self.logger.warning(f"Conversation {lead_id} already active")
            return {'error': 'Conversation already active'}
        
        conversation_state = ConversationState(
            lead_id=lead_id,
            stage1_score=stage1_score,
            current_score=stage1_score,
            ml_predicted_score=stage1_score,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            message_count=0,
            messages=[],
            analyses=[],
            stage1_data=stage1_data or {},
            routing_history=[]
        )
        
        self.active_conversations[lead_id] = conversation_state
        self._log_conversation_start(conversation_state)
        
        routing_decision = self._make_routing_decision(lead_id, stage1_score, "conversation_start")
        
        self.logger.info(f"Started Stage 2 scoring for {lead_id}, initial score: {stage1_score}")
        
        return {
            'lead_id': lead_id,
            'current_score': stage1_score,
            'ml_predicted_score': stage1_score,
            'tier': self._get_lead_tier(stage1_score),
            'routing_decision': routing_decision,
            'status': 'active'
        }
    
    def process_conversation_message(self, lead_id: str, message: str, sender: str = 'lead') -> Dict:
        """Process a new message and update conversation scoring"""
        
        if lead_id not in self.active_conversations:
            return {'error': 'Conversation not found or not active'}
        
        conversation = self.active_conversations[lead_id]
        
        if sender == 'lead':
            message_history = [msg['text'] for msg in conversation.messages if msg['sender'] == 'lead']
            analysis = self.analyzer.analyze_message(message, message_history)
            
            scores = self._calculate_updated_scores(conversation, analysis)
            
            conversation.current_score = scores['final_score']
            conversation.ml_predicted_score = scores['ml_score']
            conversation.last_activity = datetime.now()
            conversation.message_count += 1
            
            message_data = {
                'text': message,
                'sender': sender,
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis
            }
            conversation.messages.append(message_data)
            conversation.analyses.append(analysis)
            
            routing_decision = self._make_routing_decision(lead_id, scores['final_score'], "message_processed")
            
            self._log_message(lead_id, message, sender, analysis, 
                            conversation.stage1_score, scores['final_score'], routing_decision)
            
            self.logger.info(f"Processed message for {lead_id}: {conversation.stage1_score:.3f} → {scores['final_score']:.3f}")
            
            return {
                'lead_id': lead_id,
                'message': message,
                'sender': sender,
                'stage1_score': conversation.stage1_score,
                'current_score': scores['final_score'],
                'ml_predicted_score': scores['ml_score'],
                'score_change': scores['final_score'] - conversation.stage1_score,
                'tier': self._get_lead_tier(scores['final_score']),
                'message_analysis': self._format_analysis_output(analysis),
                'routing_decision': routing_decision,
                'conversation_metrics': self._get_conversation_summary(lead_id)
            }
        
        else:
            message_data = {
                'text': message,
                'sender': sender,
                'timestamp': datetime.now().isoformat(),
                'analysis': None
            }
            conversation.messages.append(message_data)
            conversation.last_activity = datetime.now()
            
            return {
                'lead_id': lead_id,
                'message': message,
                'sender': sender,
                'status': 'message_stored'
            }
    
    def _calculate_updated_scores(self, conversation: ConversationState, analysis: Dict) -> Dict:
        """Calculate updated scores using ML and rule-based approaches"""
        
        ml_features = self._extract_ml_features(conversation, analysis)
        ml_prediction = self.ml_models.predict_conversation_score(ml_features)
        ml_score = ml_prediction['predicted_score']
        
        rule_based_adjustment = self._calculate_rule_based_score(conversation, analysis)
        time_decay = self._calculate_time_decay(conversation)
        
        weights = self.config['score_weights']
        final_score = (
            conversation.stage1_score * weights['stage1_base'] +
            ml_score * weights['ml_prediction'] +
            rule_based_adjustment * weights['rule_based']
        ) * time_decay
        
        final_score *= self._calculate_engagement_multiplier(conversation, analysis)
        final_score = np.clip(final_score, 0, 1)
        
        return {
            'final_score': final_score,
            'ml_score': ml_score,
            'rule_based_adjustment': rule_based_adjustment,
            'time_decay': time_decay,
            'ml_confidence': ml_prediction.get('confidence', 0.5)
        }
    
    def _extract_ml_features(self, conversation: ConversationState, analysis: Dict) -> Dict:
        """Extract features for ML model prediction"""
        
        total_messages = len([msg for msg in conversation.messages if msg['sender'] == 'lead'])
        duration_minutes = (datetime.now() - conversation.start_time).total_seconds() / 60
        
        all_signals = []
        total_sentiment = 0
        total_engagement = 0
        question_count = 0
        
        for msg_analysis in conversation.analyses:
            if msg_analysis:
                all_signals.extend(msg_analysis.get('detected_signals', []))
                total_sentiment += msg_analysis.get('sentiment_analysis', {}).get('sentiment_score', 0)
                total_engagement += msg_analysis.get('engagement_score', 0)
                question_count += msg_analysis.get('message_metrics', {}).get('question_marks', 0)
        
        signal_counts = {
            'positive_signals': all_signals.count('positive_signals'),
            'negative_signals': all_signals.count('objections'),
            'urgency_signals': all_signals.count('urgency'),
            'booking_signals': all_signals.count('booking_intent'),
            'commitment_signals': all_signals.count('commitment_indicators')
        }
        
        avg_sentiment = total_sentiment / max(len(conversation.analyses), 1)
        avg_engagement = total_engagement / max(len(conversation.analyses), 1)
        avg_message_length = np.mean([
            len(msg['text']) for msg in conversation.messages 
            if msg['sender'] == 'lead'
        ]) if total_messages > 0 else 0
        
        return {
            'message_count': total_messages,
            'avg_message_length': avg_message_length,
            'question_count': question_count,
            'positive_signals': signal_counts['positive_signals'],
            'negative_signals': signal_counts['negative_signals'],
            'urgency_signals': signal_counts['urgency_signals'],
            'booking_signals': signal_counts['booking_signals'],
            'commitment_signals': signal_counts['commitment_signals'],
            'engagement_rate': total_messages / max(duration_minutes, 1),
            'sentiment_score': avg_sentiment,
            'signal_variety': len(set(all_signals)),
            'conversation_duration': duration_minutes
        }
    
    def _calculate_rule_based_score(self, conversation: ConversationState, analysis: Dict) -> float:
        """Calculate rule-based score adjustment"""
        
        base_score = conversation.stage1_score
        adjustment = 0
        
        detected_signals = analysis.get('detected_signals', [])
        for signal in detected_signals:
            weight = self.config['signal_weights'].get(signal, 0)
            adjustment += weight
        
        if 'budget_disclosure' in detected_signals:
            extracted_budget = analysis.get('extracted_values', {}).get('disclosed_budget')
            if extracted_budget:
                stage1_budget = conversation.stage1_data.get('budget', 0)
                if stage1_budget > 0:
                    budget_flexibility = extracted_budget / stage1_budget
                    if budget_flexibility >= 1.2:
                        adjustment += 0.10
        
        sentiment_score = analysis.get('sentiment_analysis', {}).get('sentiment_score', 0)
        adjustment += sentiment_score * 0.05
        
        engagement_score = analysis.get('engagement_score', 0)
        if engagement_score > 0.7:
            adjustment += 0.05
        
        return base_score + adjustment
    
    def _calculate_time_decay(self, conversation: ConversationState) -> float:
        """Calculate time-based score decay"""
        
        hours_since_start = (datetime.now() - conversation.start_time).total_seconds() / 3600
        hours_since_activity = (datetime.now() - conversation.last_activity).total_seconds() / 3600
        
        decay_factor = 1.0
        decay_rate = self.config['time_decay']['decay_rate']
        
        if hours_since_activity > self.config['time_decay']['rapid_decay_threshold']:
            decay_factor *= (1 - decay_rate) ** (hours_since_activity - 4)
        
        age_decay_rate = decay_rate * 0.5
        decay_factor *= (1 - age_decay_rate) ** hours_since_start
        
        return max(decay_factor, 0.1)
    
    def _calculate_engagement_multiplier(self, conversation: ConversationState, analysis: Dict) -> float:
        """Calculate engagement-based score multiplier"""
        
        multiplier = 1.0
        multipliers = self.config['engagement_multipliers']
        
        if conversation.message_count >= 3:
            multiplier *= multipliers['high_frequency']
        
        hours_since_activity = (datetime.now() - conversation.last_activity).total_seconds() / 3600
        if hours_since_activity < 1:
            multiplier *= multipliers['recent_activity']
        
        message_length = analysis.get('message_metrics', {}).get('length', 0)
        if message_length > 50:
            multiplier *= multipliers['long_messages']
        
        questions = analysis.get('message_metrics', {}).get('question_marks', 0)
        if questions >= 2:
            multiplier *= multipliers['multiple_questions']
        
        return min(multiplier, 1.5)
    
    def _make_routing_decision(self, lead_id: str, current_score: float, trigger: str) -> Dict:
        """Make routing decision based on current score and context"""
        
        thresholds = self.config['thresholds']
        conversation = self.active_conversations.get(lead_id)
        
        current_tier = self._get_lead_tier(current_score)
        
        routing_action = 'continue'
        reason = 'score_within_normal_range'
        urgency = 'low'
        confidence = 0.8
        
        if current_score >= thresholds['hot_lead']:
            routing_action = 'escalate_to_agent'
            reason = 'high_conversion_probability'
            urgency = 'high'
            confidence = 0.9
        elif current_score <= thresholds['cold_lead']:
            routing_action = 'transfer_to_ai'
            reason = 'low_engagement_detected'
            urgency = 'medium'
            confidence = 0.8
        elif current_score <= thresholds['inactive_threshold']:
            routing_action = 'mark_inactive'
            reason = 'very_low_engagement'
            urgency = 'low'
            confidence = 0.9
        
        if conversation:
            recent_signals = []
            if conversation.analyses:
                recent_signals = conversation.analyses[-1].get('detected_signals', [])
            
            if 'booking_intent' in recent_signals and current_score > 0.6:
                routing_action = 'escalate_to_agent'
                reason = 'booking_intent_detected'
                urgency = 'high'
                confidence = 0.95
            
            if 'objections' in recent_signals and current_score > 0.4:
                routing_action = 'suggest_agent_assistance'
                reason = 'objections_need_human_touch'
                urgency = 'medium'
                confidence = 0.7
        
        if routing_action != 'continue':
            self._log_routing_event(lead_id, routing_action, reason, current_score, confidence)
        
        return {
            'action': routing_action,
            'reason': reason,
            'urgency': urgency,
            'confidence': confidence,
            'current_tier': current_tier,
            'score': current_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_lead_tier(self, score: float) -> str:
        """Convert score to lead tier"""
        if score >= 0.75:
            return 'hot'
        elif score >= 0.55:
            return 'warm'
        elif score >= 0.35:
            return 'cold'
        else:
            return 'inactive'
    
    def _format_analysis_output(self, analysis: Dict) -> Dict:
        """Format analysis for API response"""
        return {
            'detected_signals': analysis.get('detected_signals', []),
            'sentiment_score': analysis.get('sentiment_analysis', {}).get('sentiment_score', 0),
            'sentiment_category': analysis.get('sentiment_analysis', {}).get('sentiment_category', 'neutral'),
            'engagement_score': analysis.get('engagement_score', 0),
            'extracted_values': analysis.get('extracted_values', {}),
            'message_length': analysis.get('message_metrics', {}).get('length', 0),
            'question_count': analysis.get('message_metrics', {}).get('question_marks', 0)
        }
    
    def _get_conversation_summary(self, lead_id: str) -> Dict:
        """Get summary metrics for active conversation"""
        
        if lead_id not in self.active_conversations:
            return {}
        
        conversation = self.active_conversations[lead_id]
        duration_minutes = (datetime.now() - conversation.start_time).total_seconds() / 60
        lead_messages = [msg for msg in conversation.messages if msg['sender'] == 'lead']
        
        all_signals = []
        sentiments = []
        engagements = []
        
        for analysis in conversation.analyses:
            if analysis:
                all_signals.extend(analysis.get('detected_signals', []))
                sentiments.append(analysis.get('sentiment_analysis', {}).get('sentiment_score', 0))
                engagements.append(analysis.get('engagement_score', 0))
        
        signal_breakdown = {
            'positive_signals': all_signals.count('positive_signals'),
            'negative_signals': all_signals.count('objections'),
            'booking_indicators': all_signals.count('booking_intent'),
            'commitment_indicators': all_signals.count('commitment_indicators'),
            'urgency_indicators': all_signals.count('urgency'),
            'qualification_questions': all_signals.count('qualification_questions')
        }
        
        avg_engagement = np.mean(engagements) if engagements else 0
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        quality = 'good' if avg_engagement > 0.6 and avg_sentiment > 0 else 'average'
        if avg_engagement > 0.8 and avg_sentiment > 0.3:
            quality = 'excellent'
        elif avg_engagement < 0.3 or avg_sentiment < -0.2:
            quality = 'poor'
        
        if len(engagements) >= 3:
            recent_engagement = np.mean(engagements[-2:])
            earlier_engagement = np.mean(engagements[:-2])
            trend = 'increasing' if recent_engagement > earlier_engagement else 'decreasing'
        else:
            trend = 'stable'
        
        budget_disclosed = any('budget_disclosure' in analysis.get('detected_signals', []) 
                             for analysis in conversation.analyses if analysis)
        
        return {
            'duration_minutes': round(duration_minutes, 1),
            'message_count': len(lead_messages),
            'engagement_rate': round(len(lead_messages) / max(duration_minutes, 1), 2),
            'avg_sentiment': round(avg_sentiment, 3),
            'avg_engagement': round(avg_engagement, 3),
            'signal_breakdown': signal_breakdown,
            'conversation_quality': quality,
            'engagement_trend': trend,
            'budget_disclosed': budget_disclosed
        }
    
    def get_active_conversations(self) -> List[Dict]:
        """Get summary of all active conversations"""
        
        active_list = []
        
        for lead_id, conversation in self.active_conversations.items():
            summary = self._get_conversation_summary(lead_id)
            summary.update({
                'lead_id': lead_id,
                'stage1_score': conversation.stage1_score,
                'current_score': conversation.current_score,
                'ml_predicted_score': conversation.ml_predicted_score,
                'tier': self._get_lead_tier(conversation.current_score),
                'start_time': conversation.start_time.isoformat(),
                'last_activity': conversation.last_activity.isoformat()
            })
            active_list.append(summary)
        
        return sorted(active_list, key=lambda x: x['current_score'], reverse=True)
    
    def end_conversation(self, lead_id: str, outcome: str = 'completed') -> Dict:
        """End conversation and generate final summary"""
        
        if lead_id not in self.active_conversations:
            return {'error': 'Conversation not found'}
        
        conversation = self.active_conversations[lead_id]
        duration_minutes = (datetime.now() - conversation.start_time).total_seconds() / 60
        score_improvement = conversation.current_score - conversation.stage1_score
        percentage_improvement = (score_improvement / conversation.stage1_score * 100) if conversation.stage1_score > 0 else 0
        
        final_summary = {
            'lead_id': lead_id,
            'stage1_score': conversation.stage1_score,
            'final_score': conversation.current_score,
            'ml_predicted_score': conversation.ml_predicted_score,
            'score_improvement': round(score_improvement, 4),
            'percentage_improvement': round(percentage_improvement, 1),
            'session_duration_minutes': round(duration_minutes, 1),
            'final_tier': self._get_lead_tier(conversation.current_score),
            'outcome': outcome,
            'conversation_summary': self._get_conversation_summary(lead_id)
        }
        
        self._log_conversation_end(conversation, final_summary)
        del self.active_conversations[lead_id]
        
        self.logger.info(f"Ended conversation {lead_id}: {conversation.stage1_score:.3f} → {conversation.current_score:.3f}")
        
        return final_summary
    
    def _log_conversation_start(self, conversation: ConversationState):
        """Log conversation start to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO conversations 
            (lead_id, stage1_score, start_time, stage1_data)
            VALUES (?, ?, ?, ?)
        ''', (
            conversation.lead_id,
            conversation.stage1_score,
            conversation.start_time.isoformat(),
            json.dumps(conversation.stage1_data)
        ))
        
        conn.commit()
        conn.close()
    
    def _log_message(self, lead_id: str, message: str, sender: str, analysis: Dict, 
                    score_before: float, score_after: float, routing_decision: Dict):
        """Log message to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages 
            (lead_id, message_text, sender, timestamp, analysis_data, 
             score_before, score_after, routing_decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            lead_id, message, sender, datetime.now().isoformat(),
            json.dumps(analysis), score_before, score_after,
            json.dumps(routing_decision)
        ))
        
        conn.commit()
        conn.close()
    
    def _log_routing_event(self, lead_id: str, action: str, reason: str, 
                          score: float, confidence: float):
        """Log routing event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO routing_events 
            (lead_id, event_type, trigger_reason, score_at_event, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            lead_id, action, reason, score, confidence, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _log_conversation_end(self, conversation: ConversationState, summary: Dict):
        """Log conversation end to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE conversations SET
            final_score = ?, ml_predicted_score = ?, end_time = ?,
            duration_minutes = ?, message_count = ?, outcome = ?,
            conversation_data = ?
            WHERE lead_id = ?
        ''', (
            summary['final_score'],
            summary['ml_predicted_score'],
            datetime.now().isoformat(),
            summary['session_duration_minutes'],
            len(conversation.messages),
            summary['outcome'],
            json.dumps({
                'messages': conversation.messages,
                'analyses': conversation.analyses,
                'routing_history': conversation.routing_history
            }),
            conversation.lead_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_analytics_dashboard(self, days: int = 30) -> Dict:
        """Generate analytics dashboard data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Date filter
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Conversation metrics
        cursor.execute('''
            SELECT COUNT(*), AVG(final_score), AVG(duration_minutes), AVG(message_count)
            FROM conversations 
            WHERE created_at >= ? AND end_time IS NOT NULL
        ''', (since_date,))
        
        conv_metrics = cursor.fetchone()
        
        # Score improvement distribution
        cursor.execute('''
            SELECT (final_score - stage1_score) as improvement
            FROM conversations 
            WHERE created_at >= ? AND end_time IS NOT NULL
        ''', (since_date,))
        
        improvements = [row[0] for row in cursor.fetchall()]
        
        # Outcome distribution
        cursor.execute('''
            SELECT outcome, COUNT(*) 
            FROM conversations 
            WHERE created_at >= ? AND end_time IS NOT NULL
            GROUP BY outcome
        ''', (since_date,))
        
        outcomes = dict(cursor.fetchall())
        
        # Routing events
        cursor.execute('''
            SELECT event_type, COUNT(*) 
            FROM routing_events 
            WHERE created_at >= ?
            GROUP BY event_type
        ''', (since_date,))
        
        routing_events = dict(cursor.fetchall())
        
        # Score tier distribution
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN final_score >= 0.75 THEN 'hot'
                    WHEN final_score >= 0.55 THEN 'warm'
                    WHEN final_score >= 0.35 THEN 'cold'
                    ELSE 'inactive'
                END as tier,
                COUNT(*)
            FROM conversations 
            WHERE created_at >= ? AND end_time IS NOT NULL
            GROUP BY tier
        ''', (since_date,))
        
        tier_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        # Calculate statistics
        total_conversations = conv_metrics[0] if conv_metrics[0] else 0
        avg_final_score = conv_metrics[1] if conv_metrics[1] else 0
        avg_duration = conv_metrics[2] if conv_metrics[2] else 0
        avg_messages = conv_metrics[3] if conv_metrics[3] else 0
        
        improvement_stats = {}
        if improvements:
            improvement_stats = {
                'mean': np.mean(improvements),
                'median': np.median(improvements),
                'positive_improvements': sum(1 for x in improvements if x > 0),
                'negative_improvements': sum(1 for x in improvements if x < 0),
                'improvement_rate': sum(1 for x in improvements if x > 0) / len(improvements) * 100
            }
        
        return {
            'period_days': days,
            'total_conversations': total_conversations,
            'avg_final_score': round(avg_final_score, 3),
            'avg_duration_minutes': round(avg_duration, 1),
            'avg_messages_per_conversation': round(avg_messages, 1),
            'score_improvements': improvement_stats,
            'outcome_distribution': outcomes,
            'routing_events': routing_events,
            'tier_distribution': tier_distribution,
            'active_conversations': len(self.active_conversations),
            'generated_at': datetime.now().isoformat()
        }
    
    def cleanup_inactive_conversations(self, max_age_hours: int = 24) -> Dict:
        """Clean up conversations that have been inactive too long"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_up = []
        
        for lead_id, conversation in list(self.active_conversations.items()):
            if conversation.last_activity < cutoff_time:
                # End the conversation with timeout outcome
                final_summary = self.end_conversation(lead_id, 'timeout_inactive')
                cleaned_up.append({
                    'lead_id': lead_id,
                    'final_score': final_summary['final_score'],
                    'duration_hours': final_summary['session_duration_minutes'] / 60
                })
        
        self.logger.info(f"Cleaned up {len(cleaned_up)} inactive conversations")
        
        return {
            'cleaned_up_count': len(cleaned_up),
            'cleaned_conversations': cleaned_up,
            'remaining_active': len(self.active_conversations)
        }