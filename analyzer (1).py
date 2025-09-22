# analyzer.py
"""
Message Analysis and Signal Detection for Stage 2 Conversation Scoring
"""

import re
import numpy as np
from typing import Dict, List
from datetime import datetime

class ConversationAnalyzer:
    """Analyzes conversation messages and detects intent signals"""
    
    def __init__(self):
        # Enhanced intent patterns with more comprehensive coverage
        self.intent_patterns = {
            'booking_intent': [
                r'want to book', r'book.*viewing', r'schedule.*viewing',
                r'arrange.*visit', r'when can.*view', r'available.*viewing',
                r'make.*appointment', r'set.*time', r'confirm.*booking',
                r'reserve.*viewing', r'book.*slot', r'viewing.*appointment',
                r'can we meet', r'let.*schedule', r'arrange.*meeting'
            ],
            'urgency': [
                r'urgent', r'asap', r'immediately', r'this week', r'soon',
                r'need.*quick', r'time.*sensitive', r'rush', r'emergency',
                r'today', r'tomorrow', r'right now', r'right away',
                r'need.*now', r'can.*today', r'moving.*soon'
            ],
            'budget_disclosure': [
                r'budget.*(\d+)', r'afford.*(\d+)', r'(\d+).*budget',
                r'price.*range.*(\d+)', r'up to.*(\d+)', r'maximum.*(\d+)',
                r'spend.*(\d+)', r'paying.*(\d+)', r'around.*(\d+)',
                r'(\d+).*month', r'monthly.*(\d+)', r'rent.*(\d+)'
            ],
            'positive_signals': [
                r'interested', r'sounds good', r'perfect', r'exactly',
                r'yes.*definitely', r'looks great', r'love it', r'amazing',
                r'wonderful', r'excellent', r'fantastic', r'ideal',
                r'just what.*looking', r'this is.*perfect', r'really like',
                r'impressive', r'beautiful', r'stunning'
            ],
            'objections': [
                r'too expensive', r'not.*interested', r'found.*another',
                r'changed.*mind', r'not.*sure', r'think.*about',
                r'maybe.*later', r'need.*time', r'hesitant',
                r'overpriced', r'out of.*budget', r'cannot.*afford',
                r'looking.*elsewhere', r'not.*right'
            ],
            'qualification_questions': [
                r'what.*facilities', r'nearby.*amenities', r'transportation',
                r'parking.*available', r'furnished', r'utilities.*included',
                r'lease.*terms', r'deposit.*required', r'move.*date',
                r'pets.*allowed', r'what.*included', r'gym.*available',
                r'swimming.*pool', r'security', r'maintenance'
            ],
            'commitment_indicators': [
                r'ready.*move', r'serious.*about', r'definitely.*want',
                r'need.*place', r'must.*have', r'looking.*immediately',
                r'committed.*to', r'decided.*on', r'ready.*sign',
                r'when.*move.*in', r'ready.*proceed', r'let.*do.*this'
            ],
            'comparison_shopping': [
                r'other.*options', r'comparing.*places', r'seen.*others',
                r'shopping.*around', r'looking.*elsewhere', r'alternatives',
                r'another.*place', r'different.*property', r'considering.*other'
            ]
        }
        
        # Sentiment keywords for simple rule-based sentiment
        self.positive_keywords = [
            'love', 'like', 'great', 'good', 'excellent', 'perfect', 'amazing',
            'wonderful', 'fantastic', 'beautiful', 'nice', 'awesome', 'cool',
            'interested', 'excited', 'happy', 'pleased', 'satisfied'
        ]
        
        self.negative_keywords = [
            'hate', 'dislike', 'bad', 'terrible', 'awful', 'horrible',
            'disappointed', 'frustrated', 'annoyed', 'upset', 'angry',
            'expensive', 'overpriced', 'cheap', 'poor', 'not interested'
        ]
    
    def analyze_message(self, message: str, message_history: List[str] = None) -> Dict:
        """
        Comprehensive message analysis
        
        Args:
            message: Current message to analyze
            message_history: Previous messages for context
            
        Returns:
            Dictionary with analysis results
        """
        message_lower = message.lower()
        
        # Signal detection
        detected_signals = []
        signal_counts = {}
        extracted_values = {}
        
        # Detect all signal types
        for signal_type, patterns in self.intent_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, message_lower)
                count += len(matches)
                
                # Extract budget values
                if signal_type == 'budget_disclosure' and matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            # Regex group matches
                            numbers = [m for m in match if m.isdigit()]
                        else:
                            numbers = re.findall(r'\d+', str(match))
                        
                        if numbers:
                            budget_value = int(numbers[0])
                            if 100 <= budget_value <= 10000:  # Reasonable budget range
                                extracted_values['disclosed_budget'] = budget_value
            
            if count > 0:
                detected_signals.append(signal_type)
                signal_counts[signal_type] = count
        
        # Message characteristics
        message_metrics = self._analyze_message_characteristics(message)
        
        # Simple sentiment analysis
        sentiment_analysis = self._analyze_sentiment(message)
        
        # Context analysis (if history provided)
        context_analysis = {}
        if message_history:
            context_analysis = self._analyze_context(message, message_history)
        
        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(message_metrics, detected_signals)
        
        return {
            'detected_signals': detected_signals,
            'signal_counts': signal_counts,
            'extracted_values': extracted_values,
            'message_metrics': message_metrics,
            'sentiment_analysis': sentiment_analysis,
            'context_analysis': context_analysis,
            'engagement_score': engagement_score,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_message_characteristics(self, message: str) -> Dict:
        """Analyze basic message characteristics"""
        return {
            'length': len(message),
            'word_count': len(message.split()),
            'sentence_count': len([s for s in message.split('.') if s.strip()]),
            'question_marks': message.count('?'),
            'exclamation_marks': message.count('!'),
            'capital_ratio': sum(1 for c in message if c.isupper()) / max(len(message), 1),
            'avg_word_length': np.mean([len(word) for word in message.split()]) if message.split() else 0,
            'punctuation_density': sum(1 for c in message if c in '.,!?;:') / max(len(message), 1)
        }
    
    def _analyze_sentiment(self, message: str) -> Dict:
        """Simple rule-based sentiment analysis"""
        words = message.lower().split()
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        # Simple sentiment score calculation
        if len(words) == 0:
            sentiment_score = 0
        else:
            sentiment_score = (positive_count - negative_count) / len(words)
            sentiment_score = np.clip(sentiment_score * 5, -1, 1)  # Scale to -1, 1
        
        # Determine sentiment category
        if sentiment_score > 0.1:
            sentiment_category = 'positive'
        elif sentiment_score < -0.1:
            sentiment_category = 'negative'
        else:
            sentiment_category = 'neutral'
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_category': sentiment_category,
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'confidence': min(abs(sentiment_score) + 0.1, 1.0)
        }
    
    def _analyze_context(self, current_message: str, message_history: List[str]) -> Dict:
        """Analyze message in context of conversation history"""
        if not message_history:
            return {}
        
        # Calculate conversation flow metrics
        message_lengths = [len(msg) for msg in message_history + [current_message]]
        
        # Detect conversation patterns
        question_trend = self._detect_question_trend(message_history + [current_message])
        engagement_trend = self._detect_engagement_trend(message_history + [current_message])
        
        return {
            'conversation_length': len(message_history) + 1,
            'avg_message_length': np.mean(message_lengths),
            'message_length_trend': 'increasing' if message_lengths[-1] > np.mean(message_lengths[:-1]) else 'decreasing',
            'question_trend': question_trend,
            'engagement_trend': engagement_trend,
            'topic_consistency': self._analyze_topic_consistency(message_history + [current_message])
        }
    
    def _detect_question_trend(self, messages: List[str]) -> str:
        """Detect if questions are increasing, decreasing, or stable"""
        if len(messages) < 3:
            return 'insufficient_data'
        
        question_counts = [msg.count('?') for msg in messages]
        recent_avg = np.mean(question_counts[-3:])
        earlier_avg = np.mean(question_counts[:-3]) if len(question_counts) > 3 else 0
        
        if recent_avg > earlier_avg + 0.5:
            return 'increasing'
        elif recent_avg < earlier_avg - 0.5:
            return 'decreasing'
        else:
            return 'stable'
    
    def _detect_engagement_trend(self, messages: List[str]) -> str:
        """Detect engagement trend based on message length and content"""
        if len(messages) < 3:
            return 'insufficient_data'
        
        # Calculate engagement scores for each message
        engagement_scores = []
        for msg in messages:
            score = len(msg) / 50  # Base score from length
            score += msg.count('!') * 0.1  # Excitement
            score += msg.count('?') * 0.1  # Inquiry
            score += len([word for word in msg.lower().split() 
                         if word in self.positive_keywords]) * 0.2  # Positive words
            engagement_scores.append(score)
        
        # Compare recent vs earlier engagement
        recent_avg = np.mean(engagement_scores[-3:])
        earlier_avg = np.mean(engagement_scores[:-3]) if len(engagement_scores) > 3 else recent_avg
        
        if recent_avg > earlier_avg * 1.2:
            return 'increasing'
        elif recent_avg < earlier_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_topic_consistency(self, messages: List[str]) -> float:
        """Analyze how consistent the conversation topic is"""
        if len(messages) < 2:
            return 1.0
        
        # Simple keyword overlap analysis
        property_keywords = [
            'property', 'apartment', 'room', 'house', 'rent', 'lease',
            'viewing', 'tour', 'visit', 'location', 'amenities', 'facilities'
        ]
        
        topic_consistency_scores = []
        for msg in messages:
            words = msg.lower().split()
            property_word_count = sum(1 for word in words if word in property_keywords)
            consistency = property_word_count / max(len(words), 1)
            topic_consistency_scores.append(consistency)
        
        return np.mean(topic_consistency_scores)
    
    def _calculate_engagement_score(self, message_metrics: Dict, detected_signals: List[str]) -> float:
        """Calculate overall engagement score for the message"""
        score = 0.0
        
        # Length factor (optimal around 30-100 characters)
        length = message_metrics['length']
        if 30 <= length <= 100:
            score += 0.3
        elif 10 <= length <= 200:
            score += 0.1
        
        # Question factor
        score += min(message_metrics['question_marks'] * 0.2, 0.4)
        
        # Exclamation factor (shows enthusiasm)
        score += min(message_metrics['exclamation_marks'] * 0.1, 0.2)
        
        # Signal factor
        positive_signals = ['positive_signals', 'booking_intent', 'commitment_indicators', 'qualification_questions']
        negative_signals = ['objections', 'comparison_shopping']
        
        for signal in detected_signals:
            if signal in positive_signals:
                score += 0.15
            elif signal in negative_signals:
                score -= 0.1
        
        # Word count factor (not too short, not too long)
        word_count = message_metrics['word_count']
        if 5 <= word_count <= 50:
            score += 0.2
        elif word_count >= 3:
            score += 0.1
        
        return np.clip(score, 0, 1)
    
    def get_signal_summary(self, conversation_signals: List[List[str]]) -> Dict:
        """Get summary of signals detected across entire conversation"""
        all_signals = [signal for message_signals in conversation_signals for signal in message_signals]
        
        signal_counts = {}
        for signal in all_signals:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
        
        # Categorize signals
        positive_signals = sum(signal_counts.get(sig, 0) for sig in [
            'positive_signals', 'booking_intent', 'commitment_indicators', 
            'qualification_questions', 'urgency'
        ])
        
        negative_signals = sum(signal_counts.get(sig, 0) for sig in [
            'objections', 'comparison_shopping'
        ])
        
        return {
            'total_signals': len(all_signals),
            'unique_signals': len(signal_counts),
            'signal_counts': signal_counts,
            'positive_signals': positive_signals,
            'negative_signals': negative_signals,
            'signal_ratio': positive_signals / max(negative_signals, 1),
            'most_common_signal': max(signal_counts.items(), key=lambda x: x[1]) if signal_counts else None
        }
    
    def assess_conversation_quality(self, message_analyses: List[Dict]) -> Dict:
        """Assess overall conversation quality from message analyses"""
        if not message_analyses:
            return {'quality': 'unknown', 'score': 0}
        
        # Aggregate metrics
        total_engagement = sum(analysis['engagement_score'] for analysis in message_analyses)
        avg_engagement = total_engagement / len(message_analyses)
        
        total_sentiment = sum(analysis['sentiment_analysis']['sentiment_score'] for analysis in message_analyses)
        avg_sentiment = total_sentiment / len(message_analyses)
        
        # Count positive and negative signals
        all_signals = []
        for analysis in message_analyses:
            all_signals.extend(analysis['detected_signals'])
        
        positive_count = sum(1 for signal in all_signals if signal in [
            'positive_signals', 'booking_intent', 'commitment_indicators', 'qualification_questions'
        ])
        negative_count = sum(1 for signal in all_signals if signal in ['objections', 'comparison_shopping'])
        
        # Calculate quality score
        quality_score = 0
        quality_score += avg_engagement * 0.4
        quality_score += (avg_sentiment + 1) / 2 * 0.3  # Normalize sentiment to 0-1
        quality_score += min(positive_count / 5, 1) * 0.2
        quality_score -= min(negative_count / 3, 1) * 0.1
        
        quality_score = np.clip(quality_score, 0, 1)
        
        # Categorize quality
        if quality_score >= 0.8:
            quality_category = 'excellent'
        elif quality_score >= 0.6:
            quality_category = 'good'
        elif quality_score >= 0.4:
            quality_category = 'average'
        else:
            quality_category = 'poor'
        
        return {
            'quality': quality_category,
            'score': quality_score,
            'avg_engagement': avg_engagement,
            'avg_sentiment': avg_sentiment,
            'positive_signal_count': positive_count,
            'negative_signal_count': negative_count,
            'message_count': len(message_analyses)
        }