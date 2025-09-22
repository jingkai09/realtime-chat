# models.py
"""
ML Models and Data Processing for Stage 2 Conversation Scoring
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os

class Stage2MLModels:
    """Handles all ML model operations for Stage 2"""
    
    def __init__(self, model_dir="models/"):
        self.model_dir = model_dir
        self.conversation_model = None
        self.text_vectorizer = None
        self.feature_scaler = None
        
        # Initialize feature columns list
        self.feature_columns = [
            'message_count', 'avg_message_length', 'question_count',
            'positive_signals', 'negative_signals', 'urgency_signals',
            'booking_signals', 'commitment_signals', 'engagement_rate',
            'sentiment_score', 'signal_variety', 'conversation_duration'
        ]
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Model file paths
        self.model_paths = {
            'conversation': os.path.join(model_dir, 'stage2_conversation_model.pkl'),
            'vectorizer': os.path.join(model_dir, 'stage2_text_vectorizer.pkl'),
            'scaler': os.path.join(model_dir, 'stage2_feature_scaler.pkl')
        }
        
        self.load_or_create_models()
    
    def load_or_create_models(self):
        """Load existing models or create new ones"""
        try:
            # Try to load existing models
            self.conversation_model = joblib.load(self.model_paths['conversation'])
            self.text_vectorizer = joblib.load(self.model_paths['vectorizer'])
            self.feature_scaler = joblib.load(self.model_paths['scaler'])
            print("✅ Loaded existing Stage 2 ML models")
        except FileNotFoundError:
            print("📦 Creating new Stage 2 ML models...")
            self.create_and_train_models()
    
    def generate_synthetic_training_data(self, n_samples=1000):
        """Generate synthetic conversation data for initial training"""
        np.random.seed(42)
        
        # Sample messages for text vectorizer training
        sample_messages = [
            "I want to book a viewing immediately",
            "This looks perfect, when can we meet?",
            "I'm very interested in this property",
            "Not sure about the price, seems expensive",
            "What facilities are available nearby?",
            "I need to think about this more",
            "Definitely interested, let's schedule something",
            "Looking at other options as well",
            "This is exactly what I'm looking for",
            "Budget is a concern for me",
            "I love this place, it's amazing",
            "When can I move in?",
            "Are pets allowed?",
            "What's included in the rent?",
            "This doesn't meet my requirements",
            "I found something better elsewhere",
            "Perfect timing for my move",
            "I'm ready to sign the lease",
            "Need to discuss with my partner",
            "Sounds good, let's proceed"
        ]
        
        data = []
        
        for i in range(n_samples):
            # Base conversation metrics
            message_count = max(1, np.random.poisson(6))
            conversation_duration = np.random.exponential(12) + 3
            avg_message_length = np.random.normal(45, 15)
            
            # Signal counts (with realistic correlations)
            positive_signals = np.random.poisson(2)
            negative_signals = max(0, np.random.poisson(1) - positive_signals//3)
            urgency_signals = np.random.poisson(1) + (positive_signals > 3)
            booking_signals = np.random.poisson(1) + (positive_signals > 2)
            commitment_signals = np.random.poisson(1) + (booking_signals > 0)
            question_count = np.random.poisson(2) + (positive_signals > 1)
            
            # Derived features
            engagement_rate = message_count / conversation_duration
            sentiment_score = np.random.normal(0.05, 0.35)
            sentiment_score = np.clip(sentiment_score, -1, 1)
            
            # Add correlation with positive signals
            if positive_signals > negative_signals:
                sentiment_score += 0.2
            
            signal_variety = len([x for x in [positive_signals, negative_signals, 
                                urgency_signals, booking_signals, commitment_signals] if x > 0])
            
            # Target conversion score (realistic business logic)
            conversion_score = 0.3  # Base score
            
            # Positive factors
            conversion_score += 0.15 * min(positive_signals / 4, 1)
            conversion_score += 0.20 * min(booking_signals / 2, 1)
            conversion_score += 0.15 * min(commitment_signals / 2, 1)
            conversion_score += 0.10 * min(urgency_signals / 2, 1)
            conversion_score += 0.10 * (sentiment_score + 1) / 2
            conversion_score += 0.05 * min(engagement_rate / 1.5, 1)
            
            # Negative factors
            conversion_score -= 0.20 * min(negative_signals / 2, 1)
            
            # Engagement bonus
            if message_count >= 8 and engagement_rate > 1:
                conversion_score += 0.10
            
            # Add some noise
            conversion_score += np.random.normal(0, 0.05)
            conversion_score = np.clip(conversion_score, 0, 1)
            
            data.append({
                'message_count': message_count,
                'avg_message_length': max(10, avg_message_length),
                'question_count': question_count,
                'positive_signals': positive_signals,
                'negative_signals': negative_signals,
                'urgency_signals': urgency_signals,
                'booking_signals': booking_signals,
                'commitment_signals': commitment_signals,
                'engagement_rate': engagement_rate,
                'sentiment_score': sentiment_score,
                'signal_variety': signal_variety,
                'conversation_duration': conversation_duration,
                'conversion_score': conversion_score,
                'sample_message': np.random.choice(sample_messages)
            })
        
        return pd.DataFrame(data)
    
    def create_and_train_models(self):
        """Create and train new ML models"""
        # Generate training data
        training_data = self.generate_synthetic_training_data()
        
        # Prepare training data
        X = training_data[self.feature_columns]
        y = training_data['conversion_score']
        
        # Train conversation model
        self.conversation_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.conversation_model.fit(X, y)
        
        # Create and train text vectorizer
        self.text_vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        sample_messages = training_data['sample_message'].tolist()
        self.text_vectorizer.fit(sample_messages)
        
        # Create and train feature scaler
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(X)
        
        # Save all models
        self.save_models()
        
        # Print model performance
        train_score = self.conversation_model.score(X, y)
        print(f"📊 Model trained with R² score: {train_score:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.conversation_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🎯 Top 5 Most Important Features:")
        for _, row in importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    
    def save_models(self):
        """Save all trained models"""
        joblib.dump(self.conversation_model, self.model_paths['conversation'])
        joblib.dump(self.text_vectorizer, self.model_paths['vectorizer'])
        joblib.dump(self.feature_scaler, self.model_paths['scaler'])
        print(f"💾 Models saved to {self.model_dir}")
    
    def predict_conversation_score(self, features_dict):
        """Predict conversation score from features dictionary"""
        try:
            # Convert to array in correct order
            feature_array = np.array([[features_dict.get(col, 0) for col in self.feature_columns]])
            
            # Scale features
            feature_array_scaled = self.feature_scaler.transform(feature_array)
            
            # Predict
            prediction = self.conversation_model.predict(feature_array_scaled)[0]
            
            # Get feature importance for this prediction
            feature_importances = dict(zip(self.feature_columns, self.conversation_model.feature_importances_))
            
            return {
                'predicted_score': float(np.clip(prediction, 0, 1)),
                'confidence': min(1.0, abs(prediction) + 0.1),  # Simple confidence metric
                'top_contributing_features': sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:3]
            }
            
        except Exception as e:
            print(f"⚠ Prediction failed: {e}")
            return {'predicted_score': 0.5, 'confidence': 0.1, 'top_contributing_features': []}
    
    def analyze_text_features(self, text):
        """Extract features from text using vectorizer"""
        try:
            # Transform text to features
            text_features = self.text_vectorizer.transform([text]).toarray()[0]
            
            # Get feature names and their values
            feature_names = self.text_vectorizer.get_feature_names_out()
            
            # Get top contributing text features
            text_feature_importance = list(zip(feature_names, text_features))
            text_feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'text_features_vector': text_features,
                'text_feature_score': np.mean(text_features),
                'top_text_features': text_feature_importance[:5]
            }
            
        except Exception as e:
            print(f"⚠ Text analysis failed: {e}")
            return {'text_features_vector': [], 'text_feature_score': 0, 'top_text_features': []}
    
    def retrain_with_real_data(self, real_conversation_data):
        """Retrain models with real conversation outcomes"""
        if len(real_conversation_data) < 50:
            print("⚠️ Need at least 50 conversations for retraining")
            return False
        
        try:
            df = pd.DataFrame(real_conversation_data)
            
            # Prepare features
            X = df[self.feature_columns]
            y = df['actual_outcome']  # Should be 0-1 conversion score
            
            # Retrain model
            self.conversation_model.fit(X, y)
            self.feature_scaler.fit(X)
            
            # Save updated models
            self.save_models()
            
            # Evaluate performance
            score = self.conversation_model.score(X, y)
            print(f"🔄 Model retrained with {len(real_conversation_data)} real conversations")
            print(f"📊 New R² score: {score:.4f}")
            
            return True
            
        except Exception as e:
            print(f"⚠ Retraining failed: {e}")
            return False
    
    def get_model_info(self):
        """Get information about loaded models"""
        if self.conversation_model is None:
            return {"error": "No models loaded"}
        
        return {
            "conversation_model": {
                "type": type(self.conversation_model).__name__,
                "n_features": len(self.feature_columns),
                "feature_names": self.feature_columns
            },
            "text_vectorizer": {
                "max_features": self.text_vectorizer.max_features,
                "vocabulary_size": len(self.text_vectorizer.vocabulary_) if hasattr(self.text_vectorizer, 'vocabulary_') else 0
            },
            "model_files_exist": {
                path: os.path.exists(file_path) 
                for path, file_path in self.model_paths.items()
            }
        }