# streamlit_app.py
"""
Streamlit UI for Stage 2 Real-Time Conversation Scoring
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time

# Import your Stage 2 system
from scorer import EnhancedStage2ConversationScorer
from models import Stage2MLModels
from analyzer import ConversationAnalyzer

# Page configuration
st.set_page_config(
    page_title="Stage 2 Conversation Scorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'scorer' not in st.session_state:
    st.session_state.scorer = EnhancedStage2ConversationScorer()
    st.session_state.current_lead_id = None
    st.session_state.conversation_active = False
    st.session_state.conversation_history = []
    st.session_state.score_history = []
    st.session_state.message_count = 0

def reset_conversation():
    """Reset conversation state"""
    if st.session_state.conversation_active and st.session_state.current_lead_id:
        st.session_state.scorer.end_conversation(st.session_state.current_lead_id, 'ui_reset')
    
    st.session_state.current_lead_id = None
    st.session_state.conversation_active = False
    st.session_state.conversation_history = []
    st.session_state.score_history = []
    st.session_state.message_count = 0

def get_tier_color(tier):
    """Get color for lead tier"""
    colors = {
        'hot': '#ff4444',
        'warm': '#ffaa00', 
        'cold': '#4488ff',
        'inactive': '#888888'
    }
    return colors.get(tier.lower(), '#666666')

def display_score_metrics(result):
    """Display score metrics in a nice layout"""
    if not result or 'error' in result:
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_score = result.get('current_score', 0)
        st.metric(
            label="Current Score",
            value=f"{current_score:.3f}",
            delta=f"{result.get('score_change', 0):+.3f}"
        )
    
    with col2:
        tier = result.get('tier', 'unknown').upper()
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border-radius: 5px; background-color: {get_tier_color(tier)}20; border: 2px solid {get_tier_color(tier)};">
            <h3 style="color: {get_tier_color(tier)}; margin: 0;">{tier} TIER</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ml_score = result.get('ml_predicted_score', 0)
        st.metric(
            label="ML Predicted",
            value=f"{ml_score:.3f}"
        )
    
    with col4:
        stage1_score = result.get('stage1_score', 0)
        st.metric(
            label="Stage 1 Score",
            value=f"{stage1_score:.3f}"
        )

def display_message_analysis(analysis):
    """Display message analysis details"""
    if not analysis:
        return
    
    st.subheader("Message Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sentiment = analysis.get('sentiment_score', 0)
        sentiment_color = 'green' if sentiment > 0 else 'red' if sentiment < 0 else 'gray'
        
        st.markdown(f"""
        **Sentiment:** <span style="color: {sentiment_color};">{sentiment:.3f}</span> ({analysis.get('sentiment_category', 'neutral')})
        
        **Engagement:** {analysis.get('engagement_score', 0):.3f}
        
        **Message Length:** {analysis.get('message_length', 0)} characters
        
        **Questions Asked:** {analysis.get('question_count', 0)}
        """, unsafe_allow_html=True)
    
    with col2:
        signals = analysis.get('detected_signals', [])
        if signals:
            st.markdown("**Detected Signals:**")
            for signal in signals:
                st.badge(signal.replace('_', ' ').title(), type="secondary")
        
        extracted_values = analysis.get('extracted_values', {})
        if extracted_values:
            st.markdown("**Extracted Values:**")
            for key, value in extracted_values.items():
                st.write(f"- {key}: {value}")

def display_routing_decision(routing):
    """Display routing decision"""
    if not routing or routing.get('action') == 'continue':
        return
    
    action = routing.get('action', '').replace('_', ' ').title()
    reason = routing.get('reason', '').replace('_', ' ').title()
    urgency = routing.get('urgency', '')
    confidence = routing.get('confidence', 0)
    
    if urgency == 'high':
        alert_type = 'error'
    elif urgency == 'medium':
        alert_type = 'warning'
    else:
        alert_type = 'info'
    
    st.alert(f"""
    **Routing Action:** {action}
    
    **Reason:** {reason}
    
    **Urgency:** {urgency.title()}
    
    **Confidence:** {confidence:.2f}
    """, type=alert_type)

def create_score_chart():
    """Create real-time score chart"""
    if not st.session_state.score_history:
        return
    
    df = pd.DataFrame(st.session_state.score_history)
    
    fig = go.Figure()
    
    # Add score line
    fig.add_trace(go.Scatter(
        x=df['message_number'],
        y=df['score'],
        mode='lines+markers',
        name='Score',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Add tier background colors
    fig.add_hline(y=0.75, line_dash="dash", line_color="red", annotation_text="Hot Tier")
    fig.add_hline(y=0.55, line_dash="dash", line_color="orange", annotation_text="Warm Tier")
    fig.add_hline(y=0.35, line_dash="dash", line_color="blue", annotation_text="Cold Tier")
    
    fig.update_layout(
        title="Score Evolution",
        xaxis_title="Message Number",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_signal_chart():
    """Create signals over time chart"""
    if not st.session_state.conversation_history:
        return
    
    signal_data = []
    for i, msg in enumerate(st.session_state.conversation_history):
        if msg['sender'] == 'lead' and 'analysis' in msg:
            signals = msg.get('analysis', {}).get('detected_signals', [])
            signal_data.append({
                'message_number': i + 1,
                'signal_count': len(signals),
                'signals': ', '.join(signals) if signals else 'None'
            })
    
    if not signal_data:
        return
    
    df = pd.DataFrame(signal_data)
    
    fig = px.bar(
        df, 
        x='message_number', 
        y='signal_count',
        title="Signals Detected Over Time",
        hover_data=['signals']
    )
    
    fig.update_layout(height=300)
    return fig

# Main UI
st.title("📊 Stage 2 Real-Time Conversation Scorer")
st.markdown("Test customer conversations and track scoring in real-time")

# Sidebar for conversation control
with st.sidebar:
    st.header("Conversation Control")
    
    # New conversation section
    st.subheader("Start New Conversation")
    
    lead_id = st.text_input(
        "Lead ID", 
        value=f"STREAMLIT_{datetime.now().strftime('%H%M%S')}"
    )
    
    stage1_score = st.slider(
        "Stage 1 Score", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.01
    )
    
    budget = st.number_input(
        "Customer Budget", 
        min_value=0, 
        value=1200, 
        step=50
    )
    
    location = st.text_input("Preferred Location", value="KL City")
    
    if st.button("Start Conversation", type="primary"):
        reset_conversation()
        
        stage1_data = {
            'budget': budget,
            'location': location
        }
        
        result = st.session_state.scorer.start_stage2_scoring(
            lead_id, stage1_score, stage1_data
        )
        
        if 'error' not in result:
            st.session_state.current_lead_id = lead_id
            st.session_state.conversation_active = True
            st.session_state.score_history = [{
                'message_number': 0,
                'score': stage1_score,
                'tier': result['tier']
            }]
            st.success(f"Started conversation: {lead_id}")
        else:
            st.error(f"Error: {result['error']}")
    
    # Current conversation status
    if st.session_state.conversation_active:
        st.success(f"Active: {st.session_state.current_lead_id}")
        st.info(f"Messages: {st.session_state.message_count}")
        
        if st.button("End Conversation", type="secondary"):
            if st.session_state.current_lead_id:
                summary = st.session_state.scorer.end_conversation(
                    st.session_state.current_lead_id, 'ui_ended'
                )
                st.success("Conversation ended")
                reset_conversation()
                st.rerun()
    else:
        st.warning("No active conversation")
    
    # Quick test scenarios
    st.subheader("Quick Test Scenarios")
    
    if st.button("Positive Scenario"):
        if not st.session_state.conversation_active:
            st.warning("Start a conversation first")
        else:
            positive_messages = [
                "Hi, I'm very interested in your property",
                "This looks perfect for my needs!",
                "I need to move in urgently",
                "Can we schedule a viewing immediately?"
            ]
            
            for msg in positive_messages:
                result = st.session_state.scorer.process_conversation_message(
                    st.session_state.current_lead_id, msg, 'lead'
                )
                
                st.session_state.conversation_history.append({
                    'sender': 'lead',
                    'message': msg,
                    'timestamp': datetime.now(),
                    'result': result
                })
                
                st.session_state.score_history.append({
                    'message_number': len(st.session_state.score_history),
                    'score': result['current_score'],
                    'tier': result['tier']
                })
                
                st.session_state.message_count += 1
            
            st.success("Positive scenario completed")
            st.rerun()

# Main content area
if st.session_state.conversation_active:
    # Current conversation metrics
    if st.session_state.conversation_history:
        latest_result = st.session_state.conversation_history[-1].get('result')
        if latest_result:
            display_score_metrics(latest_result)
    
    # Chat interface
    st.subheader("💬 Chat Interface")
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.conversation_history:
            if msg['sender'] == 'lead':
                with st.chat_message("user"):
                    st.write(msg['message'])
                    
                    # Show analysis for lead messages
                    if 'result' in msg:
                        result = msg['result']
                        analysis = result.get('message_analysis')
                        routing = result.get('routing_decision')
                        
                        with st.expander("View Analysis"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if analysis:
                                    display_message_analysis(analysis)
                            
                            with col2:
                                if routing:
                                    display_routing_decision(routing)
            
            elif msg['sender'] == 'agent':
                with st.chat_message("assistant"):
                    st.write(msg['message'])
    
    # Message input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_message = st.text_input(
            "Customer message:",
            key="user_input",
            placeholder="Type customer message here..."
        )
    
    with col2:
        send_button = st.button("Send", type="primary")
    
    # Process message
    if send_button and user_message:
        # Process customer message
        result = st.session_state.scorer.process_conversation_message(
            st.session_state.current_lead_id, user_message, 'lead'
        )
        
        if 'error' not in result:
            # Add to conversation history
            st.session_state.conversation_history.append({
                'sender': 'lead',
                'message': user_message,
                'timestamp': datetime.now(),
                'result': result
            })
            
            # Update score history
            st.session_state.score_history.append({
                'message_number': len(st.session_state.score_history),
                'score': result['current_score'],
                'tier': result['tier']
            })
            
            st.session_state.message_count += 1
            
            # Show routing decision if important
            routing = result.get('routing_decision', {})
            if routing.get('action') != 'continue':
                st.toast(f"Routing: {routing.get('action', '').title()}", icon='🚨')
            
            st.rerun()
        else:
            st.error(f"Error: {result['error']}")
    
    # Add agent response option
    st.subheader("🤖 Agent Response")
    agent_message = st.text_input(
        "Agent response:",
        placeholder="Type agent response here..."
    )
    
    if st.button("Add Agent Response") and agent_message:
        result = st.session_state.scorer.process_conversation_message(
            st.session_state.current_lead_id, agent_message, 'agent'
        )
        
        st.session_state.conversation_history.append({
            'sender': 'agent', 
            'message': agent_message,
            'timestamp': datetime.now()
        })
        
        st.rerun()
    
    # Charts section
    st.subheader("📈 Real-Time Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        score_chart = create_score_chart()
        if score_chart:
            st.plotly_chart(score_chart, use_container_width=True)
    
    with col2:
        signal_chart = create_signal_chart()
        if signal_chart:
            st.plotly_chart(signal_chart, use_container_width=True)
    
    # Conversation summary
    if st.session_state.conversation_history:
        st.subheader("📋 Conversation Summary")
        
        active_conversations = st.session_state.scorer.get_active_conversations()
        current_conv = None
        
        for conv in active_conversations:
            if conv['lead_id'] == st.session_state.current_lead_id:
                current_conv = conv
                break
        
        if current_conv:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duration", f"{current_conv['duration_minutes']:.1f} min")
                st.metric("Engagement Rate", f"{current_conv['engagement_rate']:.2f}")
            
            with col2:
                st.metric("Avg Sentiment", f"{current_conv['avg_sentiment']:.3f}")
                st.metric("Quality", current_conv['conversation_quality'].title())
            
            with col3:
                signals = current_conv['signal_breakdown']
                st.metric("Positive Signals", signals['positive_signals'])
                st.metric("Booking Indicators", signals['booking_indicators'])

else:
    # Welcome screen
    st.info("👆 Start a new conversation using the sidebar to begin testing!")
    
    # Show analytics dashboard
    st.subheader("📊 System Analytics")
    
    try:
        analytics = st.session_state.scorer.get_analytics_dashboard(days=1)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Conversations", analytics['total_conversations'])
        
        with col2:
            st.metric("Avg Final Score", f"{analytics['avg_final_score']:.3f}")
        
        with col3:
            st.metric("Avg Duration", f"{analytics['avg_duration_minutes']:.1f} min")
        
        with col4:
            st.metric("Active Conversations", analytics['active_conversations'])
        
        # Tier distribution
        tier_dist = analytics.get('tier_distribution', {})
        if tier_dist:
            st.subheader("Tier Distribution")
            
            fig = px.pie(
                values=list(tier_dist.values()),
                names=list(tier_dist.keys()),
                title="Final Conversation Tiers"
            )
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Analytics error: {e}")

# Footer
st.markdown("---")
st.markdown("Stage 2 Conversation Scoring System - Real-Time Testing Interface")
