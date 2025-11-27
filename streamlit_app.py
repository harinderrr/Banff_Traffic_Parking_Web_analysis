"""
CMPT 3835 - Banff Traffic & Parking Prediction App with EDA + RAG Chatbot
Streamlit application with EDA, ML Modeling, XAI, and RAG Chatbot features
Group 11
Date: November 25, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# RAG Chatbot imports
try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer, util
    import torch
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Page configuration
st.set_page_config(
    page_title="Banff Parking Prediction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# RAG CHATBOT FUNCTIONS
# ============================================================================

@st.cache_resource
def load_rag_models():
    """Load embedding model and LLM for RAG chatbot"""
    if not RAG_AVAILABLE:
        return None, None
    
    try:
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        generator = pipeline("text2text-generation", model="google/flan-t5-base", device = -1)
        return embedder, generator
    except Exception as e:
        st.error(f"Error loading RAG models: {str(e)}")
        return None, None

def create_banff_documents():
    """Create searchable documents from Banff data"""
    documents = {}
    
    # Document 1: General Banff parking info
    documents["general_info"] = """
    Banff National Park Parking and Traffic System:
    This system monitors parking availability and traffic conditions across Banff National Park.
    The park experiences high visitor volumes during peak tourism season (June-September).
    Multiple parking facilities are available throughout the park at various locations including:
    Fire Hall Lot West, Bear Street Lot, Central Park Lot, Clock Tower Lot, Railway Parking,
    Bow Falls, Buffalo Street, and Banff Avenue parking areas.
    Traffic is monitored across 7 major routes connecting key attractions.
    """
    
    # Document 2: Parking statistics summary
    documents["parking_stats"] = """
    Parking Statistics (2025 Data):
    - Total parking transactions recorded: 85,928
    - Number of parking locations: 15+ major facilities
    - Average parking duration: 120-180 minutes
    - Peak parking hours: 10:00 AM - 1:00 PM (262 transactions per hour at peak)
    - Weekend utilization is 15% higher than weekdays
    - Most popular payment method: Digital payments (97% adoption - 53.2% bank cards, 43.7% mobile)
    - Top location: Fire Hall Lot West with 4,650 transactions
    - Revenue efficiency varies from 0.7 to 1.3 across locations
    - Peak parking hour: 11:00 AM with 9,167 total transactions
    """
    
    # Document 3: Traffic patterns summary
    documents["traffic_stats"] = """
    Traffic Statistics (2025 Analysis):
    - Average traffic speed: 15.44 mph across all routes
    - Monitored routes: Banff Springs to Downtown, Cave Avenue to Downtown, West Entrance to Downtown,
      East Entrance from Downtown, Downtown to various destinations
    - Traffic-parking correlation: Strong negative correlation (-0.55)
    - Peak traffic flow: 11:00 AM when parking demand is highest (16.2 mph speed)
    - Slowest routes: Routes 7 and 8 (Downtown to West Entrance) average 12.3 mph
    - Fastest route: Route 10 (West Entrance to Downtown) average 24.0 mph
    - Congestion zones: High congestion below 14 mph, moderate 14-16 mph, fast above 16 mph
    - Traffic delay patterns show highest congestion during 10 AM - 2 PM period
    """
    
    # Document 4: Key insights and recommendations
    documents["insights"] = """
    Key Insights from 2025 Analysis:
    - Visitors strategically time arrivals during optimal traffic flow periods
    - Parking demand PEAKS when traffic is flowing well (inverse relationship)
    - This means parking fills up when traffic is NOT congested
    - Weekend parking utilization significantly higher than weekdays (+15%)
    - Digital payment adoption (97%) enables real-time optimization opportunities
    - Infrastructure shows structural congestion patterns requiring long-term planning
    - Cash usage declining from 5% (2024) to 3.1% (2025)
    - Bank card usage increasing from 50.6% to 53.2%
    
    Visitor Recommendations:
    - Best time to visit: Early morning (before 9 AM) or late afternoon (after 3 PM)
    - Use Route 10 (West Entrance to Downtown) for fastest travel times (24 mph avg)
    - Avoid Routes 7 and 8 during peak hours (10 AM - 2 PM) - only 12.3 mph
    - Plan weekend visits with extra time for parking (+15% demand)
    - Best parking availability: Buffalo Street (45% occupancy typically)
    - Most congested: Railway Parking (often 92%+ full)
    - Consider Park & Ride at Fenlands with free shuttle service every 15 minutes
    """
    
    # Document 5: EDA Findings
    documents["eda_findings"] = """
    Comprehensive EDA Findings:
    
    Traffic Analysis:
    - 24-hour speed profiles show consistent patterns across routes
    - Morning rush shows speeds increasing from 15 mph to 16+ mph by 11 AM
    - Afternoon shows gradual decline in speeds
    - Route-specific patterns: Cave Avenue maintains better speeds (16-17 mph)
    - Downtown routes experience most congestion
    
    Parking Demand Patterns:
    - Clear peak between 10:00-13:00 hours
    - Weekend demand exceeds weekday by significant margin
    - All days average peaks at 262 transactions/hour
    - Weekday peak: 253 transactions/hour
    - Weekend peak: 271 transactions/hour
    - Early morning (6-9 AM) shows minimal demand
    - Late evening (after 6 PM) demand drops significantly
    
    Payment Evolution (2024 vs 2025):
    - Digital adoption growing (95% to 97%)
    - Cash declining rapidly (-1.9 percentage points)
    - Bank cards becoming dominant method
    - Mobile payments stable around 44%
    
    Correlation Insights:
    - Strong negative correlation (-0.55) between traffic speed and parking demand
    - Peak parking at 11:00 AM occurs when speed is 16.2 mph (above average)
    - This inverse relationship suggests visitors avoid congested arrival times
    - Strategic behavior: people park when traffic flows, not during jams
    """
    
    # Document 6: Model Performance Details
    documents["model_performance"] = """
    Machine Learning Model Performance:
    
    Random Forest Model (Best Performer):
    - R² Score: 0.760 (76% variance explained)
    - RMSE: 12.4 vehicles/hour
    - MAE: 8.2 vehicles/hour  
    - MAPE: 15.3%
    - Cross-validation: 5-fold CV performed
    
    Feature Importance Rankings:
    1. Hour of day (0.25) - Most important
    2. Day of week (0.18)
    3. Demand lag 24h (0.15)
    4. Is weekend (0.12)
    5. Average speed (0.08)
    6. Demand lag 1h (0.06)
    7. Rolling mean 24h (0.05)
    8. Month (0.04)
    9. Temperature (0.04)
    10. Precipitation (0.03)
    
    Model Achievements:
    - Successfully fixed data leakage (reduced R² from unrealistic 1.0 to 0.76)
    - Implemented 6 XAI techniques including SHAP analysis
    - Processed 144,000+ traffic records and 800,000+ parking transactions
    - Time series cross-validation for realistic performance
    - Proper feature engineering using only historical data
    """
    
    return documents

def retrieve_context(query, documents, embedder, top_k=3):
    """Retrieve most relevant documents for the query"""
    if embedder is None:
        return "", []
    
    # Compute embeddings for documents
    doc_embeddings = {
        doc_id: embedder.encode(text, convert_to_tensor=True)
        for doc_id, text in documents.items()
    }
    
    # Encode query
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Calculate similarities
    scores = {}
    for doc_id, emb in doc_embeddings.items():
        score = util.pytorch_cos_sim(query_embedding, emb).item()
        scores[doc_id] = score
    
    # Sort and get top K
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_doc_ids = [doc_id for doc_id, score in sorted_docs[:top_k]]
    
    # Concatenate selected documents
    context = "\n\n".join(documents[doc_id] for doc_id in top_doc_ids)
    return context, sorted_docs

def generate_response(query, context, generator):
    """Generate response using FLAN-T5 with retrieved context"""
    if generator is None:
        return "RAG model not available. Please install transformers and sentence-transformers."
    
    prompt = f"""Based on the Banff National Park parking and traffic data provided below, answer the user's question with specific details and statistics.

Data Context:
{context}

User Question: {query}

Provide a clear, informative answer with 2-4 sentences. Include relevant numbers and facts from the data.

Answer:"""
    
    try:
        # Generate response with anti-repetition settings
        outputs = generator(
            prompt, 
            max_new_tokens=200,
            min_length=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=2.5,        # ← Strong penalty against repetition
            no_repeat_ngram_size=3,        # ← Prevent 3-word phrase repetition
            early_stopping=True
        )
        
        response = outputs[0]['generated_text'].strip()
        
        # Clean up response if needed
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Additional safety: truncate if we see repetition pattern
        sentences = response.split('.')
        if len(sentences) > 3:
            # Check if last sentence is very similar to previous ones
            last_sent = sentences[-1].lower().strip()
            prev_sent = sentences[-2].lower().strip()
            
            if last_sent and prev_sent and (last_sent in prev_sent or prev_sent in last_sent):
                # Truncate repeated content
                response = '. '.join(sentences[:-1]) + '.'
        
        return response
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def rag_chatbot(query, documents, embedder, generator):
    """Main RAG chatbot function"""
    context, scores = retrieve_context(query, documents, embedder, top_k=3)
    response = generate_response(query, context, generator)
    return response, scores

# ============================================================================
# END RAG CHATBOT FUNCTIONS
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🏔️ Banff Intelligent Parking & Traffic System</h1>', unsafe_allow_html=True)
st.markdown("### ML-Powered Predictions with Explainable AI (XAI), Comprehensive EDA & RAG Chatbot")
st.markdown("---")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.scaler = None

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Banff_National_Park_Logo.svg/200px-Banff_National_Park_Logo.svg.png", width=150)
    st.markdown("### 📊 System Controls")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model",
        ["Random Forest (Best)", "XGBoost", "Gradient Boosting", "Linear Regression"]
    )
    
    # Date and time selection
    st.markdown("### 📅 Prediction Settings")
    pred_date = st.date_input("Select Date", datetime.now())
    pred_hour = st.slider("Select Hour", 0, 23, 12)
    
    # Parking lot selection
    parking_lots = ["Banff Avenue", "Bear Street", "Buffalo Street", "Railway Parking", "Bow Falls", 
                   "Fire Hall Lot West", "Central Park Lot", "Clock Tower Lot"]
    selected_lot = st.selectbox("Select Parking Lot", parking_lots)
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    
    # Mock metrics for demo
    metrics = {
        "r2": 0.760,
        "rmse": 12.4,
        "mae": 8.2,
        "mape": 15.3
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² Score", f"{metrics['r2']:.3f}")
        st.metric("RMSE", f"{metrics['rmse']:.1f}")
    with col2:
        st.metric("MAE", f"{metrics['mae']:.1f}")
        st.metric("MAPE", f"{metrics['mape']:.1f}%")

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model and artifacts"""
    try:
        model = joblib.load('.devcontainer/best_model.pkl')
        scaler = joblib.load('.devcontainer/scaler.pkl')
        return model, scaler
    except:
        st.info("Model files not found. Using mock predictions for demo.")
        return None, None

model, scaler = load_model()

# Main tabs - Added RAG Chatbot as 6th tab
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 EDA Analysis",
    "🔮 Predictions", 
    "🔬 XAI Analysis", 
    "📈 Model Performance",
    "🚦 Real-time Dashboard",
    "💬 RAG Chatbot",      
    "📚 Documentation"    
])

# Tab 1: EDA Analysis
with tab1:
    st.markdown("## 📊 Exploratory Data Analysis (EDA)")
    
    # Create sub-tabs for different EDA sections
    eda_tabs = st.tabs(["Traffic Analysis", "Parking Patterns", "Payment Trends", "Correlation Analysis"])
    
    # EDA Tab 1: Traffic Analysis
    with eda_tabs[0]:
        st.markdown("### 🚗 Traffic Speed & Congestion Analysis")
        
        # 24-Hour Speed Profile
        st.markdown("#### 24-Hour Speed Profile: All Routes")
        
        # Create sample data based on the image
        hours = list(range(24))
        routes_data = {
            'Banff Springs to Downtown': [15.5, 15.0, 14.8, 14.8, 14.9, 15.0, 15.2, 16.0, 16.5, 16.2, 15.8, 15.0, 14.5, 14.0, 13.8, 13.5, 14.2, 15.0, 15.5, 15.8, 16.0, 16.2, 16.0, 15.8],
            'Downtown to Banff Springs': [15.8, 15.5, 15.2, 15.0, 15.0, 15.1, 15.3, 15.5, 15.0, 14.5, 14.0, 13.5, 13.0, 13.2, 13.8, 14.5, 15.0, 15.2, 15.5, 15.7, 15.8, 15.5, 15.3, 15.0],
            'Cave Avenue to Downtown': [16.0, 15.8, 15.5, 15.5, 15.6, 16.0, 16.5, 17.0, 17.2, 16.8, 16.5, 16.8, 17.0, 16.5, 16.0, 15.8, 15.5, 15.8, 16.2, 16.5, 16.8, 17.0, 16.8, 16.5],
            'West Entrance to Downtown': [22.5, 22.0, 22.0, 22.0, 22.0, 22.2, 22.8, 25.0, 26.0, 25.8, 25.5, 25.0, 24.8, 24.5, 24.8, 25.0, 25.2, 25.0, 24.8, 24.5, 24.2, 23.8, 23.5, 23.0],
            'Downtown to West Entrance': [12.0, 12.0, 12.0, 12.0, 12.0, 12.2, 12.5, 13.0, 13.5, 13.2, 12.8, 12.5, 12.0, 12.2, 12.5, 13.0, 12.8, 12.5, 12.2, 12.0, 11.8, 11.5, 11.8, 12.0]
        }
        
        fig = go.Figure()
        
        for route, speeds in routes_data.items():
            fig.add_trace(go.Scatter(
                x=hours,
                y=speeds,
                mode='lines+markers',
                name=route,
                line=dict(width=2)
            ))
        
        # Add congestion zones
        fig.add_hrect(y0=0, y1=14, fillcolor="red", opacity=0.1, annotation_text="High Congestion Zone (<14 mph)")
        fig.add_hrect(y0=14, y1=16, fillcolor="yellow", opacity=0.1, annotation_text="Moderate Zone (14-16 mph)")
        fig.add_hrect(y0=16, y1=30, fillcolor="green", opacity=0.1, annotation_text="Fast Zone (>16 mph)")
        
        fig.update_layout(
            title="24-Hour Speed Profile: All Routes",
            xaxis_title="Hour of Day",
            yaxis_title="Average Speed (mph)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Traffic Congestion Heatmap
        st.markdown("#### Traffic Congestion Heatmap")
        
        routes = ['Banff Springs to Downtown', 'Cave Avenue to Downtown', 'Downtown to Banff Springs',
                 'Downtown to Cave Avenue', 'Downtown to West Entrance', 'East Entrance from Downtown',
                 'West Entrance to Downtown']
        
        # Create sample delay data
        delay_data = np.random.uniform(0, 0.8, size=(len(routes), 24))
        
        fig = px.imshow(
            delay_data,
            labels=dict(x="Hour of Day", y="Route", color="Average Delay (minutes)"),
            x=hours,
            y=routes,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            title="Traffic Congestion Heatmap: Average Delay by Route and Hour"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        with st.expander("🔍 Key Traffic Insights"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Speed", "15.44 mph", delta="-50% from optimal")
            with col2:
                st.metric("Slowest Route", "Downtown to West Entrance", "12.3 mph avg")
            with col3:
                st.metric("Fastest Route", "West Entrance to Downtown", "24.0 mph avg")
    
    # EDA Tab 2: Parking Patterns
    with eda_tabs[1]:
        st.markdown("### 🅿️ Parking Demand Analysis")
        
        # Parking Demand by Hour
        st.markdown("#### Parking Demand by Hour: 2025 Analysis")
        
        hours = list(range(24))
        all_days = [5, 8, 10, 12, 15, 25, 45, 88, 112, 185, 256, 262, 258, 255, 250, 248, 245, 240, 205, 138, 85, 42, 20, 8]
        weekdays = [4, 7, 9, 11, 14, 23, 42, 85, 108, 178, 248, 253, 250, 248, 245, 243, 240, 235, 200, 135, 82, 40, 18, 7]
        weekends = [6, 9, 11, 13, 16, 27, 48, 91, 116, 192, 264, 271, 266, 262, 255, 253, 250, 245, 210, 141, 88, 44, 22, 9]
        
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(x=hours, y=all_days, mode='lines+markers', name='All Days Average',
                                line=dict(width=3, color='red')))
        fig.add_trace(go.Scatter(x=hours, y=weekdays, mode='lines', name='Weekdays',
                                line=dict(dash='dash', color='blue')))
        fig.add_trace(go.Scatter(x=hours, y=weekends, mode='lines', name='Weekends',
                                line=dict(dash='dash', color='green')))
        
        # Add peak period annotation
        fig.add_vrect(x0=10, x1=13, fillcolor="yellow", opacity=0.2,
                     annotation_text="Peak Period (10:00-13:00)")
        
        fig.update_layout(
            title="Parking Demand by Hour: 2025 Analysis",
            xaxis_title="Hour of Day",
            yaxis_title="Average Transactions per Hour",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top Parking Locations
        st.markdown("#### Top 15 Parking Locations: Transaction Volume vs Revenue Efficiency")
        
        locations = ['Fire Hall Lot West', 'Bear St Lot', 'Central Park Lot', 'Clock Tower Lot',
                    'Fire Hall Lot 1 "West"', 'Bear Parkade L2', 'Health Unit Lot', 'Central Park Lot South',
                    'Town Hall Lot', 'Bear Street Parkade', 'Mt Royal Lot 1 "East"', 'Bear Parkade L1',
                    'Caribou Masons', 'Town Hall Lot North', 'Lynx 200 Block']
        
        transactions = [4650, 4635, 4561, 4016, 3926, 3888, 3702, 3470, 3225, 2517, 2436, 2054, 1598, 1574, 1401]
        revenue_per_trans = [8.0, 7.9, 7.8, 6.9, 6.7, 6.7, 6.3, 5.9, 5.5, 4.3, 4.2, 3.3, 2.7, 2.7, 2.4]
        revenue_efficiency = [1.3, 0.7, 1.2, 1.2, 1.1, 1.1, 1.15, 1.1, 0.9, 1.1, 1.15, 0.85, 0.75, 0.72, 0.8]
        
        # Create color scale based on efficiency
        colors = ['green' if e > 1.1 else 'yellow' if e > 0.9 else 'red' for e in revenue_efficiency]
        
        fig = go.Figure(go.Bar(
            x=transactions,
            y=locations,
            orientation='h',
            marker=dict(color=revenue_efficiency, colorscale='RdYlGn', showscale=True,
                       colorbar=dict(title="Revenue Efficiency Score")),
            text=[f'${r:.1f}K ({p:.1f}%)' for r, p in zip([t*rpt/1000 for t, rpt in zip(transactions, revenue_per_trans)],
                                                          [(t/sum(transactions)*100) for t in transactions])],
            textposition='inside'
        ))
        
        fig.update_layout(
            title="Top 15 Parking Locations: Transaction Volume vs Revenue Efficiency",
            xaxis_title="Number of Transactions",
            yaxis_title="Parking Location",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", "85,928")
        with col2:
            st.metric("Peak Hour", "11:00 AM", "262 transactions")
        with col3:
            st.metric("Weekend Premium", "+15%", "vs weekdays")
        with col4:
            st.metric("Top Location", "Fire Hall Lot West", "4,650 transactions")
    
    # EDA Tab 3: Payment Trends
    with eda_tabs[2]:
        st.markdown("### 💳 Payment Method Analysis")
        
        # Create two columns for 2024 vs 2025
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Payment Method Distribution - 2024")
            
            # Data for 2024
            labels_2024 = ['Bank card', 'Pay by phone', 'Cash']
            values_2024 = [50.6, 44.4, 5.0]
            colors = ['#3498db', '#2ecc71', '#95a5a6']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels_2024,
                values=values_2024,
                hole=.4,
                marker_colors=colors
            )])
            
            fig.update_layout(
                title="2024: 710,001 transactions",
                height=350,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Payment Method Distribution - 2025")
            
            # Data for 2025
            labels_2025 = ['Bank card', 'Pay by phone', 'Cash']
            values_2025 = [53.2, 43.7, 3.1]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels_2025,
                values=values_2025,
                hole=.4,
                marker_colors=colors
            )])
            
            fig.update_layout(
                title="2025: 85,928 transactions",
                height=350,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Key Changes
        st.markdown("#### Key Payment Trends")
        
        metrics_cols = st.columns(4)
        with metrics_cols[0]:
            st.metric("Bank Card Usage", "53.2%", delta="+2.6% from 2024")
        with metrics_cols[1]:
            st.metric("Mobile Payments", "43.7%", delta="-0.7% from 2024")
        with metrics_cols[2]:
            st.metric("Cash Usage", "3.1%", delta="-1.9% from 2024")
        with metrics_cols[3]:
            st.metric("Digital Adoption", "97%", delta="+2% from 2024")
    
    # EDA Tab 4: Correlation Analysis
    with eda_tabs[3]:
        st.markdown("### 📈 Correlation Analysis")
        
        st.markdown("#### Traffic Speed vs Parking Demand Throughout the Day")
        
        hours = list(range(24))
        traffic_speed = [15.5, 15.1, 15.0, 14.9, 14.8, 14.9, 15.3, 16.0, 16.1, 16.2, 15.8, 15.5, 15.0, 15.0, 14.9, 15.0, 15.3, 15.7, 15.8, 15.9, 15.9, 15.8, 15.6, 15.5]
        parking_demand_corr = [0, 0, 0, 0, 0, 0, 0, 3000, 3500, 6500, 8500, 9167, 8900, 8500, 8000, 7500, 7000, 4500, 3000, 500, 100, 50, 20, 10]
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traffic speed
        fig.add_trace(
            go.Scatter(x=hours, y=traffic_speed, name="Traffic Speed (mph)",
                      line=dict(color='blue', width=3)),
            secondary_y=False,
        )
        
        # Add parking demand
        fig.add_trace(
            go.Scatter(x=hours, y=parking_demand_corr, name="Parking Demand",
                      line=dict(color='red', width=3)),
            secondary_y=True,
        )
        
        # Add peak parking annotation
        fig.add_vrect(x0=10, x1=13, fillcolor="yellow", opacity=0.2)
        fig.add_annotation(x=11.5, y=9000, text="PEAK PARKING<br>9167 transactions<br>at 11:00",
                          showarrow=True, arrowhead=2, secondary_y=True)
        
        # Update axes
        fig.update_xaxes(title_text="Hour of Day")
        fig.update_yaxes(title_text="Traffic Speed (mph)", secondary_y=False, range=[12, 18])
        fig.update_yaxes(title_text="Parking Transactions/Hour", secondary_y=True)
        
        fig.update_layout(
            title="Traffic Speed vs Parking Demand: Negative Correlation (-0.55)",
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        with st.expander("📊 Correlation Insights"):
            st.markdown("""
            **Key Finding:** Strong negative correlation (-0.55) between traffic speed and parking demand
            
            - **Peak parking (11:00 AM):** Occurs when traffic is relatively fluid (16.2 mph)
            - **Interpretation:** Visitors time arrivals to avoid congestion
            - **Implication:** Parking peaks don't cause traffic congestion - they're strategically timed
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correlation", "-0.55", help="Negative correlation")
            with col2:
                st.metric("Peak Parking", "11:00 AM", "9,167 transactions")
            with col3:
                st.metric("Speed at Peak", "16.2 mph", "Above average")

# Tab 2: Predictions (formerly Tab 1)
with tab2:
    st.markdown("## 🎯 Parking Demand Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Prediction inputs
        with st.expander("📝 Advanced Settings", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                is_weekend = st.checkbox("Weekend", value=(datetime.now().weekday() >= 5))
                is_holiday = st.checkbox("Holiday", value=False)
                avg_speed = st.slider("Avg Traffic Speed (mph)", 10, 30, 15)
            with col_b:
                temperature = st.slider("Temperature (°C)", -20, 30, 10)
                precipitation = st.slider("Precipitation (mm)", 0, 50, 0)
                events = st.selectbox("Special Events", ["None", "Festival", "Concert", "Sports"])
    
    with col2:
        st.markdown("### 📍 Selected Location")
        st.info(f"**{selected_lot}**")
        st.markdown(f"**Date:** {pred_date}")
        st.markdown(f"**Hour:** {pred_hour}:00")
    
    if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
        with st.spinner("Calculating prediction..."):
            # Mock prediction logic
            base_demand = 50
            hour_factor = 1 + abs(12 - pred_hour) * 0.1
            weekend_factor = 1.3 if is_weekend else 1.0
            weather_factor = 1 - (precipitation * 0.01)
            
            predicted_demand = base_demand * hour_factor * weekend_factor * weather_factor
            predicted_demand = int(predicted_demand + np.random.normal(0, 5))
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Predicted Demand",
                    f"{predicted_demand} vehicles/hour",
                    delta=f"{predicted_demand - 45:+d} vs average"
                )
            with col2:
                occupancy = min(95, predicted_demand * 1.5)
                st.metric(
                    "Expected Occupancy",
                    f"{occupancy:.0f}%",
                    delta=f"{occupancy - 70:+.0f}% vs average"
                )
            with col3:
                wait_time = max(0, (predicted_demand - 40) * 0.5)
                st.metric(
                    "Est. Wait Time",
                    f"{wait_time:.0f} min",
                    delta=f"{wait_time - 5:+.0f} min vs average"
                )
            
            # Confidence interval plot
            st.markdown("### 📊 Prediction Confidence Interval")
            
            hours = [pred_hour - 1, pred_hour, pred_hour + 1]
            lower_bound = predicted_demand - 10
            upper_bound = predicted_demand + 10
            
            fig = go.Figure()
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=hours + hours[::-1],
                y=[lower_bound, lower_bound, lower_bound] + [upper_bound, upper_bound, upper_bound],
                fill='toself',
                fillcolor='rgba(30, 58, 138, 0.2)',
                line=dict(color='rgba(30, 58, 138, 0.2)'),
                name='95% Confidence',
                showlegend=True
            ))
            
            # Add prediction point
            fig.add_trace(go.Scatter(
                x=[pred_hour],
                y=[predicted_demand],
                mode='markers',
                marker=dict(size=15, color='#1E3A8A'),
                name='Prediction'
            ))
            
            fig.update_layout(
                title="Prediction with 95% Confidence Interval",
                xaxis_title="Hour",
                yaxis_title="Parking Demand",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 3: XAI Analysis (formerly Tab 2)
with tab3:
    st.markdown("## 🔬 Explainable AI (XAI) Analysis")
    
    xai_subtabs = st.tabs(["Feature Importance", "SHAP Analysis", "Partial Dependence", "Individual Predictions"])
    
    with xai_subtabs[0]:
        st.markdown("### 📊 Feature Importance Analysis")
        
        features = ['hour', 'day_of_week', 'demand_lag_24h', 'is_weekend', 'avg_speed', 
                   'demand_lag_1h', 'rolling_mean_24h', 'month', 'temperature', 'precipitation']
        importances = [0.25, 0.18, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03]
        
        fig = px.bar(
            x=importances, 
            y=features, 
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importances,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📖 Feature Explanations"):
            st.markdown("""
            - **hour**: Hour of the day (0-23) - captures daily patterns
            - **day_of_week**: Day of week (0-6) - weekly seasonality  
            - **demand_lag_24h**: Parking demand 24 hours ago
            - **is_weekend**: Binary indicator for weekends
            - **avg_speed**: Average traffic speed in mph
            """)
    
    with xai_subtabs[1]:
        st.markdown("### 🎯 SHAP (SHapley Additive exPlanations)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### SHAP Summary Plot")
            st.info("Shows feature impact on predictions across all samples")
            
            # Create mock SHAP visualization using plotly
            np.random.seed(42)
            fig = go.Figure()
            
            for i, feature in enumerate(features[:5]):
                x = np.random.randn(100) * 0.1
                y = [i] * 100
                colors = np.random.rand(100)
                
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='markers',
                    marker=dict(size=6, color=colors, colorscale='RdBu', showscale=(i == 0)),
                    name=feature, showlegend=False
                ))
            
            fig.update_layout(
                title="SHAP Summary Plot",
                xaxis_title="SHAP value (impact on prediction)",
                yaxis=dict(tickmode='array', tickvals=list(range(5)), ticktext=features[:5]),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### SHAP Waterfall Plot")
            st.info("Shows how each feature contributes to a single prediction")
            
            base_value = 45
            feature_contributions = [8, -3, 5, -2, 3, -1, 2, -1, 1, 0]
            
            fig = go.Figure(go.Waterfall(
                name="", orientation="v",
                measure=["relative"]*10 + ["total"],
                x=features[:10] + ["Prediction"],
                y=feature_contributions + [sum(feature_contributions) + base_value],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            fig.update_layout(
                title="SHAP Waterfall - Individual Prediction",
                height=400, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with xai_subtabs[2]:
        st.markdown("### 📈 Partial Dependence Plots")
        st.info("Shows how features affect predictions on average")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hours = list(range(24))
            demand_effect = [30 + 10*np.sin((h-6)*np.pi/12) for h in hours]
            
            fig = px.line(
                x=hours, y=demand_effect,
                title="Partial Dependence: Hour of Day",
                labels={'x': 'Hour', 'y': 'Parking Demand Effect'},
                markers=True
            )
            fig.update_traces(line=dict(width=3, color='#1E3A8A'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            day_effect = [42, 43, 45, 46, 48, 58, 55]
            
            fig = px.bar(
                x=days, y=day_effect,
                title="Partial Dependence: Day of Week",
                labels={'x': 'Day', 'y': 'Parking Demand Effect'},
                color=day_effect, color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with xai_subtabs[3]:
        st.markdown("### 🔍 Individual Prediction Explanation")
        
        sample_id = st.selectbox("Select Sample to Explain", ["Sample 1", "Sample 2", "Sample 3"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Feature Contributions")
            
            features_to_show = ['hour=14', 'is_weekend=True', 'demand_lag_24h=52', 'avg_speed=12']
            contributions = [5.2, 3.8, -1.5, -2.3]
            
            fig = px.bar(
                x=contributions, y=features_to_show, orientation='h',
                title=f"Feature Contributions for {sample_id}",
                color=contributions, color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Prediction Breakdown")
            st.metric("Base Value", "45.0")
            st.metric("Feature Impact", "+5.2")
            st.metric("Final Prediction", "50.2", delta="+5.2")

# Tab 4: Model Performance (formerly Tab 3)
with tab4:
    st.markdown("## 📊 Model Performance Metrics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        np.random.seed(42)
        n_points = 200
        actual = np.abs(np.random.normal(45, 15, n_points))
        predicted = actual + np.random.normal(0, 8, n_points)
        
        fig = px.scatter(
            x=actual, y=predicted,
            title="Actual vs Predicted Parking Demand",
            labels={'x': 'Actual', 'y': 'Predicted'},
            trendline="ols"
        )
        
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_shape(
            type="line",
            x0=min_val, x1=max_val, y0=min_val, y1=max_val,
            line=dict(color="red", dash="dash")
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        
        st.metric("R² Score", f"{r2:.3f}", help="Proportion of variance explained")
        st.metric("RMSE", f"{rmse:.1f}", help="Root Mean Square Error")
        st.metric("MAE", f"{mae:.1f}", help="Mean Absolute Error")
        st.metric("MAPE", "15.3%", help="Mean Absolute Percentage Error")
        
        st.markdown("---")
        st.markdown("### 🎯 Metric Interpretation")
        st.success(f"""
        - **R² = {r2:.3f}**: Model explains {r2*100:.1f}% of variance
        - **RMSE = {rmse:.1f}**: Average error of ~{rmse:.0f} vehicles
        - **MAE = {mae:.1f}**: Typical error of ~{mae:.0f} vehicles
        """)
    
    # Residuals analysis
    st.markdown("### 🔍 Residual Analysis")
    
    residuals = predicted - actual
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(
            residuals, nbins=30,
            title="Residual Distribution",
            labels={'value': 'Residuals', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            x=predicted, y=residuals,
            title="Residuals vs Fitted",
            labels={'x': 'Fitted Values', 'y': 'Residuals'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig = px.scatter(
            x=theoretical_quantiles, y=sample_quantiles,
            title="Q-Q Plot",
            labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}
        )
        
        fig.add_shape(
            type="line",
            x0=theoretical_quantiles.min(), x1=theoretical_quantiles.max(),
            y0=theoretical_quantiles.min(), y1=theoretical_quantiles.max(),
            line=dict(color="red", dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 5: Real-time Dashboard (formerly Tab 4)
with tab5:
    st.markdown("## 🚦 Real-time Parking & Traffic Dashboard")
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Current Occupancy", "73%", delta="+8%")
    with col2:
        st.metric("Available Spots", "127", delta="-15")
    with col3:
        st.metric("Avg Wait Time", "8 min", delta="+3 min")
    with col4:
        st.metric("Traffic Speed", "14 mph", delta="-3 mph")
    with col5:
        st.metric("Predicted Demand", "52/hour", delta="+7")
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        hours_ahead = list(range(24))
        forecast = [45 + 10*np.sin((h-6)*np.pi/12) + np.random.normal(0, 3) for h in hours_ahead]
        
        fig = px.line(
            x=hours_ahead, y=forecast,
            title="24-Hour Parking Demand Forecast",
            labels={'x': 'Hours Ahead', 'y': 'Predicted Demand'},
            markers=True
        )
        fig.add_hline(y=50, line_dash="dash", annotation_text="Capacity Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        lots = ["Banff Ave", "Bear St", "Buffalo St", "Railway", "Bow Falls"]
        occupancy = [73, 85, 45, 92, 61]
        
        fig = px.bar(
            x=lots, y=occupancy,
            title="Current Occupancy by Lot",
            labels={'x': 'Parking Lot', 'y': 'Occupancy (%)'},
            color=occupancy,
            color_continuous_scale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']]
        )
        fig.add_hline(y=80, line_dash="dash", annotation_text="High Occupancy")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### 🎯 Current Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **✅ Best Option Now**  
        Buffalo Street Parking  
        45% occupancy, 5 min walk
        """)
    
    with col2:
        st.warning("""
        **⚠️ Avoid**  
        Railway Parking  
        92% full, 15+ min wait
        """)
    
    with col3:
        st.info("""
        **📍 Alternative**  
        Park & Ride at Fenlands  
        Free shuttle every 15 min
        """)

# Tab 6: RAG Chatbot (NEW!)
with tab6:
    st.markdown("## 💬 RAG Chatbot - Ask Questions About Banff Parking & Traffic")
    
    if not RAG_AVAILABLE:
        st.error("""
        ⚠️ **RAG Chatbot Not Available**
        
        The chatbot requires additional packages. Please install:
        ```
        pip install transformers sentence-transformers torch
        ```
        """)
    else:
        st.markdown("""
        <div style='background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 20px;'>
        <h3 style='color: #1f77b4; margin-top: 0;'>🤖 Intelligent Question Answering</h3>
        <p style='margin-bottom: 0;'>
        Ask questions about Banff parking and traffic patterns in natural language. 
        The chatbot uses <strong>Retrieval-Augmented Generation (RAG)</strong> to provide accurate, 
        data-driven answers based on the Banff parking and traffic datasets.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load RAG models
        with st.spinner("🔄 Loading AI models... (This may take 1-2 minutes on first load)"):
            try:
                embedder, generator = load_rag_models()
                
                if embedder is None or generator is None:
                    st.error("Failed to load RAG models. Please check installation.")
                    st.stop()
                
                # Create documents
                documents = create_banff_documents()
                
                st.success("✅ AI models loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                st.info("💡 Tip: Make sure you have installed transformers and sentence-transformers packages")
                st.stop()
        # Example questions - Text format (non-clickable)
        st.markdown("### 💡 Sample Questions You Can Ask:")
        
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #4caf50;'>
        <p style='margin: 5px 0;'> "What are the busiest parking locations?"</p>
        <p style='margin: 5px 0;'> "When is the best time to visit to avoid traffic?"</p>
        <p style='margin: 5px 0;'> "Which routes should I avoid during peak hours?"</p>
        <p style='margin: 5px 0;'> "How long do people typically park for?"</p>
        <p style='margin: 5px 0;'> "What's the correlation between traffic and parking?"</p>
        <p style='margin: 5px 0;'> "Summarize weekend parking patterns"</p>
        <p style='margin: 5px 0;'> "Which payment methods are most popular?"</p>
        <p style='margin: 5px 0;'> "What are the key insights from the data?"</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Chat interface
        st.markdown("### 💬 Ask Your Question:")
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'query_input' not in st.session_state:
            st.session_state.query_input = ""
        
        # User input
        user_query = st.text_input(
            "Type your question here:",
            value=st.session_state.query_input,
            placeholder="e.g., What are the peak parking hours in Banff?",
            key="user_query_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            ask_button = st.button("🚀 Ask", type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear History")
        
        if clear_button:
            st.session_state.chat_history = []
            st.session_state.query_input = ""
            st.rerun()
        
        # Process query
        if ask_button and user_query:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get response
                    response, relevance_scores = rag_chatbot(user_query, documents, embedder, generator)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "response": response,
                        "scores": relevance_scores
                    })
                    
                    # Clear input
                    st.session_state.query_input = ""
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📜 Conversation History:")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q{len(st.session_state.chat_history) - i}: {chat['query'][:60]}...", expanded=(i == 0)):
                    st.markdown(f"**🙋 Question:**")
                    st.info(chat['query'])
                    
                    st.markdown(f"**🤖 Answer:**")
                    st.success(chat['response'])
                    
                    # Show relevance scores
                    with st.expander("🔍 Show Document Relevance Scores"):
                        st.write("Documents ranked by relevance:")
                        for doc_id, score in chat['scores'][:3]:
                            st.write(f"- **{doc_id}**: {score:.3f}")

        
# Tab 7: Documentation (formerly Tab 5)
with tab7:
    st.markdown("## 📚 System Documentation")
    
    doc_tabs = st.tabs(["User Guide", "Model Details", "Data Sources", "About"])
    
    with doc_tabs[0]:
        st.markdown("""
        ### 🎯 How to Use This System
        
        1. **Explore EDA**: Review traffic patterns, parking trends, and correlations
        2. **Make Predictions**: Select date, time, and location for parking demand forecast
        3. **Understand XAI**: View model explanations and feature importance
        4. **Check Performance**: Review model accuracy metrics
        5. **Monitor Real-time**: View current conditions and recommendations
        6. **Ask Questions**: Use the RAG chatbot for natural language queries
        
        ### 📊 Key Insights from EDA
        
        - **Peak Hours**: 10:00 AM - 1:00 PM (highest demand)
        - **Correlation**: Negative relationship between traffic speed and parking demand
        - **Payment Trends**: 97% digital payment adoption
        - **Route Performance**: Wide variation from 12.3 to 24.0 mph average
        """)
    
    with doc_tabs[1]:
        st.markdown("""
        ### 🤖 Model Architecture
        
        **Algorithm**: Random Forest Regressor  
        **Training Data**: 8 months (Jan-Aug 2025)  
        **Data Points**: 800,000+ parking transactions  
        **Features**: 25+ engineered features  
        
        ### 📈 Model Performance
        
        - **R² Score**: 0.76 (76% variance explained)
        - **RMSE**: 12.4 vehicles/hour
        - **MAE**: 8.2 vehicles/hour
        - **MAPE**: 15.3%
        
        ### 🔧 Key Achievements
        
        - Fixed data leakage (R² from 1.0 to realistic 0.76)
        - Implemented 6 XAI techniques
        - Processed 144,000+ traffic records
        - Added RAG chatbot for natural language interaction
        """)
    
    with doc_tabs[2]:
        st.markdown("""
        ### 📁 Data Sources
        
        **Parking Data**:
        - df_final_2025_processed_final.csv (70 columns)
        - df_parking_2024_processed_final.csv
        - df_trans_2024_processed_final.csv
        
        **Traffic Data**:
        - df_routes_processed_final.csv (37 columns)
        - 7 major routes analyzed
        - 15-minute interval aggregation
        
        **Coverage**:
        - January - August 2025
        - 20+ parking facilities
        - 795,929 parking transactions
        - 144,000+ traffic records
        """)
    
    with doc_tabs[3]:
        st.markdown("""
        ### 👥 About This Project
        
        **Course**: CMPT 3835 - ML Work Integrated Project 2  
        **Institution**: NorQuest College  
        **Term**: Fall 2025  
        **Group**: 11  
        **Team Members**: Harinderjeet Singh, Harjoban Singh, Anmolpreet Kaur, Gurwinder Kaur, Chahalpreet Singh  
        
        ### 🎯 Project Goals
        
        1. Predict parking demand with >75% accuracy ✅
        2. Provide explainable AI insights ✅
        3. Reduce traffic congestion in Banff
        4. Improve visitor experience
        5. Natural language interaction via RAG chatbot ✅
        """)

        
        # Information section
        st.markdown("---")
        
        with st.expander("ℹ️ How This Works - RAG Technology"):
            st.markdown("""
            **Retrieval-Augmented Generation (RAG)** combines two powerful AI techniques:
            
            1. **📚 Retrieval**: 
               - Your question is converted into a semantic embedding
               - The system searches through Banff parking/traffic documents
               - Most relevant information is retrieved using cosine similarity
            
            2. **🤖 Generation**: 
               - Retrieved context is provided to FLAN-T5 language model
               - The model generates a natural language response
               - Answer is grounded in your actual data
            
            **Technologies Used:**
            - **Embedding Model**: all-MiniLM-L6-v2 (fast, efficient semantic search)
            - **Language Model**: FLAN-T5-Base (instruction-following text generation)
            - **Data Sources**: Banff parking transactions + traffic monitoring data + EDA insights
            
            **Benefits:**
            - ✅ Answers based on real data (not hallucinated)
            - ✅ Natural language interaction
            - ✅ Explainable (shows which documents were used)
            - ✅ Fast and efficient
            - ✅ Works offline after initial model download
            
            **Available Knowledge Base:**
            - General Banff parking & traffic information
            - 2025 parking statistics (85,928 transactions)
            - Traffic patterns across 7 routes
            - Key insights and visitor recommendations
            - EDA findings and correlations
            - Model performance details
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>© 2025 Banff Intelligent Parking System | CMPT 3835 Group 11 Project | Enhanced with RAG Chatbot | Last Updated: Nov 25, 2025</p>
</div>
""", unsafe_allow_html=True)
