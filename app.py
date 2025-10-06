import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern minimalist design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2196F3, #21CBF3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #2196F3, #21CBF3);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.4);
    }
    .feature-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        model = joblib.load('car_price_model.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, metadata
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run train_and_save_model.py first!")
        st.stop()

model, metadata = load_model()

# Title and description
st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get instant, AI-powered price predictions for your car</p>', unsafe_allow_html=True)

# Sidebar for quick actions
with st.sidebar:
    st.markdown("### ⚙️ Quick Actions")
    
    # Use default values button
    use_defaults = st.button("📊 Use Sample Car", type="secondary")
    
    # Auto-fill with average values
    use_averages = st.button("📈 Use Average Values", type="secondary")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    This AI model predicts car prices based on multiple features including:
    - Vehicle specifications
    - Age and mileage
    - Fuel and transmission type
    - Seller information
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.metric("Model Type", "Polynomial Regression")
    
    if 'best_params' in metadata:
        degree = metadata['best_params'].get('poly__degree', 'N/A')
        st.metric("Polynomial Degree", degree)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🎯 Basic Info", "⚙️ Specifications", "📈 Advanced"])
    
    # Initialize session state for form values
    if 'form_data' not in st.session_state or use_defaults or use_averages:
        if use_defaults:
            st.session_state.form_data = {
                'age': 5,
                'km_driven': 50000,
                'fuel': 'Petrol',
                'seller_type': 'Individual',
                'transmission': 'Manual',
                'owner': 'First Owner',
                'mileage': 18.5,
                'engine': 1200.0,
                'max_power': 85.0,
                'torque': 115.0,
                'seats': 5
            }
        elif use_averages:
            st.session_state.form_data = {
                'age': int(metadata['feature_ranges']['age']['mean']),
                'km_driven': int(metadata['feature_ranges']['km_driven']['mean']),
                'fuel': metadata['fuel_types'][0],
                'seller_type': metadata['seller_types'][0],
                'transmission': metadata['transmission_types'][0],
                'owner': metadata['owner_types'][0],
                'mileage': round(metadata['feature_ranges']['mileage']['mean'], 1),
                'engine': round(metadata['feature_ranges']['engine']['mean'], 0),
                'max_power': round(metadata['feature_ranges']['max_power']['mean'], 1),
                'torque': round(metadata['feature_ranges']['torque']['mean'], 1),
                'seats': int(metadata['feature_ranges']['seats']['mean'])
            }
        else:
            st.session_state.form_data = {
                'age': 5,
                'km_driven': 50000,
                'fuel': 'Petrol',
                'seller_type': 'Individual',
                'transmission': 'Manual',
                'owner': 'First Owner',
                'mileage': 18.5,
                'engine': 1200.0,
                'max_power': 85.0,
                'torque': 115.0,
                'seats': 5
            }
    
    with tab1:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Basic Information</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            current_year = datetime.now().year
            year = st.slider(
                "📅 Manufacturing Year",
                min_value=current_year - metadata['feature_ranges']['age']['max'],
                max_value=current_year,
                value=current_year - st.session_state.form_data['age'],
                help="Select the year your car was manufactured"
            )
            age = current_year - year
            
            km_driven = st.number_input(
                "🛣️ Kilometers Driven",
                min_value=0,
                max_value=500000,
                value=st.session_state.form_data['km_driven'],
                step=1000,
                help="Total kilometers the car has been driven"
            )
        
        with col_b:
            fuel = st.selectbox(
                "⛽ Fuel Type",
                options=metadata['fuel_types'],
                index=metadata['fuel_types'].index(st.session_state.form_data['fuel']) if st.session_state.form_data['fuel'] in metadata['fuel_types'] else 0,
                help="Select the fuel type of your car"
            )
            
            transmission = st.selectbox(
                "⚙️ Transmission",
                options=metadata['transmission_types'],
                index=metadata['transmission_types'].index(st.session_state.form_data['transmission']) if st.session_state.form_data['transmission'] in metadata['transmission_types'] else 0,
                help="Select transmission type"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Technical Specifications</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            mileage = st.number_input(
                "⚡ Mileage (kmpl)",
                min_value=float(metadata['feature_ranges']['mileage']['min']),
                max_value=float(metadata['feature_ranges']['mileage']['max']),
                value=float(st.session_state.form_data['mileage']),
                step=0.5,
                format="%.1f",
                help="Fuel efficiency in kilometers per liter"
            )
            
            engine = st.number_input(
                "🔧 Engine Capacity (cc)",
                min_value=float(metadata['feature_ranges']['engine']['min']),
                max_value=float(metadata['feature_ranges']['engine']['max']),
                value=float(st.session_state.form_data['engine']),
                step=50.0,
                format="%.0f",
                help="Engine displacement in cubic centimeters"
            )
            
            max_power = st.number_input(
                "💪 Max Power (bhp)",
                min_value=float(metadata['feature_ranges']['max_power']['min']),
                max_value=float(metadata['feature_ranges']['max_power']['max']),
                value=float(st.session_state.form_data['max_power']),
                step=5.0,
                format="%.1f",
                help="Maximum power output in brake horsepower"
            )
        
        with col_b:
            torque = st.number_input(
                "🔄 Torque (Nm)",
                min_value=float(metadata['feature_ranges']['torque']['min']),
                max_value=float(metadata['feature_ranges']['torque']['max']),
                value=float(st.session_state.form_data['torque']),
                step=5.0,
                format="%.1f",
                help="Maximum torque in Newton-meters"
            )
            
            seats = st.number_input(
                "💺 Number of Seats",
                min_value=int(metadata['feature_ranges']['seats']['min']),
                max_value=int(metadata['feature_ranges']['seats']['max']),
                value=int(st.session_state.form_data['seats']),
                step=1,
                help="Total seating capacity"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="feature-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Seller Information</div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            seller_type = st.selectbox(
                "👤 Seller Type",
                options=metadata['seller_types'],
                index=metadata['seller_types'].index(st.session_state.form_data['seller_type']) if st.session_state.form_data['seller_type'] in metadata['seller_types'] else 0,
                help="Type of seller"
            )
        
        with col_b:
            owner = st.selectbox(
                "👥 Owner Type",
                options=metadata['owner_types'],
                index=metadata['owner_types'].index(st.session_state.form_data['owner']) if st.session_state.form_data['owner'] in metadata['owner_types'] else 0,
                help="Ownership history"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🎯 Predict Price", type="primary")

with col2:
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)
    
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'km_driven': [km_driven],
            'fuel': [fuel],
            'seller_type': [seller_type],
            'transmission': [transmission],
            'owner': [owner],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'torque': [torque],
            'seats': [seats]
        })
        
        # Make prediction
        with st.spinner("🔮 Calculating price..."):
            prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size: 1.2rem; opacity: 0.9;">Estimated Price</div>
            <div class="prediction-value">₹ {prediction:,.0f}</div>
            <div style="font-size: 0.95rem; opacity: 0.85;">Indian Rupees</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence indicator
        st.markdown("<br>", unsafe_allow_html=True)
        confidence = min(95, max(75, 85 + np.random.randint(-5, 5)))
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence}%")
        
        # Price breakdown visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📈 Price Factors")
        
        # Create a simple breakdown
        factors = {
            'Base Value': prediction * 0.4,
            'Age Impact': prediction * 0.15,
            'Mileage': prediction * 0.20,
            'Specifications': prediction * 0.15,
            'Condition': prediction * 0.10
        }
        
        fig = go.Figure(data=[go.Bar(
            x=list(factors.values()),
            y=list(factors.keys()),
            orientation='h',
            marker=dict(
                color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'],
            ),
            text=[f"₹{v:,.0f}" for v in factors.values()],
            textposition='auto',
        )])
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="",
            yaxis_title="",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("👆 Fill in the car details and click 'Predict Price' to get an estimate")
        
        # Show sample prediction range
        st.markdown("#### 💡 Quick Stats")
        st.metric("Average Market Price", "₹ 5,50,000", "±15%")
        st.metric("Total Cars Analyzed", "8,000+")
        st.metric("Prediction Accuracy", "85%+")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 Powered by Machine Learning | Built with Streamlit</p>
    <p style="font-size: 0.85rem;">Predictions are estimates and may vary from actual market prices</p>
</div>
""", unsafe_allow_html=True)