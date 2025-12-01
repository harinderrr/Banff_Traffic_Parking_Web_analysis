# 🏔️ Banff Intelligent Parking & Traffic Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_APP_URL_HERE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning system for predicting parking demand and traffic patterns in Banff National Park. Features include EDA visualizations, XAI (Explainable AI) analysis, real-time predictions, and a RAG-powered chatbot for natural language queries.

![Banff App Screenshot](assets/screenshot.png)

## 🎯 Features

- **📊 Exploratory Data Analysis (EDA)**
  - 24-hour traffic speed profiles across 7 routes
  - Parking demand patterns (weekday vs weekend)
  - Payment method evolution (2024-2025)
  - Traffic-parking correlation analysis

- **🔮 Intelligent Predictions**
  - Location-specific parking demand forecasting
  - Real-time occupancy estimation
  - Dynamic search time calculation (0-30 min)
  - Confidence intervals with hourly breakdowns

- **🔬 Explainable AI (XAI)**
  - Feature importance analysis
  - SHAP (SHapley Additive exPlanations) visualizations
  - Partial dependence plots
  - Individual prediction explanations

- **🤖 RAG Chatbot**
  - Natural language Q&A about parking/traffic data
  - Retrieval-Augmented Generation using FLAN-T5
  - Semantic search with sentence transformers
  - Context-aware responses with source attribution

- **📱 Mobile-Responsive Design**
  - Optimized for desktop, tablet, and mobile devices
  - Touch-friendly interactions (44px minimum tap targets)
  - Adaptive layouts and typography
  - Progressive enhancement for all screen sizes

## 📈 Model Performance

- **Algorithm**: Random Forest Regressor
- **R² Score**: 0.760 (76% variance explained)
- **RMSE**: 12.4 vehicles/hour
- **MAE**: 8.2 vehicles/hour
- **MAPE**: 15.3%

## 📊 Dataset

- **Parking Data**: 85,928 transactions (Jan-Aug 2025)
- **Traffic Data**: 144,000+ records across 7 routes
- **Facilities**: 15+ parking locations
- **Study Period**: 8 months (January - August 2025)

### Key Insights

- Peak parking hours: **10:00 AM - 1:00 PM**
- Strong negative correlation: **-0.55** (traffic speed vs parking demand)
- Digital payment adoption: **97%** (53.2% cards, 43.7% mobile)
- Weekend demand: **+15%** higher than weekdays
- Average parking duration: **120-180 minutes**

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
