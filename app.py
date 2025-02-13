#!/usr/bin/env python3
# Save this file as event_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from datetime import datetime

def load_and_preprocess_data():
    # Read the CSV file
    df = pd.read_csv('hackaton.csv')
    
    # Convert date columns to datetime
    df['fecha_ev'] = pd.to_datetime(df['fecha_ev'], format='%d/%m/%Y')
    df['fecha_nac'] = pd.to_datetime(df['fecha_nac'], format='%d/%m/%Y')
    
    # Calculate age at event
    df['age'] = (df['fecha_ev'] - df['fecha_nac']).dt.days / 365.25
    
    # Create binary target variable (attended or not)
    df['attended'] = (df['estado'] == 'Asistido').astype(int)
    
    return df

def prepare_features(df):
    # Create feature set
    features = df[['age', 'importe_activos_0', 'importe_activos_1', 'tipo_ev']]
    
    # Handle categorical variables
    le = LabelEncoder()
    features['tipo_ev'] = le.fit_transform(features['tipo_ev'])
    
    # Scale numerical features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    
    return features_scaled, le

def train_model(features, target):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def main():
    st.title("Event Attendance Predictor")
    st.write("Analyze and predict event attendance based on customer profiles")
    
    # Load and process data
    try:
        df = load_and_preprocess_data()
        features, le = prepare_features(df)
        target = df['attended']
        
        # Train model
        model, X_test, y_test = train_model(features, target)
        
        # Sidebar for analysis options
        st.sidebar.header("Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis",
            ["Attendance Overview", "Profile Analysis", "Prediction Tool"]
        )
        
        if analysis_type == "Attendance Overview":
            st.header("Event Attendance Overview")
            
            # Overall attendance rate
            attendance_rate = df['attended'].mean() * 100
            st.metric("Overall Attendance Rate", f"{attendance_rate:.1f}%")
            
            # Attendance by event type
            event_attendance = df.groupby('tipo_ev')['attended'].mean().sort_values(ascending=False)
            fig = px.bar(event_attendance, 
                        title="Attendance Rate by Event Type",
                        labels={'index': 'Event Type', 'value': 'Attendance Rate'})
            st.plotly_chart(fig)
            
        elif analysis_type == "Profile Analysis":
            st.header("Customer Profile Analysis")
            
            # Age group analysis
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                   labels=['<30', '30-40', '40-50', '50-60', '>60'])
            age_attendance = df.groupby('age_group')['attended'].mean()
            fig = px.bar(age_attendance,
                        title="Attendance Rate by Age Group",
                        labels={'index': 'Age Group', 'value': 'Attendance Rate'})
            st.plotly_chart(fig)
            
            # Asset level analysis
            df['asset_quartile'] = pd.qcut(df['importe_activos_0'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            asset_attendance = df.groupby('asset_quartile')['attended'].mean()
            fig = px.bar(asset_attendance,
                        title="Attendance Rate by Asset Quartile",
                        labels={'index': 'Asset Quartile', 'value': 'Attendance Rate'})
            st.plotly_chart(fig)
            
        else:  # Prediction Tool
            st.header("Attendance Prediction Tool")
            
            # Input form for prediction
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=35)
                assets_0 = st.number_input("Current Assets", min_value=0.0, value=50000.0)
            with col2:
                assets_1 = st.number_input("Previous Year Assets", min_value=0.0, value=45000.0)
                event_type = st.selectbox("Event Type", sorted(df['tipo_ev'].unique()))
            
            if st.button("Predict Attendance Probability"):
                # Prepare input data
                input_data = pd.DataFrame({
                    'age': [age],
                    'importe_activos_0': [assets_0],
                    'importe_activos_1': [assets_1],
                    'tipo_ev': [event_type]
                })
                
                # Transform input data
                input_data['tipo_ev'] = le.transform(input_data['tipo_ev'])
                input_scaled = StandardScaler().fit(features).transform(input_data)
                
                # Make prediction
                probability = model.predict_proba(input_scaled)[0][1]
                st.metric("Attendance Probability", f"{probability:.1%}")
                
                # Show feature importance
                feature_importance = pd.DataFrame({
                    'Feature': features.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance,
                            x='Feature',
                            y='Importance',
                            title="Feature Importance for Prediction")
                st.plotly_chart(fig)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please ensure 'hackaton.csv' is in the same directory as this script.")

if __name__ == "__main__":
    main()