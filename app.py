#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def load_and_preprocess_data():
    # Read the CSV file with new name
    df = pd.read_csv('dataset_limpio.csv')
    
    # Convert date columns to datetime
    df['fecha_ev'] = pd.to_datetime(df['fecha_ev'], format='%d/%m/%Y')
    df['fecha_nac'] = pd.to_datetime(df['fecha_nac'], format='%d/%m/%Y')
    
    # Create binary target variable (attended or not)
    df['attended'] = (df['estado'] == 'Asistido').astype(int)
    
    return df

def prepare_features(df):
    # Create feature set using cat_edad instead of calculated age
    features = df[['cat_edad', 'tipo_ev']]
    
    # Handle categorical variables
    le_age = LabelEncoder()
    le_event = LabelEncoder()
    features['cat_edad'] = le_age.fit_transform(features['cat_edad'])
    features['tipo_ev'] = le_event.fit_transform(features['tipo_ev'])
    
    # Scale numerical features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    
    return features_scaled, le_age, le_event

def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def analyze_financial_evolution(df):
    # Calculate growth rate for each customer
    df['growth_rate'] = ((df['importe_activos_1'] - df['importe_activos_0']) / df['importe_activos_0'] * 100)
    
    # Calculate average growth rate by age category and event type
    growth_by_age = df.groupby('cat_edad')['growth_rate'].agg(['mean', 'count']).round(2)
    growth_by_event = df.groupby('tipo_ev')['growth_rate'].agg(['mean', 'count']).round(2)
    
    # Calculate growth rates by attendance status
    growth_by_attendance = df.groupby('attended')['growth_rate'].agg(['mean', 'count']).round(2)
    
    # Calculate growth rates by attendance status and age category
    growth_by_age_attendance = df.groupby(['cat_edad', 'attended'])['growth_rate'].mean().round(2).unstack()
    growth_by_age_attendance.columns = ['Not Attended', 'Attended']
    
    # Calculate growth rates by attendance status and event type
    growth_by_event_attendance = df.groupby(['tipo_ev', 'attended'])['growth_rate'].mean().round(2).unstack()
    growth_by_event_attendance.columns = ['Not Attended', 'Attended']
    
    # Calculate percentage of clients with positive/negative growth
    df['growth_direction'] = df['growth_rate'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    growth_direction_by_age = df.groupby(['cat_edad', 'growth_direction']).size().unstack(fill_value=0)
    growth_direction_by_age = growth_direction_by_age.div(growth_direction_by_age.sum(axis=1), axis=0) * 100
    
    growth_direction_by_event = df.groupby(['tipo_ev', 'growth_direction']).size().unstack(fill_value=0)
    growth_direction_by_event = growth_direction_by_event.div(growth_direction_by_event.sum(axis=1), axis=0) * 100
    
    return (df, growth_by_age, growth_by_event, growth_direction_by_age, growth_direction_by_event, 
            growth_by_attendance, growth_by_age_attendance, growth_by_event_attendance)

def get_group_recommendations(growth_rate, segment_type, segment_value):
    recommendations = []
    
    if growth_rate > 0:
        recommendations.extend([
            "🔹 Inversión en Fondos:",
            "  • Presentar opciones de fondos de inversión adaptados al perfil de riesgo",
            "  • Ofrecer productos con horizontes temporales variados",
            "🔹 Seguros:",
            "  • Proponer seguros de ahorro con rentabilidad garantizada",
            "  • Ofrecer seguros unit-linked para perfiles más dinámicos"
        ])
        
        # Add specific recommendations based on segment
        if segment_type == 'age':
            if 'joven' in segment_value.lower():
                recommendations.extend([
                    "🔹 Productos específicos para jóvenes inversores:",
                    "  • Planes de ahorro sistemático",
                    "  • Fondos con mayor componente de renta variable"
                ])
            elif 'adulto' in segment_value.lower():
                recommendations.extend([
                    "🔹 Productos para consolidación patrimonial:",
                    "  • Fondos mixtos balanceados",
                    "  • Seguros de ahorro garantizado"
                ])
            else:  # senior/mayor
                recommendations.extend([
                    "🔹 Productos para preservación de capital:",
                    "  • Fondos de renta fija",
                    "  • Seguros de rentas vitalicias"
                ])
    else:
        recommendations.extend([
            "🔹 Asesoría Financiera:",
            "  • Ofrecer consultas personalizadas de planificación financiera",
            "  • Realizar análisis detallado de la situación patrimonial",
            "🔹 Productos de Financiación:",
            "  • Evaluar necesidades de préstamos personales",
            "  • Proponer productos de refinanciación si es necesario"
        ])
        
        # Add specific recommendations based on segment
        if segment_type == 'age':
            if 'joven' in segment_value.lower():
                recommendations.extend([
                    "🔹 Apoyo financiero para jóvenes:",
                    "  • Préstamos en condiciones especiales",
                    "  • Programas de educación financiera"
                ])
            elif 'adulto' in segment_value.lower():
                recommendations.extend([
                    "🔹 Soluciones de consolidación:",
                    "  • Productos de reestructuración de deuda",
                    "  • Asesoramiento en optimización fiscal"
                ])
            else:  # senior/mayor
                recommendations.extend([
                    "🔹 Protección patrimonial:",
                    "  • Servicios de gestión patrimonial",
                    "  • Asesoramiento en planificación sucesoria"
                ])
    
    return recommendations

def main():
    st.title("Customer Analytics Dashboard")
    
    try:
        df = load_and_preprocess_data()
        (df, growth_by_age, growth_by_event, growth_direction_by_age, growth_direction_by_event,
         growth_by_attendance, growth_by_age_attendance, growth_by_event_attendance) = analyze_financial_evolution(df)
        features, le_age, le_event = prepare_features(df)
        target = df['attended']
        
        model, X_test, y_test = train_model(features, target)
        
        # Sidebar for analysis options
        st.sidebar.header("Analysis Options")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis",
            ["Attendance Overview", "Profile Analysis", "Prediction Tool", "Financial Evolution"]
        )
        
        if analysis_type == "Attendance Overview":
            st.header("Event Attendance Overview")
            
            attendance_rate = df['attended'].mean() * 100
            st.metric("Overall Attendance Rate", f"{attendance_rate:.1f}%")
            
            event_attendance = df.groupby('tipo_ev')['attended'].mean().sort_values(ascending=False)
            fig = px.bar(event_attendance, 
                        title="Attendance Rate by Event Type",
                        labels={'index': 'Event Type', 'value': 'Attendance Rate'})
            st.plotly_chart(fig)
            
        elif analysis_type == "Profile Analysis":
            st.header("Customer Profile Analysis")
            
            age_attendance = df.groupby('cat_edad')['attended'].mean()
            fig = px.bar(age_attendance,
                        title="Attendance Rate by Age Category",
                        labels={'index': 'Age Category', 'value': 'Attendance Rate'})
            st.plotly_chart(fig)
            
        elif analysis_type == "Prediction Tool":
            st.header("Attendance Prediction Tool")
            
            # Simplified input form using cat_edad
            age_category = st.selectbox("Age Category", sorted(df['cat_edad'].unique()))
            event_type = st.selectbox("Event Type", sorted(df['tipo_ev'].unique()))
            
            if st.button("Predict Attendance Probability"):
                input_data = pd.DataFrame({
                    'cat_edad': [age_category],
                    'tipo_ev': [event_type]
                })
                
                input_data['cat_edad'] = le_age.transform(input_data['cat_edad'])
                input_data['tipo_ev'] = le_event.transform(input_data['tipo_ev'])
                input_scaled = StandardScaler().fit(features).transform(input_data)
                
                probability = model.predict_proba(input_scaled)[0][1]
                st.metric("Attendance Probability", f"{probability:.1%}")
                
                feature_importance = pd.DataFrame({
                    'Feature': features.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance,
                            x='Feature',
                            y='Importance',
                            title="Feature Importance for Prediction")
                st.plotly_chart(fig)
                
        else:  # Financial Evolution
            st.header("Financial Evolution Analysis")
            
            tab1, tab2, tab3 = st.tabs(["By Age Category", "By Event Type", "By Attendance Status"])
            
            with tab1:
                st.subheader("Financial Evolution by Age Category")
                
                # Growth rate visualization
                fig = px.bar(growth_by_age.reset_index(), 
                           x='cat_edad', 
                           y='mean',
                           title='Average Growth Rate by Age Category',
                           labels={'mean': 'Growth Rate (%)', 'cat_edad': 'Age Category'},
                           text='mean')
                fig.update_traces(texttemplate='%{text:.1f}%')
                st.plotly_chart(fig)
                
                # Growth rate comparison by attendance
                fig = px.bar(growth_by_age_attendance.reset_index(), 
                           x='cat_edad',
                           y=['Attended', 'Not Attended'],
                           title='Growth Rate Comparison: Attended vs Not Attended by Age Category',
                           labels={'value': 'Growth Rate (%)', 'cat_edad': 'Age Category'},
                           barmode='group')
                fig.update_traces(texttemplate='%{y:.1f}%')
                st.plotly_chart(fig)
                
                # Growth direction visualization
                fig = px.bar(growth_direction_by_age.reset_index(), 
                           x='cat_edad',
                           y=['Positive', 'Negative'],
                           title='Distribution of Growth Direction by Age Category',
                           labels={'value': 'Percentage', 'cat_edad': 'Age Category'},
                           barmode='stack')
                st.plotly_chart(fig)
                
                # Recommendations for selected age category
                selected_age_cat = st.selectbox("Select Age Category for Recommendations", 
                                              growth_by_age.index)
                growth_rate = growth_by_age.loc[selected_age_cat, 'mean']
                
                st.subheader(f"Recommendations for Age Category: {selected_age_cat}")
                st.write(f"Average Growth Rate: {growth_rate:.1f}%")
                
                recommendations = get_group_recommendations(growth_rate, 'age', selected_age_cat)
                for rec in recommendations:
                    st.write(rec)
            
            with tab2:
                st.subheader("Financial Evolution by Event Type")
                
                # Growth rate visualization
                fig = px.bar(growth_by_event.reset_index(), 
                           x='tipo_ev', 
                           y='mean',
                           title='Average Growth Rate by Event Type',
                           labels={'mean': 'Growth Rate (%)', 'tipo_ev': 'Event Type'},
                           text='mean')
                fig.update_traces(texttemplate='%{text:.1f}%')
                st.plotly_chart(fig)
                
                # Growth rate comparison by attendance
                fig = px.bar(growth_by_event_attendance.reset_index(), 
                           x='tipo_ev',
                           y=['Attended', 'Not Attended'],
                           title='Growth Rate Comparison: Attended vs Not Attended by Event Type',
                           labels={'value': 'Growth Rate (%)', 'tipo_ev': 'Event Type'},
                           barmode='group')
                fig.update_traces(texttemplate='%{y:.1f}%')
                st.plotly_chart(fig)
                
                # Growth direction visualization
                fig = px.bar(growth_direction_by_event.reset_index(), 
                           x='tipo_ev',
                           y=['Positive', 'Negative'],
                           title='Distribution of Growth Direction by Event Type',
                           labels={'value': 'Percentage', 'tipo_ev': 'Event Type'},
                           barmode='stack')
                st.plotly_chart(fig)
                
                # Recommendations for selected event type
                selected_event_type = st.selectbox("Select Event Type for Recommendations", 
                                                 growth_by_event.index)
                growth_rate = growth_by_event.loc[selected_event_type, 'mean']
                
                st.subheader(f"Recommendations for Event Type: {selected_event_type}")
                st.write(f"Average Growth Rate: {growth_rate:.1f}%")
                
                recommendations = get_group_recommendations(growth_rate, 'event', selected_event_type)
                for rec in recommendations:
                    st.write(rec)
                    
            with tab3:
                st.subheader("Financial Evolution by Attendance Status")
                
                # Overall attendance impact
                fig = px.bar(growth_by_attendance.reset_index(), 
                           x='attended',
                           y='mean',
                           title='Average Growth Rate by Attendance Status',
                           labels={'mean': 'Growth Rate (%)', 'attended': 'Attended Event'},
                           text='mean')
                fig.update_traces(texttemplate='%{text:.1f}%')
                fig.update_xaxes(ticktext=['No', 'Yes'], tickvals=[0, 1])
                st.plotly_chart(fig)
                
                # Additional statistics
                st.subheader("Attendance Impact Statistics")
                attended_mean = growth_by_attendance.loc[1, 'mean']
                not_attended_mean = growth_by_attendance.loc[0, 'mean']
                difference = attended_mean - not_attended_mean
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Attended Growth Rate", f"{attended_mean:.1f}%")
                with col2:
                    st.metric("Not Attended Growth Rate", f"{not_attended_mean:.1f}%")
                with col3:
                    st.metric("Difference", f"{difference:.1f}%", 
                             delta=f"{difference:.1f}%")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please ensure 'dataset_limpio.csv' is in the same directory as this script.")

if __name__ == "__main__":
    main()