#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

def clean_data(df):
    """Clean and preprocess the uploaded data"""
    with st.spinner('Cleaning data...'):
        # Show initial data statistics
        st.subheader("Initial Data Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Missing values before cleaning:")
            st.write(df.isnull().sum())
        with col2:
            st.write("Data types before cleaning:")
            st.write(df.dtypes)

        # Clean event types
        tipo_ev_map = {
            10: "Alimentación y estilo de vida",
            11: "Arte",
            12: "Charlas",
            13: "Deportes",
            14: "Fondos de inversión",
            15: "Música",
            16: "Planes universitarios",
            17: "Seguros de ahorro",
            18: "Tecnología"
        }
        
        df["tipo_ev"] = pd.to_numeric(df["tipo_ev"], errors="coerce")
        df = df[df["tipo_ev"].between(10, 18)]
        df["tipo_ev"] = df["tipo_ev"].map(tipo_ev_map)
        
        # Fix event names
        errores_nombre_ev = {
            "Estrateggias paraacum ular fondos": "Estrategias para acumular fondos"
        }
        df["nombre_ev"] = df["nombre_ev"].replace(errores_nombre_ev)
        
        # Fill missing values
        df["prod_as_ev"].fillna(df["nombre_ev"], inplace=True)
        
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna("N/A", inplace=True)
        
        # Normalize abbreviations
        abreviaturas = {
            "ZGZ": "Zaragoza", "BCN": "Barcelona", "MAD": "Madrid", "SEV": "Sevilla",
            "BIL": "Bilbao", "zaragoza": "Zaragoza", "madrid": "Madrid",
            "bilbao": "Bilbao", "sevilla": "Sevilla"
        }
        df["loc_ev"] = df["loc_ev"].replace(abreviaturas)
        df["prov"] = df["prov"].replace(abreviaturas)
        
        # Normalize gender
        df["sexo"] = df["sexo"].replace({"H": "Hombre", "M": "Mujer"})
        
        # Format amounts
        df["importe_activos_0"] = pd.to_numeric(df["importe_activos_0"], errors='coerce')
        df["importe_activos_1"] = pd.to_numeric(df["importe_activos_1"], errors='coerce')
        df["importe_activos_0"] = df["importe_activos_0"].astype(float).round(2)
        df["importe_activos_1"] = df["importe_activos_1"].astype(float).round(2)
        
        # Calculate age category
        def calcular_categoria_edad(fecha_nac):
            if pd.isna(fecha_nac):
                return np.nan
            
            try:
                fecha_nac = datetime.strptime(fecha_nac, "%d/%m/%Y")
                edad = datetime.today().year - fecha_nac.year - ((datetime.today().month, datetime.today().day) < (fecha_nac.month, fecha_nac.day))
                
                if 18 <= edad <= 25:
                    return "18-25 años"
                elif 26 <= edad <= 35:
                    return "26-35 años"
                elif 36 <= edad <= 50:
                    return "35-50 años"
                elif 51 <= edad <= 65:
                    return "51-65 años"
                else:
                    return "> 65 años"
            except:
                return np.nan
        
        df["cat_edad"] = df["fecha_nac"].apply(calcular_categoria_edad)
        
        # Show final data statistics
        st.subheader("Final Data Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Missing values after cleaning:")
            st.write(df.isnull().sum())
        with col2:
            st.write("Data types after cleaning:")
            st.write(df.dtypes)
        
        # Create binary target variable
        df['attended'] = (df['estado'] == 'Asistido').astype(int)
        
        return df

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
    # [Previous recommendation function remains the same]
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
        
        if segment_type == 'age':
            if 'joven' in segment_value.lower() or '18-25' in segment_value or '26-35' in segment_value:
                recommendations.extend([
                    "🔹 Productos específicos para jóvenes inversores:",
                    "  • Planes de ahorro sistemático",
                    "  • Fondos con mayor componente de renta variable"
                ])
            elif 'adulto' in segment_value.lower() or '35-50' in segment_value:
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
    
    return recommendations

def main():
    st.title("Customer Analytics Dashboard")
    st.write("Upload your CSV file for cleaning and analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read and clean the data
            df = pd.read_csv(uploaded_file)
            st.write("Original Data Preview:")
            st.write(df.head())
            
            cleaned_df = clean_data(df)
            
            # Add download button for cleaned data
            st.download_button(
                label="Download cleaned data as CSV",
                data=cleaned_df.to_csv(index=False).encode('utf-8'),
                file_name='cleaned_data.csv',
                mime='text/csv',
            )
            
            # Analysis section
            st.sidebar.header("Analysis Options")
            analysis_type = st.sidebar.selectbox(
                "Choose Analysis",
                ["Data Quality Report", "Attendance Overview", "Profile Analysis", "Financial Evolution"]
            )
            
            if analysis_type == "Data Quality Report":
                st.header("Data Quality Report")
                
                # Data completeness visualization
                completeness = (cleaned_df.count() / len(cleaned_df) * 100).round(2)
                fig = px.bar(
                    x=completeness.index, 
                    y=completeness.values,
                    labels={'x': 'Column', 'y': 'Completeness (%)'},
                    title="Data Completeness by Column"
                )
                st.plotly_chart(fig)
                
                # Value distributions
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Event Types Distribution")
                    fig = px.pie(cleaned_df, names='tipo_ev', title="Event Type Distribution")
                    st.plotly_chart(fig)
                
                with col2:
                    st.subheader("Age Categories Distribution")
                    fig = px.pie(cleaned_df, names='cat_edad', title="Age Category Distribution")
                    st.plotly_chart(fig)
            
            elif analysis_type == "Attendance Overview":
                st.header("Event Attendance Overview")
                
                attendance_rate = cleaned_df['attended'].mean() * 100
                st.metric("Overall Attendance Rate", f"{attendance_rate:.1f}%")
                
                # Event type attendance
                event_attendance = cleaned_df.groupby('tipo_ev')['attended'].mean().sort_values(ascending=False) * 100
                fig = px.bar(
                    event_attendance,
                    title="Attendance Rate by Event Type",
                    labels={'index': 'Event Type', 'value': 'Attendance Rate (%)'},
                    text=event_attendance.round(1)
                )
                fig.update_traces(texttemplate='%{text}%')
                st.plotly_chart(fig)
            
            elif analysis_type == "Profile Analysis":
                st.header("Customer Profile Analysis")
                
                # Age attendance
                age_attendance = cleaned_df.groupby('cat_edad')['attended'].mean() * 100
                fig = px.bar(
                    age_attendance,
                    title="Attendance Rate by Age Category",
                    labels={'index': 'Age Category', 'value': 'Attendance Rate (%)'},
                    text=age_attendance.round(1)
                )
                fig.update_traces(texttemplate='%{text}%')
                st.plotly_chart(fig)
                
                # Gender analysis
                gender_attendance = cleaned_df.groupby('sexo')['attended'].mean() * 100
                fig = px.bar(
                    gender_attendance,
                    title="Attendance Rate by Gender",
                    labels={'index': 'Gender', 'value': 'Attendance Rate (%)'},
                    text=gender_attendance.round(1)
                )
                fig.update_traces(texttemplate='%{text}%')
                st.plotly_chart(fig)
            
            else:  # Financial Evolution
                st.header("Financial Evolution Analysis")
                
                # Process financial data
                (df, growth_by_age, growth_by_event, growth_direction_by_age, 
                 growth_direction_by_event, growth_by_attendance, growth_by_age_attendance,
                 growth_by_event_attendance) = analyze_financial_evolution(cleaned_df)
                
                tab1, tab2, tab3 = st.tabs(["By Age Category", "By Event Type", "By Attendance Status"])
                
                with tab1:
                    st.subheader("Financial Evolution by Age Category")
                    
                    # Growth rate visualization
                    fig = px.bar(
                        growth_by_age.reset_index(), 
                        x='cat_edad', 
                        y='mean',
                        title='Average Growth Rate by Age Category',
                        labels={'mean': 'Growth Rate (%)', 'cat_edad': 'Age Category'},
                        text='mean'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%')
                    st.plotly_chart(fig)
                    
                    # Growth rate comparison
                    fig = px.bar(
                        growth_by_age_attendance.reset_index(), 
                        x='cat_edad',
                        y=['Attended', 'Not Attended'],
                        title='Growth Rate Comparison: Attended vs Not Attended by Age Category',
                        labels={'value': 'Growth Rate (%)', 'cat_edad': 'Age Category'},
                        barmode='group'
                    )
                    fig.update_traces(texttemplate='%{y:.1f}%')
                    st.plotly_chart(fig)
                    
                    # Recommendations
                    selected_age_cat = st.selectbox(
                        "Select Age Category for Recommendations", 
                        growth_by_age.index
                    )
                    growth_rate = growth_by_age.loc[selected_age_cat, 'mean']
                    
                    st.subheader(f"Recommendations for Age Category: {selected_age_cat}")
                    st.write(f"Average Growth Rate: {growth_rate:.1f}%")
                    
                    recommendations = get_group_recommendations(growth_rate, 'age', selected_age_cat)
                    for rec in recommendations:
                        st.write(rec)
                
                with tab2:
                    st.subheader("Financial Evolution by Event Type")
                    
                    # Growth rate visualization
                    fig = px.bar(
                        growth_by_event.reset_index(), 
                        x='tipo_ev', 
                        y='mean',
                        title='Average Growth Rate by Event Type',
                        labels={'mean': 'Growth Rate (%)', 'tipo_ev': 'Event Type'},
                        text='mean'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}%')
                    st.plotly_chart(fig)
                    
                    # Growth rate comparison
                    fig = px.bar(
                        growth_by_event_attendance.reset_index(), 
                        x='tipo_ev',
                        y=['Attended', 'Not Attended'],
                        title='Growth Rate Comparison: Attended vs Not Attended by Event Type',
                        labels={'value': 'Growth Rate (%)', 'tipo_ev': 'Event Type'},
                        barmode='group'
                    )
                    fig.update_traces(texttemplate='%{y:.1f}%')
                    st.plotly_chart(fig)
                    
                    # Growth direction visualization
                    fig = px.bar(
                        growth_direction_by_event.reset_index(), 
                        x='tipo_ev',
                        y=['Positive', 'Negative'],
                        title='Distribution of Growth Direction by Event Type',
                        labels={'value': 'Percentage', 'tipo_ev': 'Event Type'},
                        barmode='stack'
                    )
                    st.plotly_chart(fig)
                    
                    # Recommendations for selected event type
                    selected_event_type = st.selectbox(
                        "Select Event Type for Recommendations", 
                        growth_by_event.index
                    )
                    growth_rate = growth_by_event.loc[selected_event_type, 'mean']
                    
                    st.subheader(f"Recommendations for Event Type: {selected_event_type}")
                    st.write(f"Average Growth Rate: {growth_rate:.1f}%")
                    
                    recommendations = get_group_recommendations(growth_rate, 'event', selected_event_type)
                    for rec in recommendations:
                        st.write(rec)
                
                with tab3:
                    st.subheader("Financial Evolution by Attendance Status")
                    
                    # Overall attendance impact
                    fig = px.bar(
                        growth_by_attendance.reset_index(), 
                        x='attended',
                        y='mean',
                        title='Average Growth Rate by Attendance Status',
                        labels={'mean': 'Growth Rate (%)', 'attended': 'Attended Event'},
                        text='mean'
                    )
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
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your CSV file has the correct format.")
    
    else:
        st.info("Please upload a CSV file to begin the analysis.")
        
        # Show example of expected CSV structure
        st.subheader("Expected CSV Structure")
        example_data = {
            'id_ev': ['881'],
            'tipo_ev': ['16'],
            'nombre_ev': ['Estrategias para acumular fondos'],
            'prod_as_ev': ['Seguro de Ahorro Educativo'],
            'fecha_ev': ['06/07/2018'],
            'loc_ev': ['BCN'],
            'estado': ['Asistido'],
            'id_cli': ['400237'],
            'fecha_nac': ['21/10/1971'],
            'sexo': ['H'],
            'prov': ['BIL'],
            'importe_activos_0': [104623.0],
            'importe_activos_1': [185796.13]
        }
        st.dataframe(pd.DataFrame(example_data))

if __name__ == "_main_":
    main()