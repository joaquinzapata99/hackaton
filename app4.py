#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

# Cache the data cleaning function to avoid recomputing
@st.cache_data
def clean_data(df):
    """Clean and preprocess the uploaded data"""
    # Basic data type corrections and mapping
    tipo_ev_map = {
        10: "Alimentación", 11: "Arte", 12: "Charlas", 
        13: "Deportes", 14: "Fondos", 15: "Música",
        16: "Planes", 17: "Seguros", 18: "Tecnología"
    }
    
    df["tipo_ev"] = pd.to_numeric(df["tipo_ev"], errors="coerce")
    df = df[df["tipo_ev"].between(10, 18)]
    df["tipo_ev"] = df["tipo_ev"].map(tipo_ev_map)
    
    # Basic cleaning
    df["prod_as_ev"].fillna(df["nombre_ev"], inplace=True)
    df["importe_activos_0"] = pd.to_numeric(df["importe_activos_0"], errors='coerce')
    df["importe_activos_1"] = pd.to_numeric(df["importe_activos_1"], errors='coerce')
    
    # Calculate age category
    def get_age_category(fecha_nac):
        try:
            fecha_nac = datetime.strptime(fecha_nac, "%d/%m/%Y")
            edad = datetime.today().year - fecha_nac.year
            if edad <= 25: return "18-25"
            elif edad <= 35: return "26-35"
            elif edad <= 50: return "36-50"
            elif edad <= 65: return "51-65"
            return ">65"
        except:
            return "N/A"
    
    df["cat_edad"] = df["fecha_nac"].apply(get_age_category)
    df['attended'] = (df['estado'] == 'Asistido').astype(int)
    
    return df

# Cache financial calculations
@st.cache_data
def analyze_financial_evolution(df):
    df['growth_rate'] = ((df['importe_activos_1'] - df['importe_activos_0']) / df['importe_activos_0'] * 100)
    
    # Basic aggregations
    growth_by_age = df.groupby('cat_edad')['growth_rate'].mean().round(2)
    growth_by_event = df.groupby('tipo_ev')['growth_rate'].mean().round(2)
    growth_by_attendance = df.groupby('attended')['growth_rate'].mean().round(2)
    
    return growth_by_age, growth_by_event, growth_by_attendance

def get_recommendations(growth_rate):
    if growth_rate > 0:
        return [
            "🔹 Inversión en Fondos",
            "🔹 Seguros de ahorro",
            "🔹 Productos con horizonte temporal"
        ]
    return [
        "🔹 Asesoría Financiera",
        "🔹 Análisis patrimonial",
        "🔹 Productos de financiación"
    ]

def main():
    st.set_page_config(page_title="Customer Analytics", layout="wide")
    st.title("Customer Analytics Dashboard")
    
    uploaded_file = st.file_uploader("Cargar archivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            cleaned_df = clean_data(df)
            
            # Download button for cleaned data
            st.download_button(
                "Descargar datos limpios (CSV)",
                cleaned_df.to_csv(index=False).encode('utf-8'),
                "datos_limpios.csv",
                "text/csv"
            )
            
            # Analysis sections
            analysis = st.selectbox(
                "Seleccionar Análisis",
                ["Vista General", "Evolución Financiera"]
            )
            
            if analysis == "Vista General":
                col1, col2 = st.columns(2)
                
                with col1:
                    # Attendance by event type
                    event_attendance = cleaned_df.groupby('tipo_ev')['attended'].mean() * 100
                    fig = px.bar(
                        event_attendance,
                        title="Asistencia por Tipo de Evento",
                        labels={'value': 'Tasa (%)', 'tipo_ev': 'Tipo'},
                        text=event_attendance.round(1)
                    )
                    fig.update_traces(texttemplate='%{text}%')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Attendance by age
                    age_attendance = cleaned_df.groupby('cat_edad')['attended'].mean() * 100
                    fig = px.bar(
                        age_attendance,
                        title="Asistencia por Edad",
                        labels={'value': 'Tasa (%)', 'cat_edad': 'Edad'},
                        text=age_attendance.round(1)
                    )
                    fig.update_traces(texttemplate='%{text}%')
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Evolución Financiera
                growth_by_age, growth_by_event, growth_by_attendance = analyze_financial_evolution(cleaned_df)
                
                tab1, tab2 = st.tabs(["Por Categoría", "Por Asistencia"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Growth by age
                        fig = px.bar(
                            growth_by_age,
                            title='Crecimiento por Edad',
                            labels={'value': '%', 'cat_edad': 'Edad'},
                            text=growth_by_age.round(1)
                        )
                        fig.update_traces(texttemplate='%{text}%')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Growth by event
                        fig = px.bar(
                            growth_by_event,
                            title='Crecimiento por Evento',
                            labels={'value': '%', 'tipo_ev': 'Tipo'},
                            text=growth_by_event.round(1)
                        )
                        fig.update_traces(texttemplate='%{text}%')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    selected_category = st.selectbox("Seleccionar Categoría", growth_by_age.index)
                    growth_rate = growth_by_age[selected_category]
                    st.write(f"Crecimiento: {growth_rate:.1f}%")
                    for rec in get_recommendations(growth_rate):
                        st.write(rec)
                
                with tab2:
                    # Attendance impact
                    attended_growth = growth_by_attendance[1]
                    not_attended_growth = growth_by_attendance[0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Asistieron", f"{attended_growth:.1f}%")
                    with col2:
                        st.metric("No Asistieron", f"{not_attended_growth:.1f}%")
                    
                    fig = px.bar(
                        growth_by_attendance,
                        title='Crecimiento por Asistencia',
                        labels={'value': '%', 'attended': 'Asistió'},
                        text=growth_by_attendance.round(1)
                    )
                    fig.update_traces(texttemplate='%{text}%')
                    fig.update_xaxes(ticktext=['No', 'Sí'], tickvals=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Asegúrese de que el archivo CSV tiene el formato correcto.")
    
    else:
        st.info("Por favor, cargue un archivo CSV para comenzar el análisis.")
        
        # Example structure
        st.subheader("Estructura CSV Esperada")
        example_data = {
            'tipo_ev': ['16'],
            'nombre_ev': ['Estrategias fondos'],
            'fecha_ev': ['06/07/2018'],
            'estado': ['Asistido'],
            'fecha_nac': ['21/10/1971'],
            'importe_activos_0': [104623.0],
            'importe_activos_1': [185796.13]
        }
        st.dataframe(pd.DataFrame(example_data))

if __name__ == "_main_":
    main()