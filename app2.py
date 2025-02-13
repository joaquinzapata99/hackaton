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

        # 2. Corregir tipo_ev
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
        
        # 3. Corregir errores de escritura en nombre_ev
        errores_nombre_ev = {
            "Estrateggias paraacum ular fondos": "Estrategias para acumular fondos"
        }
        df["nombre_ev"] = df["nombre_ev"].replace(errores_nombre_ev)
        
        # 4. Rellenar valores faltantes
        df["prod_as_ev"].fillna(df["nombre_ev"], inplace=True)
        
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna("N/A", inplace=True)
        
        # 5. Normalizar abreviaturas
        abreviaturas = {
            "ZGZ": "Zaragoza", "BCN": "Barcelona", "MAD": "Madrid", "SEV": "Sevilla",
            "BIL": "Bilbao", "zaragoza": "Zaragoza", "madrid": "Madrid",
            "bilbao": "Bilbao", "sevilla": "Sevilla"
        }
        df["loc_ev"] = df["loc_ev"].replace(abreviaturas)
        df["prov"] = df["prov"].replace(abreviaturas)
        
        # 6. Normalizar sexo
        df["sexo"] = df["sexo"].replace({"H": "Hombre", "M": "Mujer"})
        
        # 7. Formatear importes
        df["importe_activos_0"] = pd.to_numeric(df["importe_activos_0"], errors='coerce')
        df["importe_activos_1"] = pd.to_numeric(df["importe_activos_1"], errors='coerce')
        df["importe_activos_0"] = df["importe_activos_0"].astype(float).round(2)
        df["importe_activos_1"] = df["importe_activos_1"].astype(float).round(2)
        
        # 8. Calcular categoría de edad
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

def main():
    st.title("Customer Analytics Dashboard")
    st.write("Upload your CSV file for cleaning and analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Original Data Preview:")
            st.write(df.head())
            
            # Clean the data
            cleaned_df = clean_data(df)
            
            # Add download button for cleaned data
            st.download_button(
                label="Download cleaned data as CSV",
                data=cleaned_df.to_csv(index=False).encode('utf-8'),
                file_name='cleaned_data.csv',
                mime='text/csv',
            )
            
            # Continue with the analysis
            st.header("Analysis Options")
            analysis_type = st.selectbox(
                "Choose Analysis",
                ["Data Quality Report", "Attendance Overview", "Financial Evolution"]
            )
            
            if analysis_type == "Data Quality Report":
                st.subheader("Data Quality Report")
                
                # Show data completeness
                st.write("Data Completeness")
                completeness = (cleaned_df.count() / len(cleaned_df) * 100).round(2)
                fig = px.bar(
                    x=completeness.index, 
                    y=completeness.values,
                    labels={'x': 'Column', 'y': 'Completeness (%)'},
                    title="Data Completeness by Column"
                )
                st.plotly_chart(fig)
                
                # Show value distributions
                st.write("Value Distributions")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Event Types Distribution")
                    st.write(cleaned_df['tipo_ev'].value_counts())
                with col2:
                    st.write("Age Categories Distribution")
                    st.write(cleaned_df['cat_edad'].value_counts())
            
            elif analysis_type == "Attendance Overview":
                st.subheader("Event Attendance Overview")
                
                attendance_rate = cleaned_df['attended'].mean() * 100
                st.metric("Overall Attendance Rate", f"{attendance_rate:.1f}%")
                
                # Attendance by event type
                event_attendance = cleaned_df.groupby('tipo_ev')['attended'].mean().sort_values(ascending=False)
                fig = px.bar(
                    event_attendance,
                    title="Attendance Rate by Event Type",
                    labels={'index': 'Event Type', 'value': 'Attendance Rate'}
                )
                st.plotly_chart(fig)
                
                # Attendance by age category
                age_attendance = cleaned_df.groupby('cat_edad')['attended'].mean().sort_values(ascending=False)
                fig = px.bar(
                    age_attendance,
                    title="Attendance Rate by Age Category",
                    labels={'index': 'Age Category', 'value': 'Attendance Rate'}
                )
                st.plotly_chart(fig)
            
            else:  # Financial Evolution
                st.subheader("Financial Evolution Analysis")
                
                # Calculate growth rates
                cleaned_df['growth_rate'] = ((cleaned_df['importe_activos_1'] - cleaned_df['importe_activos_0']) / 
                                           cleaned_df['importe_activos_0'] * 100)
                
                # Overall growth distribution
                fig = px.histogram(
                    cleaned_df,
                    x='growth_rate',
                    title="Distribution of Growth Rates",
                    labels={'growth_rate': 'Growth Rate (%)'}
                )
                st.plotly_chart(fig)
                
                # Growth by attendance status
                growth_by_attendance = cleaned_df.groupby('attended')['growth_rate'].mean()
                fig = px.bar(
                    x=['Not Attended', 'Attended'],
                    y=growth_by_attendance.values,
                    title="Average Growth Rate by Attendance Status",
                    labels={'x': 'Attendance Status', 'y': 'Average Growth Rate (%)'}
                )
                st.plotly_chart(fig)
                
                # Growth by age category
                growth_by_age = cleaned_df.groupby('cat_edad')['growth_rate'].mean().sort_values(ascending=False)
                fig = px.bar(
                    growth_by_age,
                    title="Average Growth Rate by Age Category",
                    labels={'index': 'Age Category', 'value': 'Average Growth Rate (%)'}
                )
                st.plotly_chart(fig)
                
                # Growth by event type
                growth_by_event = cleaned_df.groupby('tipo_ev')['growth_rate'].mean().sort_values(ascending=False)
                fig = px.bar(
                    growth_by_event,
                    title="Average Growth Rate by Event Type",
                    labels={'index': 'Event Type', 'value': 'Average Growth Rate (%)'}
                )
                st.plotly_chart(fig)
        
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

if __name__ == "__main__":
    main()