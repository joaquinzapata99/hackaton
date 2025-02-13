import pandas as pd
import numpy as np
from datetime import datetime

# 1. Cargar el dataset
df = pd.read_csv("hackaton.csv", encoding="utf-8")

# 2. Corregir tipo_ev con nombres adecuados y valores válidos
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

df["tipo_ev"] = pd.to_numeric(df["tipo_ev"], errors="coerce")  # Convertir a numérico
df = df[df["tipo_ev"].between(10, 18)]  # Filtrar valores fuera del rango
df["tipo_ev"] = df["tipo_ev"].map(tipo_ev_map)  # Reemplazar por nombres

# 3. Corregir errores de escritura en nombre_ev
errores_nombre_ev = {
    "Estrateggias paraacum ular fondos": "Estrategias para acumular fondos"
}
df["nombre_ev"] = df["nombre_ev"].replace(errores_nombre_ev)

# 4. Rellenar valores faltantes en prod_as_ev con nombre_ev
df["prod_as_ev"].fillna(df["nombre_ev"], inplace=True)

# 5. Normalizar abreviaturas en loc_ev y prov
abreviaturas = {
    "ZGZ": "Zaragoza", "BCN": "Barcelona", "MAD": "Madrid", "SEV": "Sevilla",
    "BIL": "Bilbao", "zaragoza": "Zaragoza", "madrid": "Madrid",
    "bilbao": "Bilbao", "sevilla": "Sevilla"
}
df["loc_ev"] = df["loc_ev"].replace(abreviaturas)
df["prov"] = df["prov"].replace(abreviaturas)

# 6. Cambiar H y M en sexo a Hombre y Mujer
df["sexo"] = df["sexo"].replace({"H": "Hombre", "M": "Mujer"})

# 7. Asegurar que importe_activos_0 y importe_activos_1 sean float con 2 decimales
df["importe_activos_0"] = df["importe_activos_0"].astype(float).round(2)
df["importe_activos_1"] = df["importe_activos_1"].astype(float).round(2)

# 8. Calcular edad y categorizarla
def calcular_categoria_edad(fecha_nac):
    if pd.isna(fecha_nac):
        return np.nan  # Si la fecha es nula, dejamos NaN
    
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

df["cat_edad"] = df["fecha_nac"].apply(calcular_categoria_edad)

# 📌 9. Guardar el dataset limpio
df.to_csv("dataset_limpio.csv", index=False, encoding="utf-8")

print(" Dataset limpiado y guardado como dataset_limpio.csv")