import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import joblib
import os


# Definir los mapeos para las variables (debe coincidir con el entrenamiento)
mapeo_horas_estudio = {"Alta": 1, "Baja": 0}
mapeo_asistencia = {"Buena": 1, "Mala": 0}
mapeo_resultado = {1: "Sí", 0: "No"} # Mapeo inverso para la predicción

# Crear un DataFrame dummy para entrenar y obtener el mapeo de LabelEncoder
# Esto es necesario para que el LabelEncoder tenga el fit_transform hecho
data_dummy = {
    "Horas de Estudio": ["Alta", "Baja", "Baja", "Alta", "Alta"],
    "Asistencia": ["Buena", "Buena", "Mala", "Mala", "Buena"],
    "Resultado": ["Sí", "No", "No", "Sí", "Sí"]
}
df_dummy = pd.DataFrame(data_dummy)

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in df_dummy.columns:
  label_encoders[column] = LabelEncoder()
  df_dummy[column] = label_encoders[column].fit_transform(df_dummy[column])

# Separar características (X) y etiqueta (y)
X_dummy = df_dummy[["Horas de Estudio", "Asistencia"]]
y_dummy = df_dummy["Resultado"]

# Entrenar el modelo (solo para tener un modelo guardado si no existe)
model = MultinomialNB()
model.fit(X_dummy, y_dummy)

# Guardar el modelo si no existe
filename = '/content/modelo_naive_bayes.jb'
if not os.path.exists(filename):
    joblib.dump(model, filename)

# Cargar el modelo entrenado
try:
    model = joblib.load(filename)
except FileNotFoundError:
    st.error(f"Error: El archivo del modelo '{filename}' no se encontró.")
    st.stop()


st.title("Predicción de Clase")

# Subtítulo en rojo
st.markdown('<p style="color:red;">Elaborado por: JuanMaVelasco</p>', unsafe_allow_html=True)


st.sidebar.header("Ingrese los valores de las variables")

# Inputs del usuario para las variables
horas_estudio_input = st.sidebar.selectbox("Horas de Estudio", list(mapeo_horas_estudio.keys()))
asistencia_input = st.sidebar.selectbox("Asistencia", list(mapeo_asistencia.keys()))

# Botón para hacer la predicción
if st.sidebar.button("Predecir"):
    # Codificar la nueva observación
    nueva_observacion_data = {
        "Horas de Estudio": [horas_estudio_input],
        "Asistencia": [asistencia_input]
    }
    nueva_observacion_df = pd.DataFrame(nueva_observacion_data)

    # Usar los LabelEncoders fiteados en el dummy DataFrame
    for column in nueva_observacion_df.columns:
        try:
            nueva_observacion_df[column] = label_encoders[column].transform(nueva_observacion_df[column])
        except ValueError as e:
             st.error(f"Error al codificar la entrada '{nueva_observacion_df[column].iloc[0]}' en la columna '{column}': {e}")
             st.stop()


    # Realizar la predicción
    prediccion_codificada = model.predict(nueva_observacion_df)

    # Decodificar la predicción
    prediccion_decodificada = label_encoders["Resultado"].inverse_transform(prediccion_codificada)

    # Mostrar el resultado con caritas
    st.subheader("Resultado de la Predicción:")
    if prediccion_decodificada[0] == "Sí":
        st.success(f"¡Felicitaciones Aprueba! 😊")
    else:
        st.error(f"No aprueba 😔")

