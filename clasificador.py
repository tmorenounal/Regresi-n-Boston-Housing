import streamlit as st
import pickle
import gzip
import numpy as np

def load_model():
    """Carga el modelo desde un archivo comprimido."""
    try:
        with gzip.open('model_trained_regressor.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def main():
    st.title("Predicción de Precios de Viviendas en Boston")
    st.write("Introduce las características de la casa para predecir su precio.")

    # Definir nombres de características y valores iniciales
    feature_names = [
        "Tasa de criminalidad (CRIM)", "Proporción de terrenos residenciales (ZN)",
        "Proporción de acres de negocios (INDUS)", "Variable ficticia Charles River (CHAS)",
        "Concentración de óxidos de nitrógeno (NOX)", "Número promedio de habitaciones (RM)",
        "Proporción de unidades antiguas (AGE)", "Distancia a centros de empleo (DIS)",
        "Índice de accesibilidad a autopistas (RAD)", "Tasa de impuesto a la propiedad (TAX)",
        "Proporción alumno-maestro (PTRATIO)", "Índice de población afroamericana (B)",
        "Porcentaje de población de estatus bajo (LSTAT)"
    ]
    
    # Crear entradas con valores por defecto y restricciones
    inputs = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
        inputs.append(value)
    
    if st.button("Predecir Precio"):
        model = load_model()
        if model is not None:
            try:
                # Convertir a numpy array y reformatear para la predicción
                features_array = np.array(inputs).reshape(1, -1)
                
                # Realizar la predicción
                prediction = model.predict(features_array)
                
                # Mostrar el resultado con formato adecuado
                st.success(f"El precio predicho de la casa es: ${prediction[0]:,.2f}")
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

if __name__ == "__main__":
    main()

