import streamlit as st
import pickle
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model():
    """Carga el modelo desde un archivo comprimido y verifica su integridad."""
    try:
        with gzip.open('model_trained_regressor.pkl.gz', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def load_scaler():
    """Carga el escalador utilizado en el entrenamiento, si existe."""
    try:
        with gzip.open('scaler.pkl.gz', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception:
        return None

def main():
    st.title("Predicción de Precios de Viviendas en Boston")
    st.write("Introduce las características de la casa para predecir su precio.")

    # Definir nombres de características
    feature_names = [
        "Tasa de criminalidad (CRIM)", "Proporción de terrenos residenciales (ZN)",
        "Proporción de acres de negocios (INDUS)", "Variable ficticia Charles River (CHAS)",
        "Concentración de óxidos de nitrógeno (NOX)", "Número promedio de habitaciones (RM)",
        "Proporción de unidades antiguas (AGE)", "Distancia a centros de empleo (DIS)",
        "Índice de accesibilidad a autopistas (RAD)", "Tasa de impuesto a la propiedad (TAX)",
        "Proporción alumno-maestro (PTRATIO)", "Índice de población afroamericana (B)",
        "Porcentaje de población de estatus bajo (LSTAT)"
    ]
    
    # Crear entradas
    inputs = [st.number_input(f"{feature}", min_value=0.0, format="%.4f") for feature in feature_names]
    
    if st.button("Predecir Precio"):
        model = load_model()
        scaler = load_scaler()
        
        if model is not None:
            try:
                # Convertir a numpy array
                features_array = np.array(inputs).reshape(1, -1)
                
                # Aplicar escalado si es necesario
                if scaler:
                    features_array = scaler.transform(features_array)
                
                # Realizar la predicción
                prediction = model.predict(features_array)
                
                # Mostrar el resultado
                st.success(f"El precio predicho de la casa es: ${prediction[0]:,.2f}")
            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")

if __name__ == "__main__":
    main()
