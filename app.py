# app.py - Taller 6 ICYA3004 - Juan Felipe Muriel (202321092)
# Clasificacion de nuevas muestras con K-NN entrenado sobre clusters de K-Means.
# Pestana 1: dataset canonico Wine (entregable del enunciado). Pestana 2: envios de e-commerce.
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

CARPETA = Path(__file__).parent / "modelos"

st.set_page_config(page_title="Taller 6 - K-Means + K-NN", page_icon=":wine_glass:", layout="wide")
st.title("Clasificacion de segmentos con K-Means + K-NN")
st.caption("ICYA3004 Herramientas de Inteligencia Artificial - Universidad de los Andes - Taller 6")


@st.cache_resource
def cargar(nombre):
    # carga knn, escalador y metadatos de un dataset; devuelve None si faltan archivos
    try:
        knn = joblib.load(CARPETA / f"knn_{nombre}.joblib")
        scaler = joblib.load(CARPETA / f"scaler_{nombre}.joblib")
        meta = joblib.load(CARPETA / f"meta_{nombre}.joblib")
        return knn, scaler, meta
    except FileNotFoundError:
        return None


def formulario(meta, clave):
    # un control numerico por variable, con el rango real de los datos y la mediana como valor inicial
    valores = {}
    cols = st.columns(3)
    for i, v in enumerate(meta["variables"]):
        vmin, vmax, vmed = meta["rangos"][v]
        entero = float(vmin).is_integer() and float(vmax).is_integer() and (vmax - vmin) > 20
        with cols[i % 3]:
            if entero:
                valores[v] = st.number_input(v, min_value=int(vmin), max_value=int(vmax), value=int(vmed), step=1, key=f"{clave}_{v}")
            else:
                paso = round((vmax - vmin) / 100, 4) or 0.01
                valores[v] = st.number_input(v, min_value=float(vmin), max_value=float(vmax), value=float(vmed), step=paso, format="%.2f", key=f"{clave}_{v}")
    return pd.DataFrame([valores])


def predecir(modelo, muestra):
    knn, scaler, meta = modelo
    z = scaler.transform(muestra)
    grupo = int(knn.predict(z)[0])
    proba = knn.predict_proba(z)[0]
    dist, idx = knn.kneighbors(z)
    return grupo, proba, dist[0]


def mostrar_resultado(modelo, muestra, titulo_grupo):
    knn, scaler, meta = modelo
    grupo, proba, dist = predecir(modelo, muestra)
    nombre = meta["nombres"][grupo]
    st.subheader(f"{titulo_grupo}: {nombre}")
    st.write(meta["protocolo"][grupo])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Votos de los {meta['k_vecinos']} vecinos mas cercanos**")
        votos = pd.DataFrame({"grupo": [meta["nombres"][int(c)] for c in knn.classes_],
                              "proporcion": proba})
        st.bar_chart(votos.set_index("grupo"))
        if proba.max() < 0.75:
            st.warning("Muestra de frontera: los vecinos no coinciden. Revisar manualmente.")
        st.caption(f"Distancia media a los vecinos (espacio estandarizado): {dist.mean():.2f}")
    with c2:
        st.markdown("**Muestra ingresada vs centroide del grupo asignado**")
        cent = pd.Series(meta["centroides"][f"C{grupo}"])
        comp = pd.DataFrame({"ingresado": muestra.iloc[0], "centroide del grupo": cent.reindex(muestra.columns)})
        st.dataframe(comp.round(2), width="stretch")


tab_wine, tab_envios = st.tabs(["Wine (dataset canonico)", "Envios e-commerce"])

with tab_wine:
    modelo = cargar("wine")
    if modelo is None:
        st.error("Faltan los archivos del modelo Wine en la carpeta modelos/.")
    else:
        knn, scaler, meta = modelo
        st.markdown(
            "Ingrese las 13 mediciones quimicas de un vino. El K-NN (k = %d) lo asigna a uno de los %d grupos "
            "descubiertos por K-Means sobre los 178 vinos de Piamonte. Los grupos coinciden con los cultivares "
            "reales en %.1f%% de los casos." % (meta["k_vecinos"], len(meta["nombres"]), meta["coincidencia_cultivar"] * 100))
        muestra = formulario(meta, "wine")
        if st.button("Clasificar vino", type="primary"):
            mostrar_resultado(modelo, muestra, "Grupo asignado")

with tab_envios:
    modelo = cargar("envios")
    if modelo is None:
        st.error("Faltan los archivos del modelo de envios en la carpeta modelos/.")
    else:
        knn, scaler, meta = modelo
        st.markdown(
            "Ingrese los datos de un envio nuevo. El K-NN (k = %d) lo asigna a uno de los %d segmentos operativos "
            "descubiertos por K-Means sobre 10,999 envios, y muestra el protocolo de servicio sugerido." % (meta["k_vecinos"], len(meta["nombres"])))
        muestra = formulario(meta, "envios")
        if st.button("Clasificar envio", type="primary"):
            mostrar_resultado(modelo, muestra, "Segmento asignado")

st.divider()
st.caption("Modelos entrenados con scikit-learn. El escalador se ajusto solo con los datos de entrenamiento y se aplica a cada muestra nueva antes de buscar vecinos.")
