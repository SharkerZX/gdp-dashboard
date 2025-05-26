import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util
import os
import requests

# Configuración de modelos
MODELOS = {
    "phi-2": ("Modelos/phi-2.Q4_K_M.gguf", "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"),
    "tinyllama": ("Modelos/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    "mistral": ("Modelos/mistral-7b-instruct-v0.1.Q4_0.gguf", "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_0.gguf"),
    "llama2": ("Modelos/llama-2-7b-chat.Q4_K_M.gguf", "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf")
}

# Descargar modelos si no existen
def descargar_modelo_si_no_existe(ruta, url):
    if not os.path.exists(ruta):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(ruta, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Modelo descargado: {ruta}")

# Cargar prompts y respuestas esperadas
@st.cache_data
def cargar_datos():
    prompts = open("prompts.txt", encoding="utf-8").read().splitlines()
    respuestas_df = pd.read_csv("respuestas_esperadas.csv")
    respuestas_esperadas = dict(zip(respuestas_df.Prompt, respuestas_df.RespuestaEsperada))
    return prompts, respuestas_esperadas

# Inicializar modelo de similitud
modelo_similitud = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Evaluación de modelos
@st.cache_data
def evaluar_modelos(prompts, respuestas_esperadas):
    resultados = []
    for nombre, (ruta, url) in MODELOS.items():
        descargar_modelo_si_no_existe(ruta, url)
        llm = Llama(model_path=ruta)
        for prompt in prompts:
            inicio = time.time()
            output = llm(prompt, max_tokens=100)
            latencia = round(time.time() - inicio, 2)

            respuesta = output["choices"][0]["text"].strip()
            longitud = len(respuesta.split())

            esperada = respuestas_esperadas.get(prompt, "")
            sim_semantica = round(util.cos_sim(
                modelo_similitud.encode(respuesta),
                modelo_similitud.encode(esperada)
            ).item(), 4) if esperada else None

            cobertura = round(util.cos_sim(
                modelo_similitud.encode(respuesta),
                modelo_similitud.encode(prompt)
            ).item(), 4)

            resultados.append({
                "Modelo": nombre,
                "Prompt": prompt,
                "Respuesta": respuesta,
                "Latencia (s)": latencia,
                "SimilitudSemantica": sim_semantica,
                "Longitud": longitud,
                "CoberturaPrompt": cobertura
            })
    return pd.DataFrame(resultados)

# Streamlit UI
st.title("Evaluación Comparativa de Modelos LLM Cuantizados")

if st.button("Iniciar Evaluación"):
    with st.spinner("Evaluando modelos..."):
        prompts, respuestas_esperadas = cargar_datos()
        df_resultados = evaluar_modelos(prompts, respuestas_esperadas)

        st.success("Evaluación completada")
        st.dataframe(df_resultados)

        for col in ["Latencia (s)", "SimilitudSemantica", "Longitud", "CoberturaPrompt"]:
            df_resultados[col] = pd.to_numeric(df_resultados[col], errors="coerce")

        resumen = df_resultados.groupby("Modelo")[["Latencia (s)", "SimilitudSemantica", "Longitud", "CoberturaPrompt"]].mean().reset_index()
        resumen = resumen.round(3)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        sns.barplot(data=resumen, x="Modelo", y="Latencia (s)", ax=axes[0, 0], palette="Blues_d")
        axes[0, 0].set_title("Latencia promedio (s)")

        sns.barplot(data=resumen, x="Modelo", y="SimilitudSemantica", ax=axes[0, 1], palette="Greens_d")
        axes[0, 1].set_title("Precisión semántica promedio")

        sns.barplot(data=resumen, x="Modelo", y="Longitud", ax=axes[1, 0], palette="Oranges_d")
        axes[1, 0].set_title("Longitud promedio de respuesta")

        sns.barplot(data=resumen, x="Modelo", y="CoberturaPrompt", ax=axes[1, 1], palette="Purples_d")
        axes[1, 1].set_title("Cobertura promedio del prompt")

        plt.tight_layout()
        st.pyplot(fig)
