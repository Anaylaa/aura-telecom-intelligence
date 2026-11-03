"""
=============================================================================
ARQUIVO: app.py
O QUE FAZ: Interface principal em Streamlit, Dashboard BI, ML e gerador de PDF.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.recommender import recomendar_planos
from src.db import carregar_catalogo_do_supabase
from src.predictor import prever_churn_cliente
from src.pdf_generator import gerar_pdf_proposta

st.set_page_config(page_title="Claro | Aura Telecom Intelligence", page_icon="assets/claro-esfera.png", layout="wide")

st.markdown("""
<style>
    :root {
        --claro-red: #DA291C;
        --claro-red-dark: #AF272F;
        --claro-charcoal: #2D2926;
        --claro-gray: #A7A8AA;
        --canvas: #F5F5F5;
        --surface: #FFFFFF;
        --line: #E2E2E2;
    }
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--canvas);
        color: var(--claro-charcoal);
        font-family: "AMX", Arial, sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(245, 245, 245, 0.96); }
    .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3, h4 { color: var(--claro-charcoal) !important; font-weight: 700; letter-spacing: -0.02em; }
    p, label, [data-testid="stMarkdownContainer"] { color: var(--claro-charcoal); }
    hr { border-color: var(--line); }

    .brand-title {
        margin: 0;
        color: var(--claro-charcoal);
        font-family: "AMX", Arial, sans-serif;
        font-size: clamp(1.4rem, 2.4vw, 2rem);
        font-weight: 700;
        letter-spacing: -0.04em;
        line-height: 1.12;
    }
    .brand-subtitle { margin: 0.45rem 0 0 0; color: #686868 !important; font-size: 1rem; }
    [data-testid="stImage"] img { display: block; }

    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background: var(--claro-charcoal);
        border-right: 1px solid #413B37;
    }
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button {
        background: #FFFFFF !important;
    }
    [data-testid="collapsedControl"] button {
        color: var(--claro-red) !important;
        border: 1px solid #F0D2CF !important;
        border-radius: 8px !important;
    }
    [data-testid="collapsedControl"] button svg,
    [data-testid="collapsedControl"] button svg * {
        color: var(--claro-red) !important;
        fill: var(--claro-red) !important;
        stroke: var(--claro-red) !important;
    }
    [data-testid="collapsedControl"] button:hover {
        background: #FCEAE8 !important;
        border-color: var(--claro-red) !important;
    }
    button[aria-label="Expand sidebar"], button[title="Expand sidebar"] {
        color: #DA291C !important;
        background: #FFFFFF !important;
        border: 1px solid #F0D2CF !important;
        border-radius: 8px !important;
    }
    button[aria-label="Expand sidebar"] *, button[title="Expand sidebar"] * {
        color: #DA291C !important;
        fill: #DA291C !important;
        stroke: #DA291C !important;
    }
    button[aria-label="Expand sidebar"]:hover, button[title="Expand sidebar"]:hover {
        background: #FCEAE8 !important;
        border-color: #DA291C !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { font-size: 0.92rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 0.45rem 0.6rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.12);
    }
    [data-testid="stSidebar"] [role="radiogroup"] *,
    [data-testid="stSidebar"] [role="radiogroup"] p {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div { color: var(--claro-charcoal); }

    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] button {
        border-color: #D2D2D2;
        border-radius: 8px;
    }
    [data-baseweb="input"] > div:focus-within,
    [data-baseweb="select"] > div:focus-within { border-color: var(--claro-red); }

    .stButton > button, .stDownloadButton > button {
        min-height: 2.8rem;
        width: 100%;
        border: 1px solid var(--claro-red);
        border-radius: 8px;
        background: var(--claro-red);
        color: #FFFFFF;
        font-family: "AMX", Arial, sans-serif;
        font-weight: 700;
        transition: background-color 120ms ease, border-color 120ms ease;
    }
    .stButton > button *, .stDownloadButton > button * { color: #FFFFFF !important; }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--claro-red-dark);
        background: var(--claro-red-dark);
        color: #FFFFFF;
    }
    .stButton > button:focus, .stDownloadButton > button:focus {
        border-color: var(--claro-red-dark);
        box-shadow: 0 0 0 0.2rem rgba(218, 41, 28, 0.18);
        color: #FFFFFF;
    }

    .sci-card {
        min-height: 148px;
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 4px solid var(--claro-red);
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .sci-card h4 { margin: 0 0 0.45rem 0; font-size: 1rem; }
    .sci-card h3 { margin: 0 0 0.4rem 0; color: var(--claro-red) !important; font-size: 1.55rem; }
    .sci-card p { margin: 0; color: #686868; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
    [data-testid="stAlert"] {
        border: 1px solid #E8B6B2;
        border-left: 4px solid var(--claro-red);
        border-radius: 8px;
        background: #FFF8F7;
        color: var(--claro-charcoal);
    }
    #MainMenu, footer { visibility: hidden; }

    @media (max-width: 640px) {
        .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
        .brand-subtitle { font-size: 0.9rem; }
        .sci-card { min-height: 132px; padding: 0.9rem; }
    }
</style>
""", unsafe_allow_html=True)

if 'historico' not in st.session_state:
    st.session_state.historico = []

logo_col, brand_col = st.columns([0.20, 0.80], vertical_alignment="center")
with logo_col:
    st.image("assets/claro-esfera.png", width=112)
with brand_col:
    st.markdown('<h1 class="brand-title">Aura Telecom Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Inteligência comercial, recomendação de planos e retenção.</p>', unsafe_allow_html=True)
st.markdown('<hr style="margin: 1.25rem 0 1.75rem 0;">', unsafe_allow_html=True)

df_catalogo = carregar_catalogo_do_supabase()

st.sidebar.markdown("### 🧭 NAVEGAÇÃO")
menu = st.sidebar.radio("Módulo:", ["⚡ Recomendador, Churn (ML) & PDF", "📈 Dashboard & Upsell Regional", "📂 Histórico"])

if menu == "⚡ Recomendador, Churn (ML) & PDF":
    st.sidebar.markdown("### 🎛️ PARÂMETROS")
    nome = st.sidebar.text_input("Cliente", "Felipe Ribeiro (Feris)")
    dados_gb = st.sidebar.slider("Dados (GB/mês)", 1.0, 100.0, 25.0)
    minutos = st.sidebar.slider("Voz / Minutos", 50, 1500, 500)
    streaming = st.sidebar.slider("Streaming (h/mês)", 0.0, 50.0, 10.0)
    mobilidade = st.sidebar.selectbox("Mobilidade", [0, 1, 2, 3], format_func=lambda x: ["Local", "Regional", "Nacional", "Global"][x])
    meses = st.sidebar.number_input("Tempo de Relacionamento (Meses)", 1, 240, 24)
    fatura = st.sidebar.number_input("Fatura Atual (R$)", 0.0, 999.0, 119.90)
    reclamacoes = st.sidebar.slider("Reclamações", 0, 5, 1)

    if st.button("⚡ PROCESSAR ANÁLISE COM MACHINE LEARNING"):
        resultado = recomendar_planos(df_catalogo, dados_gb, minutos, streaming, mobilidade, meses)
        campeao = resultado.iloc[0]
        churn_score = prever_churn_cliente(dados_gb, meses, fatura, reclamacoes)
        
        st.markdown("### 🏆 PÓDIO OTIMIZADO")
        cols = st.columns(len(resultado))
        for idx, (_, p) in enumerate(resultado.iterrows()):
            with cols[idx]:
                st.markdown(f'<div class="sci-card"><h4>{p["nome"]}</h4><h3>R$ {p["preco_mensal_brl"]:.2f}</h3><p>Franquia: {p["franquia_gb"]} GB</p></div>', unsafe_allow_html=True)
                
        st.markdown(f"### 🤖 CHURN SCORE (Random Forest): {churn_score*100:.1f}%")
        
        economia = fatura - campeao['preco_mensal_brl']
        caminho_pdf = f"proposta_{nome.replace(' ', '_').lower()}.pdf"
        gerar_pdf_proposta(caminho_pdf, {'nome': nome, 'meses': meses, 'consumo_gb': dados_gb, 'fatura_atual': fatura, 'churn_score': churn_score}, campeao.to_dict(), economia)
        
        with open(caminho_pdf, "rb") as f:
            st.download_button("📥 BAIXAR PROPOSTA EM PDF", data=f.read(), file_name=caminho_pdf, mime="application/pdf")

elif menu == "📈 Dashboard & Upsell Regional":
    st.markdown("## 📈 DASHBOARD EXECUTIVO & UPSELL REGIONAL (H3)")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#FFFFFF'); ax.set_facecolor('#FFFFFF')
        ax.barh(['Zona Norte', 'Zona Sul', 'Zona Leste', 'Zona Oeste'], [85, 45, 92, 60], color='#DA291C')
        ax.tick_params(colors='#2D2926')
        ax.grid(axis='x', color='#E2E2E2', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#E2E2E2')
        st.pyplot(fig)
    with col2:
        st.dataframe(df_catalogo, use_container_width=True)

elif menu == "📂 Histórico":
    st.markdown("## 📂 LOG DE AUDITORIA")
    st.info("Nenhum atendimento processado nesta sessão.")
