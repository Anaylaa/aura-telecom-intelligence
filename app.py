
"""
=============================================================================
ARQUIVO: app.py
Aura Telecom Intelligence — Interface visual aprimorada
Lógica de negócio, ML, Supabase e PDF preservados.
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

st.set_page_config(
    page_title="Claro | Aura Telecom Intelligence",
    page_icon="assets/claro-esfera.png",
    layout="wide"
)

# ==========================================================
# IDENTIDADE VISUAL
# ==========================================================

st.markdown("""
<style>
    :root {
        --red: #DA291C;
        --red-dark: #AF272F;
        --red-light: #FFF1EF;
        --charcoal: #2D2926;
        --gray: #777777;
        --canvas: #F5F6F8;
        --white: #FFFFFF;
        --border: #E8E8E8;
    }

    .stApp {
        background: var(--canvas);
        color: var(--charcoal);
        font-family: "AMX", Arial, sans-serif;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3, h4 {
        color: var(--charcoal) !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }

    p, label {
        color: var(--charcoal);
    }

    hr {
        border-color: var(--border);
    }

    /* SIDEBAR */

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: var(--charcoal);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #45403C;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 0.9rem;
        letter-spacing: 0.04em;
        margin-top: 1rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.65rem;
        border-radius: 8px;
        transition: 0.2s;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #403A36;
    }

    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background: #393431;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] input {
        color: white !important;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #F1F1F1 !important;
        font-size: 0.88rem;
    }

    /* CABEÇALHO */

    .brand-title {
        font-size: clamp(1.7rem, 3vw, 2.5rem);
        font-weight: 800;
        letter-spacing: -0.05em;
        margin: 0;
        color: var(--charcoal);
    }

    .brand-subtitle {
        color: var(--gray);
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    .section-label {
        color: var(--red);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    /* CARDS */

    .sci-card {
        background: var(--white);
        border: 1px solid var(--border);
        border-left: 4px solid var(--red);
        border-radius: 12px;
        padding: 1.2rem;
        min-height: 145px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.025);
    }

    .sci-card h4 {
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }

    .sci-card h3 {
        color: var(--red) !important;
        font-size: 1.65rem;
        margin: 0.3rem 0;
    }

    .sci-card p {
        color: var(--gray);
        font-size: 0.85rem;
        margin: 0;
    }

    /* BOTÕES */

    .stButton > button,
    .stDownloadButton > button {
        background: var(--red);
        color: white !important;
        border: 1px solid var(--red);
        border-radius: 8px;
        min-height: 2.8rem;
        font-weight: 700;
        transition: 0.2s;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: var(--red-dark);
        border-color: var(--red-dark);
        color: white !important;
    }

    .stButton > button *,
    .stDownloadButton > button * {
        color: white !important;
    }

    /* CAMPOS */

    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div {
        border-radius: 8px;
        border-color: #D9D9D9;
    }

    [data-baseweb="input"] > div:focus-within,
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--red);
    }

    /* TABELAS */

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    /* MÉTRICAS */

    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--gray);
    }

    [data-testid="stAlert"] {
        border-left: 4px solid var(--red);
        border-radius: 8px;
    }

    /* GRÁFICOS */

    [data-testid="stPlotlyChart"],
    [data-testid="stImage"] {
        border-radius: 10px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media(max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .brand-title {
            font-size: 1.5rem;
        }

        .sci-card {
            min-height: 120px;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# CABEÇALHO
# ==========================================================

if 'historico' not in st.session_state:
    st.session_state.historico = []

logo_col, brand_col = st.columns([0.13, 0.87], vertical_alignment="center")

with logo_col:
    st.image("assets/claro-esfera.png", width=95)

with brand_col:
    st.markdown(
        '<h1 class="brand-title">Aura Telecom Intelligence</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="brand-subtitle">'
        'Inteligência comercial • Recomendação de planos • Retenção'
        '</p>',
        unsafe_allow_html=True
    )

st.divider()

# ==========================================================
# DADOS E NAVEGAÇÃO
# ==========================================================

df_catalogo = carregar_catalogo_do_supabase()

st.sidebar.markdown("### 🧭 NAVEGAÇÃO")

menu = st.sidebar.radio(
    "Módulo:",
    [
        "⚡ Recomendador, Churn (ML) & PDF",
        "📈 Dashboard & Upsell Regional",
        "📂 Histórico"
    ]
)


# ==========================================================
# MÓDULO 1 — RECOMENDADOR + MACHINE LEARNING
# ==========================================================

if menu == "⚡ Recomendador, Churn (ML) & PDF":

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ PARÂMETROS DO CLIENTE")

    nome = st.sidebar.text_input(
        "Cliente",
        "Felipe Ribeiro (Feris)"
    )

    dados_gb = st.sidebar.slider(
        "Dados (GB/mês)", 1.0, 100.0, 25.0
    )

    minutos = st.sidebar.slider(
        "Voz / Minutos", 50, 1500, 500
    )

    streaming = st.sidebar.slider(
        "Streaming (h/mês)", 0.0, 50.0, 10.0
    )

    mobilidade = st.sidebar.selectbox(
        "Mobilidade",
        [0, 1, 2, 3],
        format_func=lambda x: [
            "Local", "Regional", "Nacional", "Global"
        ][x]
    )

    meses = st.sidebar.number_input(
        "Tempo de Relacionamento (Meses)",
        1, 240, 24
    )

    fatura = st.sidebar.number_input(
        "Fatura Atual (R$)",
        0.0, 999.0, 119.90
    )

    reclamacoes = st.sidebar.slider(
        "Reclamações", 0, 5, 1
    )

    # CABEÇALHO DO MÓDULO

    st.markdown(
        '<p class="section-label">INTELIGÊNCIA COMERCIAL</p>',
        unsafe_allow_html=True
    )

    st.markdown("## ⚡ Análise e recomendação inteligente")

    st.markdown(
        "Configure os parâmetros do cliente na lateral e processe "
        "a análise para identificar oportunidades comerciais, "
        "recomendar planos e avaliar o risco de cancelamento."
    )

    # RESUMO DOS PARÂMETROS

    st.markdown("### Perfil de consumo")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Internet", f"{dados_gb:.0f} GB")
    c2.metric("Voz", f"{minutos} min")
    c3.metric("Streaming", f"{streaming:.0f} h")
    c4.metric("Relacionamento", f"{meses} meses")

    st.markdown("")

    # PROCESSAMENTO

    if st.button(
        "⚡ PROCESSAR ANÁLISE COM MACHINE LEARNING",
        use_container_width=True
    ):

        resultado = recomendar_planos(
            df_catalogo,
            dados_gb,
            minutos,
            streaming,
            mobilidade,
            meses
        )

        campeao = resultado.iloc[0]

        churn_score = prever_churn_cliente(
            dados_gb,
            meses,
            fatura,
            reclamacoes
        )

        st.divider()

        # RESULTADOS

        st.markdown(
            '<p class="section-label">RESULTADOS DA ANÁLISE</p>',
            unsafe_allow_html=True
        )

        st.markdown("## 🏆 Planos recomendados")

        cols = st.columns(len(resultado))

        for idx, (_, p) in enumerate(resultado.iterrows()):

            with cols[idx]:

                st.markdown(
                    f"""
                    <div class="sci-card">
                        <h4>{p["nome"]}</h4>
                        <h3>R$ {p["preco_mensal_brl"]:.2f}</h3>
                        <p>Franquia de {p["franquia_gb"]} GB</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # CHURN

        st.markdown("## 🤖 Análise de retenção")

        st.metric(
            "CHURN SCORE — RANDOM FOREST",
            f"{churn_score * 100:.1f}%"
        )

        st.progress(
            min(max(float(churn_score), 0.0), 1.0)
        )

        st.caption(
            "Indicador de risco de cancelamento estimado pelo modelo."
        )

        # PROPOSTA COMERCIAL

        economia = fatura - campeao['preco_mensal_brl']

        st.markdown("## 💼 Oportunidade comercial")

        m1, m2 = st.columns(2)

        m1.metric(
            "Plano recomendado",
            campeao["nome"]
        )

        m2.metric(
            "Diferença para a fatura atual",
            f"R$ {economia:.2f}"
        )

        caminho_pdf = (
            f"proposta_{nome.replace(' ', '_').lower()}.pdf"
        )

        gerar_pdf_proposta(
            caminho_pdf,
            {
                'nome': nome,
                'meses': meses,
                'consumo_gb': dados_gb,
                'fatura_atual': fatura,
                'churn_score': churn_score
            },
            campeao.to_dict(),
            economia
        )

        with open(caminho_pdf, "rb") as f:

            st.download_button(
                "📥 BAIXAR PROPOSTA COMERCIAL EM PDF",
                data=f.read(),
                file_name=caminho_pdf,
                mime="application/pdf",
                use_container_width=True
            )


# ==========================================================
# MÓDULO 2 — DASHBOARD EXECUTIVO
# ==========================================================

elif menu == "📈 Dashboard & Upsell Regional":

    st.markdown(
        '<p class="section-label">BUSINESS INTELLIGENCE</p>',
        unsafe_allow_html=True
    )

    st.markdown("## 📈 Dashboard Executivo")

    st.markdown(
        "Análise da distribuição regional de oportunidades "
        "e catálogo de planos disponíveis."
    )

    st.divider()

    st.markdown("### 📍 Oportunidades de Upsell por Região")

    col1, col2 = st.columns([1.05, 1])

    with col1:

        with st.container(border=True):

            st.markdown("#### Distribuição regional")

            fig, ax = plt.subplots(figsize=(6, 4))

            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#FFFFFF')

            ax.barh(
                ['Zona Norte', 'Zona Sul', 'Zona Leste', 'Zona Oeste'],
                [85, 45, 92, 60],
                color='#DA291C',
                height=0.55
            )

            ax.tick_params(colors='#2D2926')

            ax.grid(
                axis='x',
                color='#E2E2E2',
                linewidth=0.8
            )

            ax.set_axisbelow(True)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#E2E2E2')

            ax.set_xlabel(
                "Índice de oportunidade",
                color='#777777'
            )

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)

    with col2:

        with st.container(border=True):

            st.markdown("#### Catálogo de planos")

            st.dataframe(
                df_catalogo,
                use_container_width=True,
                hide_index=True
            )


# ==========================================================
# MÓDULO 3 — HISTÓRICO
# ==========================================================

elif menu == "📂 Histórico":

    st.markdown(
        '<p class="section-label">AUDITORIA E ACOMPANHAMENTO</p>',
        unsafe_allow_html=True
    )

    st.markdown("## 📂 Histórico de atendimentos")

    st.markdown(
        "Área destinada ao acompanhamento das análises "
        "realizadas pelo sistema."
    )

    st.divider()

    if not st.session_state.historico:

        with st.container(border=True):

            st.markdown("### 📋 Nenhum atendimento registrado")

            st.caption(
                "As análises realizadas poderão ser apresentadas "
                "nesta área quando houver registros disponíveis."
            )

    else:

        st.dataframe(
            pd.DataFrame(st.session_state.historico),
            use_container_width=True
        )
