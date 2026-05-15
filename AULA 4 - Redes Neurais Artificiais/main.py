import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ==========================================
# Configuração da Página e Cabeçalho
# ==========================================
st.set_page_config(page_title="Regressão Linear Didática", layout="centered")
st.title("Análise de Regressão Linear Simples")
st.markdown(
    """
    Este módulo demonstra a aplicação prática da biblioteca `scikit-learn` 
    para modelar e prever a nota de um estudante com base nas horas dedicadas ao estudo.
    """
)

# ==========================================
# 1. Base de Dados Original
# ==========================================
dados_estudo = pd.DataFrame(
    {"notas": [1, 2, 4, 6, 8, 10], "horas": [2, 4, 5, 7, 9, 10]}
)

st.subheader("1. Dados Históricos de Treinamento")
col1, col2 = st.columns([1, 2])

with col1:
    st.dataframe(dados_estudo, use_container_width=True)

with col2:
    st.info(
        f"**Total de Amostras:** {dados_estudo.shape[0]}\n\n"
        "**X (Variável Independente):** Horas de Estudo\n\n"
        "**y (Variável Dependente):** Notas obtidas"
    )

# ==========================================
# 2. Treinamento do Modelo (Scikit-Learn)
# ==========================================
X = dados_estudo[["horas"]]
y = dados_estudo["notas"]

modelo = LinearRegression()
modelo.fit(X, y)

# ==========================================
# 3. Interface de Predição Dinâmica
# ==========================================
st.subheader("2. Simulação e Inferência em Tempo Real")

# Sidebar ou Slider para entrada do usuário
horas_input = st.slider(
    "Selecione a quantidade de horas de estudo:",
    min_value=0.0,
    max_value=12.0,
    value=6.0,
    step=0.5,
)

# Predição baseada na entrada do Slider
novo_dado = pd.DataFrame({"horas": [horas_input]})
nota_predita = modelo.predict(novo_dado)[0]

# Garantir que a nota predita permaneça dentro dos limites acadêmicos lógicos (0 a 10)
nota_final = max(0.0, min(10.0, nota_predita))

st.metric(
    label=f"Nota Prevista para {horas_input} horas de estudo",
    value=f"{nota_final:.2f} / 10.0",
)

# ==========================================
# 4. Representação Gráfica com Altair
# ==========================================
st.subheader("3. Visualização do Modelo e da Reta de Regressão")

# Gerando pontos da reta para o gráfico (do mínimo ao máximo de horas)
horas_reta = np.linspace(0, 12, 100).reshape(-1, 1)
notas_reta = modelo.predict(horas_reta)
df_reta = pd.DataFrame({"horas": horas_reta.flatten(), "notas": notas_reta})

# Gráfico de Dispersão (Dados Reais)
grafico_pontos = (
    alt.Chart(dados_estudo)
    .mark_circle(size=80, color="blue", tooltip=True)
    .encode(
        x=alt.X("horas:Q", title="Horas de Estudo"),
        y=alt.Y("notas:Q", title="Nota"),
    )
)

# Gráfico de Linha (Reta de Regressão)
grafico_linha = (
    alt.Chart(df_reta)
    .mark_line(color="red", strokeWidth=2)
    .encode(x="horas:Q", y="notas:Q")
)

# Ponto Interativo da Predição Atual
ponto_usuario = (
    alt.Chart(pd.DataFrame({"horas": [horas_input], "notas": [nota_final]}))
    .mark_circle(size=150, color="orange", tooltip=True)
    .encode(x="horas:Q", y="notas:Q")
)

# Combinando as camadas dos gráficos
grafico_final = alt.layer(
    grafico_pontos, grafico_linha, ponto_usuario
).properties(width=600, height=400)

st.altair_chart(grafico_final, use_container_width=True)

# ==========================================
# 5. Métricas Matemáticas do Modelo
# ==========================================
st.markdown("---")
st.subheader("4. Parâmetros Matemáticos Calculados")
st.latex(rf"y = {modelo.coef_[0]:.4f} \cdot X + ({modelo.intercept_:.4f})")

st.text(f"Coeficiente Angular (Slope): {modelo.coef_[0]:.4f}")
st.text(f"Intercepto (Bias): {modelo.intercept_:.4f}")