import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# -----------------------------
# Base de dados
# -----------------------------
sorvete = pd.DataFrame({
    'temperatura': [18, 20, 24, 27, 30, 35],
    'vendas': [20, 25, 40, 55, 70, 100]
})

# -----------------------------
# Variáveis de treino
# -----------------------------
X = sorvete[['temperatura']]
y = sorvete['vendas']

# -----------------------------
# Criação do modelo
# -----------------------------
modelo = LinearRegression()

# Treinamento
modelo.fit(X, y)

# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("🍦 Previsão de Vendas de Sorvetes")

st.write("""
Sistema de Inteligência Artificial utilizando Regressão Linear
para prever vendas de sorvetes com base na temperatura.
""")

# Exibir dados
st.subheader("Base de Dados")
st.dataframe(sorvete)

# Entrada do usuário
temperatura = st.slider(
    "Informe a temperatura:",
    min_value=15,
    max_value=45,
    value=30
)

# Previsão
previsao = modelo.predict([[temperatura]])

# Resultado
st.subheader("Resultado da Previsão")
st.success(
    f"Com {temperatura}°C, a previsão é de aproximadamente "
    f"{previsao[0]:.0f} sorvetes vendidos."
)

# -----------------------------
# Representação gráfica
# -----------------------------
st.subheader("Gráfico de Temperatura x Vendas")

grafico = pd.DataFrame({
    "Temperatura": sorvete["temperatura"],
    "Vendas": sorvete["vendas"]
})

st.line_chart(
    grafico,
    x="Temperatura",
    y="Vendas"
)

# -----------------------------
# Coeficientes do modelo
# -----------------------------
st.subheader("Informações do Modelo")

st.write(f"Coeficiente Angular: {modelo.coef_[0]:.2f}")
st.write(f"Intercepto: {modelo.intercept_:.2f}")