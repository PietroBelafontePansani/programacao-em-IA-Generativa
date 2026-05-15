import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# -----------------------------------
# Base de dados
# -----------------------------------
pets = pd.DataFrame({
    'passeios': [1, 2, 3, 4, 5],
    'felicidade': [2, 4, 5, 8, 10]
})

# -----------------------------------
# Variáveis de entrada e saída
# -----------------------------------
X = pets[['passeios']]
y = pets['felicidade']

# -----------------------------------
# Criação do modelo de IA
# -----------------------------------
modelo = LinearRegression()

# Treinamento do modelo
modelo.fit(X, y)

# -----------------------------------
# Interface Streamlit
# -----------------------------------
st.title("🐶 IA Pet Feliz")

st.write("""
Sistema de Inteligência Artificial que prevê
o nível de felicidade do cachorro baseado
na quantidade de passeios realizados.
""")

# Exibir tabela de dados
st.subheader("Base de Dados")
st.dataframe(pets)

# Entrada do usuário
passeios = st.slider(
    "Quantidade de passeios:",
    min_value=1,
    max_value=10,
    value=3
)

# Realizar previsão
previsao = modelo.predict([[passeios]])

# Resultado
st.subheader("Resultado da Previsão")

st.success(
    f"Com {passeios} passeios, "
    f"o nível estimado de felicidade é "
    f"{previsao[0]:.1f}."
)

# -----------------------------------
# Representação gráfica
# -----------------------------------
st.subheader("Gráfico: Passeios x Felicidade")

grafico = pd.DataFrame({
    "Passeios": pets["passeios"],
    "Felicidade": pets["felicidade"]
})

# Gráfico nativo do Streamlit
st.line_chart(
    grafico,
    x="Passeios",
    y="Felicidade"
)

# -----------------------------------
# Informações do modelo
# -----------------------------------
st.subheader("Informações do Modelo")

st.write(f"Coeficiente Angular: {modelo.coef_[0]:.2f}")
st.write(f"Intercepto: {modelo.intercept_:.2f}")

# -----------------------------------
# Teste manual
# -----------------------------------
st.subheader("Teste Manual")

novo_valor = st.number_input(
    "Digite outro valor de passeios:",
    min_value=0
)

if st.button("Prever Felicidade"):
    resultado = modelo.predict([[novo_valor]])

    st.info(
        f"Felicidade prevista: "
        f"{resultado[0]:.2f}"
    )
