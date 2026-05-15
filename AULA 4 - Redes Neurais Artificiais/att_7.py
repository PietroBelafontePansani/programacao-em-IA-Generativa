import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuração da página para fins de apresentação e usabilidade
st.set_page_config(page_title="Preditor de Nota de Filme", layout="centered")

# Título e Introdução Conceitual (Tom Acadêmico)
st.title("🔬 Regressão Linear Simples com Scikit-Learn")
st.markdown("""
Este aplicativo demonstra de forma didática a aplicação prática de um modelo de **Regressão Linear Simples** utilizando a biblioteca `scikit-learn`. O objetivo é modelar a relação matemática entre a variável independente 
(*Duração*) e a variável dependente (*Nota*), validando o processo de inferência estatística.
""")

# 1. Definição do Dataset Experimental
st.header("1. Dataset de Treinamento")

filmes = pd.DataFrame({
    'duracao': [80, 90, 100, 110, 120],
    'nota': [4, 5, 7, 8, 9]
})

# Exibição tabular dos dados estruturados
st.dataframe(
    filmes.rename(columns={'duracao': 'Duração (minutos)', 'nota': 'Nota (0-10)'}), 
    use_container_width=True
)

# 2. Processamento de Dados e Ajuste do Modelo
# O Scikit-learn exige que a matriz de atributos (X) seja bidimensional. 
# Portanto, extrai-se como DataFrame: filmes[['duracao']], e não como Series.
X = filmes[['duracao']] 
y = filmes['nota']        

# Instanciação do estimador e treinamento via Mínimos Quadrados Ordinários (OLS)
modelo = LinearRegression()
modelo.fit(X, y)

# Extração dos parâmetros analíticos (coeficientes do modelo)
b0 = modelo.intercept_
b1 = modelo.coef_[0]

st.header("2. Parâmetros do Modelo Encontrados")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Intercepto (β₀)", value=f"{b0:.2f}", help="Nota teórica para um filme de 0 minutos")
with col2:
    st.metric(label="Coeficiente Angular (β₁)", value=f"{b1:.4f}", help="Variação na nota por cada minuto adicional")

# Equação matemática renderizada via LaTeX
st.latex(rf"f(X) = {b0:.2f} + {b1:.2f} \cdot X")

# 3. Representação Gráfica Interativa (Sem Matplotlib)
st.header("3. Representação Gráfica e Linha de Tendência")

# Geração de pontos sintéticos para compor a reta contínua de regressão
x_linha = np.linspace(filmes['duracao'].min() - 5, filmes['duracao'].max() + 5, 100)
y_linha = modelo.predict(x_linha.reshape(-1, 1))

# Unificação dos dados para consumo do componente gráfico nativo do Streamlit
df_grafico_real = filmes.copy()
df_grafico_real['Tipo'] = 'Dados Reais'

df_grafico_linha = pd.DataFrame({
    'duracao': x_linha,
    'nota': y_linha,
    'Tipo': 'Linha de Regressão'
})

df_total = pd.concat([df_grafico_real, df_grafico_linha], ignore_index=True)

# Renderização do gráfico usando WebGL interno do Streamlit (scatter_chart)
st.scatter_chart(
    data=df_total,
    x='duracao',
    y='nota',
    color='Tipo',
    use_container_width=True
)

# 4. Interface de Inferência (Predição Dinâmica)
st.header("4. Laboratório de Inferência (Predição)")
st.markdown("Altere o valor abaixo para testar a capacidade de generalização do modelo ajustado:")

# Widget interativo para entrada de dados por parte do usuário
duracao_teste = st.number_input(
    "Insira a duração do filme (em minutos) para inferir a nota:",
    min_value=30, map_index=None, max_value=240, value=105, step=5
)

# Execução da predição com a devida formatação matricial exigida pelo Predictor api [[ valor ]]
nota_predita = modelo.predict([[duracao_teste]])[0]

# Garantindo consistência matemática dentro do intervalo padrão de notas [0, 10]
nota_predita_clip = max(0.0, min(10.0, nota_predita))

# Retorno visual do resultado mapeado
st.success(f"🎥 **Resultado da Inferência:** Para um filme com **{duracao_teste} minutos**, a nota prevista é **{nota_predita_clip:.2f}**")

# Nota técnica final de Engenharia de Dados
st.markdown("""
---
**Nota Didática de Engenharia:** 1. **Formato das Features ($X$)**: O constructo `filmes[['duracao']]` retorna um objeto bidimensional (DataFrame), atendendo à assinatura estrutural padrão exigida pelo método `.fit()` do *Scikit-Learn*.
2. **Abstração Gráfica**: Utilizou-se o método `st.scatter_chart` que abstrai camadas complexas de renderização frontend e elimina dependências pesadas de bibliotecas de imagens estáticas.
""")