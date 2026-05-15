import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# CONFIGURAÇÃO DO AMBIENTE E INTERFACE
# ==============================================================================
st.set_page_config(
    page_title="Detector de Aprovação Ninja", 
    page_icon="🥷", 
    layout="centered"
)

st.title("🥷 Detector de Aprovação Ninja")
st.markdown("""
### Análise Estatística Pragmática via Regressão Logística
Este sistema demonstra a aplicação de um classificador linear supervisionado para 
mapear a relação de dependência entre o volume de faltas (variável independente) 
e o status de aprovação de um discente (variável dependente binária).
""")

# ==============================================================================
# 1. ESPECIFICAÇÃO DO CONJUNTO DE DADOS (DATASET)
# ==============================================================================
st.header("1. Dataset de Treinamento")

# Base de dados fornecida no escopo do problema
alunos = pd.DataFrame({
    'faltas': [0, 1, 2, 5, 7, 10],
    'resultado': [1, 1, 1, 0, 0, 0]  # 1: Aprovado, 0: Reprovado
})

# Formatação visual para fins didáticos
view_df = alunos.copy()
view_df['Status Acadêmico'] = view_df['resultado'].map({1: 'Aprovado', 0: 'Reprovado'})
st.dataframe(
    view_df[['faltas', 'Status Acadêmico']].rename(columns={'faltas': 'Número de Faltas'}),
    use_container_width=True
)

# ==============================================================================
# 2. ENGENHARIA DE RECURSOS E TREINAMENTO DO MODELO
# ==============================================================================
# Isolação das features (X) e do vetor de labels (y)
X = alunos[['faltas']]  # Formato matricial [n_samples, n_features]
y = alunos['resultado'] # Vetor unidimensional

# Instanciação e ajuste do modelo de Regressão Logística
modelo = LogisticRegression()
modelo.fit(X, y)

# ==============================================================================
# 3. REPRESENTAÇÃO GRÁFICA INTERNA (NATIVA DO STREAMLIT)
# ==============================================================================
st.header("2. Fronteira de Decisão e Curva de Probabilidade Sigmoide")

# Geração de um espaço vetorial contínuo para plotagem da curva de inferência
X_continuo = np.linspace(0, 12, 100).reshape(-1, 1)

# Cálculo probabilístico da classe positiva (Aprovado - P(Y=1|X))
probabilidades = modelo.predict_proba(X_continuo)[:, 1]

# Estruturação dos dados para renderização gráfica nativa (Abstração do Matplotlib)
chart_data = pd.DataFrame({
    'Faltas': X_continuo.flatten(),
    'Probabilidade de Aprovação': probabilidades
}).set_index('Faltas')

# Renderização do gráfico de linha do Streamlit
st.line_chart(chart_data)
st.caption(
    "Figura 1: Curva logística ajustada aos dados empíricos. "
    "A transição de fase denota a perda de probabilidade com o incremento das faltas."
)

# ==============================================================================
# 4. AMBIENTE DE SIMULAÇÃO E PREDIÇÃO INTERATIVA
# ==============================================================================
st.header("3. Inferência de Novos Alunos (Simulação)")

# Widget interativo para entrada de dados
input_faltas = st.slider(
    "Selecione o número de faltas avaliado:", 
    min_value=0, max_value=12, value=3, step=1
)

# Execução da predição pontual e extração do valor probabilístico
predicao = modelo.predict([[input_faltas]])[0]
prob_sucesso = modelo.predict_proba([[input_faltas]])[0][1]

# Apresentação didática do resultado da classificação
if predicao == 1:
    st.success(
        f"**Classificação do Modelo: APROVADO** \n"
        f"Confiança Estatística: {prob_sucesso * 100:.2f}% de chance de aprovação."
    )
else:
    st.error(
        f"**Classificação do Modelo: REPROVADO** \n"
        f"Confiança Estatística: {(1 - prob_sucesso) * 100:.2f}% de chance de reprovação."
    )

# ==============================================================================
# 5. ANÁLISE MATEMÁTICA E FUNDAMENTAÇÃO ACADÊMICA
# ==============================================================================
st.header("4. Fundamentação Matemática do Modelo")

# Recuperação dos parâmetros ótimos encontrados pelo algoritmo de otimização
beta_0 = modelo.intercept_[0]
beta_1 = modelo.coef_[0][0]

# Cálculo analítico do ponto de inflexão (limiar de decisão onde P(Y=1) = 0.5)
ponto_inflexao = -beta_0 / beta_1

st.markdown(f"""
O classificador logístico estima a probabilidade por meio da função sigmoide (logística):

$$ P(Y=1|X) = \\frac{{1}}{{1 + e^{{-(\\beta_0 + \\beta_1 X)}}}} $$

Parâmetros calculados por Máxima Verossimilhança:
* **Intercepto ($\beta_0$):** `{beta_0:.4f}` (Sensibilidade base do modelo)
* **Coeficiente Angular ($\beta_1$):** `{beta_1:.4f}` (Penalidade por falta adicionada)

**Fronteira Analítica de Decisão:** O ponto de equilíbrio matemático em que a probabilidade é exatamente de 50% ocorre em **{ponto_inflexao:.2f}** faltas. 
Portanto:
* $\\le {int(ponto_inflexao)}$ faltas $\\rightarrow$ Classificado como **Aprovado**.
* $\\ge {int(ponto_inflexao) + 1}$ faltas $\\rightarrow$ Classificado como **Reprovado**.
""")