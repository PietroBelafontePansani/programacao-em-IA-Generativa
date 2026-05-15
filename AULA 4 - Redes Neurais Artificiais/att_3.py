import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# Configuração da UI do Streamlit
st.set_page_config(page_title="Gamer Fatigue Predictor", layout="wide")

# ==========================================
# 1. Camada de Dados (Data Layer)
# ==========================================
def get_training_data():
    """Retorna o dataset inicial para treinamento do modelo."""
    return pd.DataFrame({
        'horas_jogo': [1, 2, 4, 6, 8, 10],
        'cansaco': [1, 2, 3, 5, 8, 10]
    })

# ==========================================
# 2. Camada de Modelagem (ML Layer)
# ==========================================
def train_model(df):
    """Realiza o treinamento do modelo de Regressão Linear."""
    # X (Features) - Matrix 2D obrigatória para sklearn
    X = df[['horas_jogo']]
    # y (Target) - Vector 1D
    y = df['cansaco']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Avaliação do modelo (R²)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    return model, r2

# ==========================================
# 3. Camada de Apresentação (Streamlit)
# ==========================================
def main():
    st.title("🛡️ Detector de Sono Gamer: Análise Preditiva")
    st.markdown("""
    **Objetivo Acadêmico:** Demonstrar o uso de Regressão Linear para inferência de fadiga 
    com base na exposição temporal a sessões de jogos eletrônicos.
    """)

    # Carregamento e Treinamento
    df = get_training_data()
    model, r2 = train_model(df)

    # Layout em Colunas
    col_input, col_metrics = st.columns([1, 2])

    with col_input:
        st.subheader("Configuração da Simulação")
        horas_input = st.number_input(
            "Horas de jogo contínuas:", 
            min_value=0.0, 
            max_value=24.0, 
            value=5.0, 
            step=0.5
        )
        
        # Predição em tempo real
        pred_input = np.array([[horas_input]])
        nivel_cansaco = model.predict(pred_input)[0]
        
        # Lógica de negócio: Normalização acadêmica (0 a 10)
        nivel_cansaco = max(0, min(10, nivel_cansaco))
        
        st.metric(label="Nível de Cansaço Estimado", value=f"{nivel_cansaco:.2f} / 10")
        
        # Alerta de saúde gamer
        if nivel_cansaco > 7.5:
            st.error("⚠️ ALERTA: Nível de fadiga crítico. Risco de sono involuntário.")
        elif nivel_cansaco > 5.0:
            st.warning("⚡ AVISO: Fadiga moderada detectada. Recomenda-se pausa.")
        else:
            st.success("🎮 Status: Operacional. Nível de atenção adequado.")

    with col_metrics:
        st.subheader("Visualização Estatística do Modelo")
        
        # Gerar pontos para a linha de regressão
        x_range = np.linspace(0, 15, 100).reshape(-1, 1)
        y_range = model.predict(x_range)

        # Plotagem com Plotly (Representação Gráfica)
        fig = go.Figure()

        # Dados Reais
        fig.add_trace(go.Scatter(
            x=df['horas_jogo'], y=df['cansaco'],
            mode='markers', name='Dados Reais',
            marker=dict(size=12, color='#636EFA')
        ))

        # Reta de Regressão
        fig.add_trace(go.Scatter(
            x=x_range.flatten(), y=y_range,
            mode='lines', name='Reta de Regressão',
            line=dict(color='#EF553B', dash='dash')
        ))

        # Ponto Atual da Predição
        fig.add_trace(go.Scatter(
            x=[horas_input], y=[nivel_cansaco],
            mode='markers', name='Predição Atual',
            marker=dict(size=15, color='#00CC96', symbol='x')
        ))

        fig.update_layout(
            xaxis_title="Tempo de Exposição (Horas)",
            yaxis_title="Índice de Cansaço",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Rodapé Técnico
    with st.expander("Ver Detalhes Analíticos do Modelo"):
        st.latex(rf"f(x) = {model.coef_[0]:.4f}x + ({model.intercept_:.4f})")
        st.write(f"**Coeficiente de Determinação (R²):** {r2:.4f}")
        st.write(f"**ID de Referência:** jlwm1z")

if __name__ == "__main__":
    main()