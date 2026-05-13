import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression


st.title('ENSINAR A MÁQUINA PREVER O FUTURO')
st.header('opções de campeão...')
st.write('Prever o campeão da copa ⚽')

#dados
dados = pd.DataFrame({
'gols':[12,15,10,18,14,11,16],
'ranking':[1,3,2,1,4,10,2],
'pais':['Brasil','Argentina','frança','Brasil','França','Argentina','Brasil']
})


#alinhamentodo modelo
modelo_copa = DecisionTreeClassifier()
#treinamento
modelo_copa.fit(dados[['gols','ranking']], dados['pais'])

gols_input = st.number_input('Quantos gol o time fez?', 0,30,15)
rank_input = st.number_input('Qual a posição ', 1,100,1)

if st.button('Prever'):
    #previsão
    reultado_copa = modelo_copa.predict([[gols_input, rank_input]])
    st.success(f'o provavel campeão é {reultado_copa}')


#_____________________________________________________________


# NOTAS DE ESTUDOS 


st.header('ANALISE DE NOTAS - PREVENDO')
estudos = pd.DataFrame({
'notas':[1,2,4,6,8,10],
'horas':[2,4,5,7,9,10]
})


st.scatter_chart(estudos, x = 'horas', y= 'notas')
modelo_escola = LinearRegression() 
modelo_escola.fit(estudos[['horas']], estudos['notas'])


h_estudo = st.slider('horas de estudos', 0,12,5)
nota_final = modelo_escola.predict([[h_estudo]])
print(nota_final)


st.metric(f'sua nota seria' ,f'{min(nota_final[0], 10.0):.1f}')

#_______________________________________________________________


import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.header("Previsão de Vendas")

# Dados: [Investimento em Marketing] -> Faturamento
dados_vendas = pd.DataFrame({
    'investimento': [100, 200, 300, 400, 500, 600],
    'faturamento': [1200, 2500, 3200, 4800, 5100, 6300]
})

x = dados_vendas[['investimento']]

y = dados_vendas['faturamento']


modelo = LinearRegression()
modelo.fit(x, y)


novo_investimento = st.number_input(
    "Digite o investimento em marketing:",
    min_value=0.0,
    value=100.0,
    step=50.0
)


if st.button("Prever Faturamento"):

   
    previsao = modelo.predict([[novo_investimento]])

    
    st.success(
        f"Faturamento previsto: R$ {previsao[0]:,.2f}"
    )



# objetivo: previsão de FATURAMENTO 

