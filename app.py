import streamlit as st

st.title('portfolio')

# Nome
st.header('nome')
st.write('pietro belafonte pansani')

#Sobre 
st.header('sobre mim')
st.write(""" gosto de ler.
         """)

# curiosidade
st.header('Curiosidades')
st.write(
   'gosto esportes, '
   'gosto academia, '
   'joga video game. '
)

#foto
st.header('foto')
st.image('img.jpg')
width = 200
