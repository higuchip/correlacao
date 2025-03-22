import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, pearsonr, spearmanr

st.title("Análise de Normalidade e Correlação")
st.write("Este app processa arquivos CSV com duas colunas, onde as colunas são separadas por ';' e os números usam vírgula como separador decimal.")

# 1. Exemplo de Arquivo CSV para Visualização e Download
st.header("Exemplo de Arquivo CSV")
st.write("O exemplo a seguir demonstra o formato esperado para o arquivo CSV:")

# Criando um dataframe de exemplo
df_exemplo = pd.DataFrame({
    "Variavel1": [1.2, 2.3, 3.4, 4.5, 5.6],
    "Variavel2": [2.1, 3.2, 4.3, 5.4, 6.5]
})
st.dataframe(df_exemplo)

# Gerando uma string CSV com separador ';' e decimal ','
csv_exemplo = df_exemplo.to_csv(sep=";", decimal=",", index=False)
st.download_button(
    label="Download do arquivo exemplo",
    data=csv_exemplo,
    file_name="exemplo.csv",
    mime="text/csv"
)

# 2. Upload do Arquivo CSV
st.header("Upload do Arquivo CSV")
st.write("Faça o upload do seu arquivo CSV com o mesmo formato do exemplo.")

uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leitura do CSV com separador ';' e decimais com ','
    df = pd.read_csv(uploaded_file, sep=";", decimal=",")
    st.subheader("Visualização dos dados")
    st.dataframe(df.head())

    # Verifica se o arquivo tem pelo menos duas colunas
    if df.shape[1] < 2:
        st.error("O arquivo precisa ter pelo menos duas colunas.")
    else:
        # Permite que o usuário escolha as colunas a serem analisadas
        colunas = df.columns.tolist()
        var1 = st.selectbox("Selecione a variável 1", colunas)
        var2 = st.selectbox("Selecione a variável 2", colunas, index=1 if len(colunas) > 1 else 0)

        # Remove valores nulos
        dados1 = df[var1].dropna()
        dados2 = df[var2].dropna()

        # 3. Teste de Normalidade
        st.header("Teste de Normalidade (Shapiro-Wilk)")
        stat1, p1 = shapiro(dados1)
        stat2, p2 = shapiro(dados2)
        st.write(f"**{var1}**: Estatística = {stat1:.4f}, p-valor = {p1:.4f}")
        st.write(f"**{var2}**: Estatística = {stat2:.4f}, p-valor = {p2:.4f}")

        # 4. Análise de Correlação
        st.header("Análise de Correlação")
        metodo = st.radio("Escolha o método de correlação", ("Pearson", "Spearman"))
        if metodo == "Pearson":
            coef, p_corr = pearsonr(dados1, dados2)
        else:
            coef, p_corr = spearmanr(dados1, dados2)
        st.write(f"Método: **{metodo}**")
        st.write(f"Coeficiente de Correlação = {coef:.4f}, p-valor = {p_corr:.4f}")

        # 5. Visualização Gráfica
        st.header("Visualização Gráfica")
        fig, ax = plt.subplots()
        sns.scatterplot(x=dados1, y=dados2, ax=ax)
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title("Relação entre as Variáveis")
        # Se o método for Pearson, adiciona uma linha de regressão
        if metodo == "Pearson":
            m, b = np.polyfit(dados1, dados2, 1)
            ax.plot(dados1, m * dados1 + b, color='red', label="Linha de Regressão")
            ax.legend()
        st.pyplot(fig)
