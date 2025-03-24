import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, shapiro
import locale
import statsmodels.api as sm

# Configurar locale para usar vírgula como decimal
try:
    locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_NUMERIC, 'Portuguese_Brazil.1252')
    except:
        pass

def main():
    st.title("Análise de Correlação entre Variáveis")
    
    # Upload do arquivo CSV
    st.header("1. Upload de Dados")
    uploaded_file = st.file_uploader("Faça upload de um arquivo CSV (separado por ; com decimal ,)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Carregando os dados com separador ; e decimal ,
            df = pd.read_csv(uploaded_file, sep=';', decimal=',')
            
            # Exibindo preview dos dados e informações sobre os tipos
            st.header("2. Visualização dos Dados")
            st.write("Visualização das primeiras linhas dos dados:")
            st.dataframe(df.head())
            
            st.write(f"Colunas encontradas no arquivo: {', '.join(df.columns)}")
            st.write(f"Total de colunas: {len(df.columns)}")
            
            # Criar uma cópia do DataFrame para manipulação
            df_numeric = df.copy()
            
            # Converter todas as colunas para numérico quando possível
            for col in df_numeric.columns:
                try:
                    # Tentar converter diretamente
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Não foi possível converter a coluna '{col}' para numérico: {e}")
            
            # Opção para selecionar quais colunas usar na análise
            st.header("3. Seleção de Colunas para Análise")
            
            # Mostrar todas as colunas como opções, mesmo as não numéricas
            all_columns = df.columns.tolist()
            
            st.write("Selecione quais colunas deseja incluir na análise:")
            selected_columns = []
            
            # Criar checkboxes para cada coluna
            cols_per_row = 2
            rows = [all_columns[i:i + cols_per_row] for i in range(0, len(all_columns), cols_per_row)]
            
            for row in rows:
                cols = st.columns(cols_per_row)
                for i, col_name in enumerate(row):
                    if i < len(cols) and col_name in all_columns:
                        if cols[i].checkbox(col_name, value=True):
                            selected_columns.append(col_name)
            
            # Converter apenas as colunas selecionadas
            if selected_columns:
                analysis_df = df[selected_columns].copy()
                
                for col in analysis_df.columns:
                    analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
                
                # Exibir estatísticas básicas das colunas selecionadas
                st.write("Estatísticas básicas das colunas selecionadas:")
                st.dataframe(analysis_df.describe())
                
                # Filtrar para manter apenas colunas que têm valores numéricos
                numeric_columns = analysis_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                
                st.write(f"Colunas numéricas disponíveis para análise: {', '.join(numeric_columns)}")
                
                if len(numeric_columns) < 2:
                    st.error("É necessário pelo menos duas colunas numéricas para análise de correlação.")
                    st.write("Verifique se as colunas selecionadas contêm valores numéricos válidos.")
                else:
                    # Testes de normalidade - utilizando apenas Shapiro-Wilk sem Q-Q plot
                    st.header("4. Teste de Normalidade (Shapiro-Wilk)")
                    
                    # Opção para selecionar quais variáveis testar para normalidade
                    st.subheader("Selecione as variáveis para testar normalidade:")
                    
                    selected_norm_test_columns = st.multiselect(
                        "Escolha as variáveis para o teste de normalidade:", 
                        numeric_columns,
                        default=numeric_columns[0] if numeric_columns else None
                    )
                    
                    if st.button("Executar Teste de Shapiro-Wilk"):
                        if not selected_norm_test_columns:
                            st.warning("Selecione pelo menos uma variável para o teste de normalidade.")
                        else:
                            for col in selected_norm_test_columns:
                                st.subheader(f"Análise de Normalidade: {col}")
                                
                                # Remover valores ausentes para o teste
                                data = analysis_df[col].dropna()
                                
                                if len(data) < 3:
                                    st.warning(f"Dados insuficientes para teste de normalidade em '{col}'")
                                    continue
                                    
                                # Teste de Shapiro-Wilk
                                shapiro_stat, shapiro_p = shapiro(data)
                                
                                st.write("**Histograma e Curva Normal**")
                                fig, ax = plt.subplots(figsize=(8, 5))
                                sns.histplot(data, kde=True, ax=ax)
                                plt.title(f'Distribuição de {col}')
                                plt.xlabel(col)
                                plt.ylabel('Frequência')
                                st.pyplot(fig)
                                
                                st.write("**Resultados do Teste de Shapiro-Wilk**")
                                
                                # Resultados e interpretação em formato mais intuitivo
                                st.write(f"**Estatística do teste:** {shapiro_stat:.4f}")
                                st.write(f"**Valor-p:** {shapiro_p:.4f}")
                                
                                if shapiro_p < 0.05:
                                    st.error("A distribuição **NÃO é normal**")
                                    st.write("O valor-p é menor que 0.05, o que indica que os dados NÃO seguem uma distribuição normal.")
                                    st.write("**Recomendação:** Utilize a correlação de Spearman, que é adequada para dados não-normais.")
                                else:
                                    st.success("A distribuição pode ser considerada normal")
                                    st.write("O valor-p é maior ou igual a 0.05, o que indica que não há evidência contra a normalidade dos dados.")
                                    st.write("**Recomendação:** Você pode utilizar a correlação de Pearson, que é mais adequada para dados normais.")
                    
                    # Seleção das variáveis para correlação
                    st.header("5. Seleção de Variáveis para Correlação")
                    
                    col1 = st.selectbox("Selecione a primeira variável:", numeric_columns)
                    col2 = st.selectbox("Selecione a segunda variável:", 
                                        [col for col in numeric_columns if col != col1], 
                                        index=0)
                    
                    # Seleção do método de correlação
                    correlation_method = st.radio(
                        "Selecione o método de correlação:",
                        ["Pearson (para dados normais)", 
                         "Spearman (para dados não normais)"]
                    )
                    
                    # Análise de correlação
                    st.header("6. Análise de Correlação")
                    
                    # Removendo valores NaN para o cálculo da correlação
                    valid_data = analysis_df[[col1, col2]].dropna()
                    
                    if len(valid_data) < 2:
                        st.error("Não há dados suficientes para calcular a correlação após remover valores ausentes.")
                    else:
                        # Matriz de correlação
                        st.subheader("Matriz de Correlação")
                        
                        if "Pearson" in correlation_method:
                            corr_matrix = valid_data.corr(method='pearson')
                            corr_coef, p_value = pearsonr(valid_data[col1], valid_data[col2])
                            method_name = "Pearson"
                        else:
                            corr_matrix = valid_data.corr(method='spearman')
                            corr_coef, p_value = spearmanr(valid_data[col1], valid_data[col2])
                            method_name = "Spearman"
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                        plt.title(f'Matriz de Correlação de {method_name}')
                        st.pyplot(fig)
                        
                        # Gráfico de dispersão
                        st.subheader("Gráfico de Dispersão")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.scatterplot(x=valid_data[col1], y=valid_data[col2], ax=ax)
                        plt.xlabel(col1)
                        plt.ylabel(col2)
                        plt.title(f"Relação entre {col1} e {col2}")
                        
                        # Adicionar linha de tendência
                        m, b = np.polyfit(valid_data[col1], valid_data[col2], 1)
                        plt.plot(valid_data[col1], m*valid_data[col1] + b, color='red')
                        st.pyplot(fig)
                        
                        # Resultados da correlação - Interpretação mais intuitiva
                        st.subheader("Resultados da Correlação")
                        
                        # Coeficiente com colorização visual
                        if abs(corr_coef) < 0.3:
                            st.info(f"**Coeficiente de {method_name}:** {corr_coef:.4f} (Correlação fraca)")
                        elif abs(corr_coef) < 0.7:
                            st.warning(f"**Coeficiente de {method_name}:** {corr_coef:.4f} (Correlação moderada)")
                        else:
                            st.success(f"**Coeficiente de {method_name}:** {corr_coef:.4f} (Correlação forte)")
                        
                        # Direção da correlação com explicação clara
                        if corr_coef > 0:
                            st.write("**Direção:** Correlação positiva ↗️")
                            st.write("Isso significa que as variáveis tendem a **aumentar juntas**. Quando uma aumenta, a outra também tende a aumentar.")
                        else:
                            st.write("**Direção:** Correlação negativa ↘️")
                            st.write("Isso significa que as variáveis tendem a se **mover em direções opostas**. Quando uma aumenta, a outra tende a diminuir.")
                        
                        # Significância estatística
                        st.write(f"**Valor P:** {p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.success("**Resultado:** Correlação estatisticamente significativa! ✓")
                            st.write("O valor-p é menor que 0.05, o que indica que a correlação encontrada é estatisticamente significativa e provavelmente não ocorreu por acaso.")
                            st.write(f"Podemos afirmar com pelo menos 95% de confiança que existe uma relação entre {col1} e {col2}.")
                        else:
                            st.error("**Resultado:** Correlação não é estatisticamente significativa! ✗")
                            st.write("O valor-p é maior ou igual a 0.05, o que indica que não temos evidência estatística suficiente de que existe uma correlação real.")
                            st.write(f"A correlação observada entre {col1} e {col2} pode ter ocorrido por acaso.")
                        
                        # Interpretação final em linguagem simples
                        st.subheader("Interpretação Final")
                        
                        if p_value < 0.05:
                            if abs(corr_coef) < 0.3:
                                conclusao = f"Existe uma correlação fraca e {('positiva' if corr_coef > 0 else 'negativa')} entre {col1} e {col2}."
                            elif abs(corr_coef) < 0.7:
                                conclusao = f"Existe uma correlação moderada e {('positiva' if corr_coef > 0 else 'negativa')} entre {col1} e {col2}."
                            else:
                                conclusao = f"Existe uma correlação forte e {('positiva' if corr_coef > 0 else 'negativa')} entre {col1} e {col2}."
                        else:
                            conclusao = f"Não foi encontrada uma correlação estatisticamente significativa entre {col1} e {col2}."
                        
                        st.write(conclusao)
            else:
                st.error("Selecione pelo menos duas colunas para análise.")
                
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            st.error("Verifique se o arquivo está no formato correto (separado por ; com decimal ,)")

if __name__ == "__main__":
    main()
