# Importar as bibliotecas necessárias

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#%% LEITURA DO ARQUIVO

entrada = pd.read_csv("BankChurners.csv", sep=",")

#%% ANÁLISE EXPLORATÓRIA (DESCRITIVA)

# Lista de colunas para o boxplot
colunas_boxplot = ['Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy']  # Substitua pelas suas colunas

# Criar boxplots para cada coluna
for coluna in colunas_boxplot:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=entrada, x=coluna)
    plt.title(f'Boxplot - {coluna}')
    plt.xlabel(coluna)
    plt.show()

# Lista de colunas categóricas para gráficos de barras
colunas_barra = ['Gender', 'Marital_Status', 'Card_Category']  # Substitua pelas suas colunas

# Criar gráficos de barras para cada coluna categórica
for coluna in colunas_barra:
    plt.figure(figsize=(8, 5))
    entrada[coluna].value_counts().sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title(f'Contagem por Categoria - {coluna}')
    plt.xlabel('Categorias')
    plt.ylabel('Frequência')
    plt.xticks(rotation=45)
    plt.show()

#%% TRATAMENTO INICIAL DOS DADOS

# Remover todas as linhas com valores NA
dado_limpo = entrada.dropna()

# Classificar as cateogorias
category = {"Existing Customer":1,"Attrited Customer":0}

category_cols = ["Attrition_Flag"]

# Aplicar o mapeamento
for col in category_cols:
        dado_limpo[col] = dado_limpo[col].map(category)
        
#%% TRATAMENTO AVANÇADO
        
# Dicionário para armazenar os mapeamentos
mapeamentos = {}

# Converter cada coluna categórica para valores numéricos
dado_limpo['Gender'] = dado_limpo['Gender'].map({'M': 0, 'F': 1})
mapeamentos['Gender'] = {'M': 0, 'F': 1}

dado_limpo['Education_Level'] = dado_limpo['Education_Level'].map({
    'Uneducated': 0, 'High School': 1, 'College': 2, 'Graduate': 3,
    'Post-Graduate': 4, 'Doctorate': 5, 'Unknown': -1
})
mapeamentos['Education_Level'] = {
    'Uneducated': 0, 'High School': 1, 'College': 2, 'Graduate': 3,
    'Post-Graduate': 4, 'Doctorate': 5, 'Unknown': -1
}

dado_limpo['Marital_Status'] = dado_limpo['Marital_Status'].map({
    'Single': 0, 'Married': 1, 'Divorced': 2, 'Unknown': -1
})
mapeamentos['Marital_Status'] = {
    'Single': 0, 'Married': 1, 'Divorced': 2, 'Unknown': -1
}

dado_limpo['Income_Category'] = dado_limpo['Income_Category'].map({
    'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2,
    '$80K - $120K': 3, '$120K +': 4, 'Unknown': -1
})
mapeamentos['Income_Category'] = {
    'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2,
    '$80K - $120K': 3, '$120K +': 4, 'Unknown': -1
}

dado_limpo['Card_Category'] = dado_limpo['Card_Category'].map({
    'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3
})
mapeamentos['Card_Category'] = {
    'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3
}

# Visualizar os mapeamentos
for coluna, mapeamento in mapeamentos.items():
    print(f"Coluna: {coluna}")
    print(f"Mapeamento: {mapeamento}")
    print("-" * 50)
                
#%% VERIFICAR COLUNAS DE ALFANUMÉRICOS

import pandas as pd

# Identificar colunas categóricas (alfanuméricas)
colunas_categoricas = dado_limpo.select_dtypes(include=['object']).columns

# Exibir os valores únicos (equivalente ao SELECT DISTINCT)
for coluna in colunas_categoricas:
    print(f"Coluna: {coluna}")
    print(f"Valores únicos: {dado_limpo[coluna].unique()}")
    print("-" * 50)

#%% SEPARAÇÃO DE VARIÁVEIS E CRIAÇÃO DE MODELO

X = dado_limpo.drop(columns=["Attrition_Flag", "CLIENTNUM","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"])  # Variáveis independentes
y = dado_limpo["Attrition_Flag"] # variável dependente

# Divisão em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,stratify=y, random_state=123)

# Primeira Versão da Árvore de Decisão
modelo_dt = DecisionTreeClassifier(random_state=123)

# Rodar Modelo Decision Tree
modelo_dt.fit(X_train, y_train)

# Rodar previsões
prediction_dt = modelo_dt.predict(X_test)

# Avaliar modelo
accuracy = accuracy_score(y_test, prediction_dt)
print(f"Acurácia: {accuracy:.4f}")

#%% MATRIZ DE CONFUSÃO

conf_matrix = confusion_matrix(y_test, prediction_dt)

plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d',cmap='Blues')
plt.title('Matriz de Confusão - Árvore de Decisão')
plt.xlabel('Previsto')
plt.ylabel('Observado')
plt.show()

#%% DEMAIS AVALIAÇÕES


# Precisão, Sensibilidade (Recall) e F1-Score para cada classe
precisao = precision_score(y_test, prediction_dt, average=None)
sensibilidade = recall_score(y_test, prediction_dt, average=None)
f1_scores = f1_score(y_test, prediction_dt, average=None)

# Classes da coluna 'Attrition_Flag'
classes = [0, 1]  # Substitua pelos valores reais de suas classes, se forem diferentes

# Exibir as métricas para cada classe
for i, classe in enumerate(classes):
    print(f"\nClasse Attrition_Flag: {classe}")
    print(f"Precisão: {precisao[i]:.4f}")
    print(f"Sensibilidade (Recall): {sensibilidade[i]:.4f}")
    print(f"F1-Score: {f1_scores[i]:.4f}")


#%% Definições de Plotagem
 
## plotar dentro do spyder
#     %matplotlib inline

## plotar fora do spyder
#     %matplotlib qt


#%% Versão Simplificada

# Plotar a árvore com limite de profundidade para visualização
plt.figure(figsize=(20, 10))
plot_tree(
    decision_tree=modelo_dt,
    feature_names=X.columns,
    class_names=['Churn', 'No Churn'],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=2  # Limita a profundidade exibida a 3 níveis
)
plt.savefig("decision_tree_limited.png", dpi=300, bbox_inches='tight')
plt.show()

#%% Versão Maior em PDF

from sklearn.tree import export_graphviz
import graphviz

# Exportar a árvore em formato DOT
dot_data = export_graphviz(
    modelo_dt, 
    out_file=None,
    feature_names=X.columns,           # Nomes das features
    class_names=['No Churn', 'Churn'], # Classes
    filled=True,                       # Cores para os nós
    rounded=True,                      # Nós arredondados
    special_characters=True            # Caracteres especiais
)

# Gerar e visualizar a árvore com Graphviz
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Salva como 'decision_tree.pdf'
graph.view()  # Abre o arquivo gerado
