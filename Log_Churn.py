import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


#%% LEITURA DO ARQUIVO

entrada = pd.read_csv("BankChurners.csv", sep=",")

#%% TRATAMENTO INICIAL DOS DADOS

# Remover todas as linhas com valores NA
dado_limpo = entrada.dropna()

# Mapear categorias binárias
category = {"Existing Customer": 1, "Attrited Customer": 0}
dado_limpo["Attrition_Flag"] = dado_limpo["Attrition_Flag"].map(category)

# Converter variáveis categóricas para numéricas
mapeamentos = {
    "Gender": {'M': 0, 'F': 1},
    "Education_Level": {
        'Uneducated': 0, 'High School': 1, 'College': 2, 'Graduate': 3,
        'Post-Graduate': 4, 'Doctorate': 5, 'Unknown': -1
    },
    "Marital_Status": {'Single': 0, 'Married': 1, 'Divorced': 2, 'Unknown': -1},
    "Income_Category": {
        'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2,
        '$80K - $120K': 3, '$120K +': 4, 'Unknown': -1
    },
    "Card_Category": {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
}

for coluna, mapeamento in mapeamentos.items():
    dado_limpo[coluna] = dado_limpo[coluna].map(mapeamento)

#%% SEPARAÇÃO DE VARIÁVEIS


X = dado_limpo.drop(columns=["Attrition_Flag", "CLIENTNUM","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2","Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"])  # Variáveis independentes
y = dado_limpo["Attrition_Flag"]  # Variável dependente

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

#%% MODELO DE REGRESSÃO LOGÍSTICA

# Inicializar o modelo
modelo_lr = LogisticRegression(max_iter=1000, random_state=123)

# Treinar o modelo
modelo_lr.fit(X_train, y_train)

# Realizar previsões
prediction_lr = modelo_lr.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, prediction_lr)
print(f"Acurácia - Regressão Logística: {accuracy:.4f}")

#%% MATRIZ DE CONFUSÃO

conf_matrix = confusion_matrix(y_test, prediction_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Regressão Logística')
plt.xlabel('Previsto')
plt.ylabel('Observado')
plt.show()

#%% DEMAIS AVALIAÇÕES

# Precisão, Sensibilidade (Recall) e F1-Score para cada classe
precisao = precision_score(y_test, prediction_lr, average=None)
sensibilidade = recall_score(y_test, prediction_lr, average=None)
f1_scores = f1_score(y_test, prediction_lr, average=None)

# Exibir métricas por classe
classes = [0, 1]
for i, classe in enumerate(classes):
    print(f"\nClasse Attrition_Flag: {classe}")
    print(f"Precisão: {precisao[i]:.4f}")
    print(f"Sensibilidade (Recall): {sensibilidade[i]:.4f}")
    print(f"F1-Score: {f1_scores[i]:.4f}")

#%% CURVA ROC E AUC

# Obter as probabilidades preditas
y_proba = modelo_lr.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva (1)

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='red')  # Linha diagonal para baseline
plt.title('Curva ROC - Regressão Logística')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print(f"AUC (Área Sob a Curva): {auc:.4f}")

#%% ANÁLISE DO CUTOFF (Limiar de Decisão)

# Criar uma análise dos diferentes valores de cutoff
cutoff_values = thresholds  # Limiar de decisão
tpr_values = tpr            # True Positive Rate (Sensibilidade)
fpr_values = fpr            # False Positive Rate

# Exibir os primeiros valores de cutoff e suas métricas associadas
print("Cutoff Analysis (Primeiros 5 valores):")
print(f"{'Cutoff':<10} {'TPR (Recall)':<15} {'FPR':<10}")
for i in range(5):  # Apenas os 5 primeiros para exemplo
    print(f"{cutoff_values[i]:<10.4f} {tpr_values[i]:<15.4f} {fpr_values[i]:<10.4f}")

# Escolher um cutoff personalizado (por exemplo, 0.5 ou otimizado por você)
custom_cutoff = 0.65  # Limiar padrão
y_pred_custom = (y_proba >= custom_cutoff).astype(int)

# Avaliar o modelo com o cutoff personalizado
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"\nAvaliação com cutoff = {custom_cutoff}:")
print(f"Acurácia: {accuracy_custom:.4f}")
print(f"Precisão: {precision_score(y_test, y_pred_custom):.4f}")
print(f"Sensibilidade (Recall): {recall_score(y_test, y_pred_custom):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_custom):.4f}")

#%% NOVOS INDICADORES PÓS-CUTTOFF


# Gerar a matriz de confusão com o novo cutoff
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)

# Plotar a matriz de confusão ajustada
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_custom, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matriz de Confusão com Cutoff = {custom_cutoff}')
plt.xlabel('Previsto')
plt.ylabel('Observado')
plt.show()

# Exibir a matriz de confusão no console (opcional)
print("Matriz de Confusão com Cutoff Ajustado:")
print(conf_matrix_custom)

#%% TESTE LÓGICO
print(modelo_lr.classes_)

custom_cutoff = 0.5
y_pred_custom = (y_proba >= custom_cutoff).astype(int)

print(f"Cutoff: {custom_cutoff}")
print(f"Classificação com cutoff: {custom_cutoff}")
print(pd.DataFrame({'Probabilidade': y_proba, 'Previsto': y_pred_custom, 'Real': y_test.values}).head(10))

#%%

# Obter as probabilidades preditas
y_proba = modelo_lr.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva (1)

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# Encontrar o índice do cutoff mais próximo ao custom_cutoff
cutoff_index = (abs(thresholds - custom_cutoff)).argmin()

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', label='Baseline')  # Linha diagonal para baseline

# Destacar o ponto correspondente ao novo cutoff
plt.scatter(fpr[cutoff_index], tpr[cutoff_index], color='orange', label=f'Cutoff = {custom_cutoff:.2f}', s=100)

# Configurações do gráfico
plt.title('Curva ROC - Regressão Logística')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Exibir o AUC
print(f"AUC (Área Sob a Curva): {auc:.4f}")

