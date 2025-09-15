# importação de bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# Configura o estilo dos gráficos
sns.set_style("whitegrid")

"""
Esta seção carrega o dataset, prepara os dados para o modelo e divide-os em conjuntos de treino e teste.
"""

# URL direta do arquivo de dados no UCI
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Nomes de colunas
feature_groups = ["mean", "se", "worst"]
base_features = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension"
]
feature_cols = [f"{b}_{g}" for g in feature_groups for b in base_features]
cols = ["id", "diagnosis"] + feature_cols

# Carrega o dataset
df = pd.read_csv(DATA_URL, header=None, names=cols)

# Converte a variável 'diagnosis' para binário (M = 1, B = 0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Separa o dataset em variáveis X (atributos) e y (diagnóstico)
X = df[feature_cols]
y = df['diagnosis']

# Divide os dados em conjuntos de treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Dados de treino e teste preparados.")


""""
2. Treinamento do Modelo
Aqui, você treina o modelo de Regressão Logística, avalia seu desempenho com a acurácia no conjunto de teste e calcula métricas importantes usando a matriz de confusão.
"""

# Treina o modelo de Regressão Logística
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = model.predict(X_test)

# Exibe o score de acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo no conjunto de teste: {accuracy:.4f}")

# Mostra a matriz de confusão e as métricas de classificação
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.title('Matriz de Confusão')
plt.show()

# Relatório de classificação (precisão, recall, F1-score)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))

"""
3. Interpretação dos Coeficientes
Esta seção se concentra em extrair e visualizar a importância de cada variável para o modelo. Os coeficientes (betas) indicam o peso de cada atributo na decisão de classificação.
"""
# Extrai os coeficientes do modelo
coefficients = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
})

# Calcula o valor absoluto dos coeficientes para ordenação
coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
coefficients = coefficients.sort_values(by='abs_coefficient', ascending=False).reset_index(drop=True)

print("\nTabela de Coeficientes Ordenada por Importância:")
print(coefficients[['feature', 'coefficient']].head(10))

# Plota um gráfico de barras horizontais dos coeficientes
plt.figure(figsize=(12, 8))
sns.barplot(
    data=coefficients.sort_values(by='coefficient', ascending=False),
    x='coefficient',
    y='feature',
    palette='viridis'
)
plt.title("Importância dos Atributos (Coeficientes do Modelo)", fontsize=16)
plt.xlabel("Coeficiente", fontsize=12)
plt.ylabel("Atributo", fontsize=12)
plt.tight_layout()
plt.show()

"""
4. Visualizações Adicionais
Aqui, você cria gráficos avançados para avaliar o desempenho do modelo, como a Curva ROC e a Curva de Precisão-Recall, e também visualiza a fronteira de decisão.
"""

# a. Curva ROC (Receiver Operating Characteristic)
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC e AUC do Modelo de Regressão Logística')
plt.legend(loc="lower right")
plt.show()

# b. Curva de Precisão vs Recall
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Curva de Precisão vs Recall')
plt.show()

# c. Fronteira de Decisão (exemplo com 2 variáveis)
# Este trecho de código é mais complexo e requer a criação de uma grade de pontos
# para visualização. Vamos usar 'radius_mean' e 'area_mean' como exemplo.
X_2d = X[['radius_mean', 'area_mean']]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y, test_size=0.3, random_state=42)

# Treinamento de um novo modelo com apenas 2 variáveis
model_2d = LogisticRegression()
model_2d.fit(X_train_2d, y_train_2d)

# Cria a grade de pontos para a fronteira de decisão
x_min, x_max = X_2d['radius_mean'].min() - 1, X_2d['radius_mean'].max() + 1
y_min, y_max = X_2d['area_mean'].min() - 1, X_2d['area_mean'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 10))
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
plt.scatter(X_test_2d['radius_mean'], X_test_2d['area_mean'], c=y_test_2d, s=50, cmap=plt.cm.RdBu, edgecolors='k')
plt.xlabel('Radius Mean')
plt.ylabel('Area Mean')
plt.title('Fronteira de Decisão da Regressão Logística')
plt.show()