# importação de bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# URL direta do arquivo de dados no UCI:
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

# Nomes de colunas (32 no total: id, diagnosis + 30 atributos)
feature_groups = ["mean", "se", "worst"]
base_features = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry",
    "fractal_dimension"
]
feature_cols = [f"{b}_{g}" for g in feature_groups for b in base_features]
cols = ["id", "diagnosis"] + feature_cols  # total 32


# Carrega CSV sem header e atribui nomes
df = pd.read_csv(DATA_URL, header=None, names=cols)

# Questão 1 item 1
# 1. Mostrar o número de linhas e colunas
print(f"Número de linhas e colunas: {df.shape}")

# Questão 1 item 2
# Mostrar as 5 primeiras linhas
print("\nAs 5 primeiras linhas do DataFrame:")
print(df.head())

################### Segunda parte da questão ########################
# Ajuste no estilo dos gráficos para melhor visualização
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-whitegrid")


# a. Histograma de 'radius_mean' separado por diagnóstico
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="radius_mean",
    hue="diagnosis",
    multiple="stack",
    palette={"M": "salmon", "B": "skyblue"},
    kde=True
)
plt.title("Distribuição de 'radius_mean' por Diagnóstico", fontsize=16)
plt.xlabel("Radius Mean", fontsize=12)
plt.ylabel("Contagem", fontsize=12)
plt.legend(title="Diagnóstico", labels=["Maligno", "Benigno"])
plt.show()


# b. Boxplot de 'area_mean' comparando Benignos vs Malignos
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="diagnosis",
    y="area_mean",
    palette={"M": "salmon", "B": "skyblue"}
)
plt.title("Boxplot de 'area_mean' por Diagnóstico", fontsize=16)
plt.xlabel("Diagnóstico (M=Maligno, B=Benigno)", fontsize=12)
plt.ylabel("Area Mean", fontsize=12)
plt.show()


# c. Scatterplot entre 'radius_mean' e 'area_mean'
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="radius_mean",
    y="area_mean",
    hue="diagnosis",
    palette={"M": "salmon", "B": "skyblue"},
    s=70,  # Tamanho dos pontos
    alpha=0.8  # Transparência
)
plt.title("Relação entre 'radius_mean' e 'area_mean' por Diagnóstico", fontsize=16)
plt.xlabel("Radius Mean", fontsize=12)
plt.ylabel("Area Mean", fontsize=12)
plt.show()


# d. Heatmap de correlação
plt.figure(figsize=(8, 7))
# Seleciona as colunas para o heatmap
corr_cols = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean"]
corr_matrix = df[corr_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,  # Mostra os valores de correlação no mapa
    cmap="coolwarm",
    fmt=".2f",  # Formata os valores com 2 casas decimais
    linewidths=.5
)
plt.title("Heatmap de Correlação das Variáveis 'Mean'", fontsize=16)
plt.show()