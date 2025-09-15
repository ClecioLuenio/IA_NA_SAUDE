import pandas as pd
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
