import pandas as pd
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler

dataset = pd.read_csv('binaryfingerprintsLogP_ALL.csv')

print("Primeiras linhas do dataset:")
print(dataset.head())
print("Estatísticas descritivas do dataset:")
print(dataset.describe())

# Remoção de outliers da variável alvo 'logP'
Q1 = dataset['logP'].quantile(0.25)
Q3 = dataset['logP'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

dataset = dataset[(dataset['logP'] >= lower_bound) & (dataset['logP'] <= upper_bound)]
print(f"Total de linhas no DataFrame após remover outliers: {dataset.shape[0]}")

# Seleção das features (todas exceto a última coluna)
features = dataset.columns[:-1]

# Definição das variáveis X (features) e y (variável alvo)
X = dataset[features]
y = dataset['logP']

# Normalização das features usando RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# create an SVR model with a linear kernel 
svr = SVR(kernel='linear') 

# Definir uma grade de parâmetros para ajuste
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# Configurar GridSearchCV com 5-fold cross-validation
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Treinar o modelo com ajuste de hiperparâmetros
grid_search.fit(X_train, y_train)

# Obter o melhor modelo
best_model = grid_search.best_estimator_

# Fazer previsões no conjunto de teste
y_pred = best_model.predict(X_test)


# Criação e treinamento do modelo Random Forest
#rf_model = LinearRegression()
#rf_model.fit(X_train, y_train)
print("Modelo Random SVR treinado.")

# Previsão nos dados de teste
#y_pred = rf_model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Estatística F
f_statistic = (r2 / X_test.shape[1]) / ((1 - r2) / (len(y_test) - X_test.shape[1] - 1))

# Exibir as métricas
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print(f"R² Ajustado: {adjusted_r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Estatística F: {f_statistic}")

# Plotar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Toxicidade - Valores Reais")
plt.ylabel("Toxicidade - Valores Preditos")
plt.title("Random Forest")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linha de identidade
plt.savefig("SVR_logP_Bin.png",dpi=300, bbox_inches='tight')

# Exibir as 10 primeiras previsões e valores reais
print('Primeiras 10 previsões =', y_pred[:10])
print('Primeiros 10 valores reais =', y_test.values[:10])
