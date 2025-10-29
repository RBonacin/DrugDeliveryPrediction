import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from itertools import product
import matplotlib.pyplot as plt

# Carregar o dataset
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

first_layer_neurons = np.arange(80, 160, 20)
second_layer_neurons = np.arange(40, 120, 20)
hidden_layer_sizes = list(product(first_layer_neurons, second_layer_neurons))
parameters = {'solver': ['adam','lbfgs'], 'max_iter': [400, 600,1000], 'alpha': 10.0 ** -np.arange(1,10,2), 'hidden_layer_sizes':hidden_layer_sizes, 'random_state':[3,6,8], "activation": ["logistic","relu"]}

mlp_model = GridSearchCV(MLPRegressor(), parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)


# Criação do modelo MLP
#mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=500, random_state=42)

# Validação cruzada com 5 folds
#cv_scores = cross_val_score(mlp_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error',n_jobs=-1)
#mean_cv_score = np.mean(np.abs(cv_scores))

#print(f"Mean Cross-Validation MSE: {mean_cv_score}")

# Treinamento do modelo MLP com os dados de treinamento completos
mlp_model.fit(X_train, y_train)

# Melhor valor dos hiperparâmetros
best_params = mlp_model.best_params_
print(f"Melhores hiperparâmetros encontrados: {best_params}")

print("Modelo MLP treinado.")

# Previsão nos dados de teste
y_pred = mlp_model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Exibir as métricas
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plotar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Log_p - Valores Reais")
plt.ylabel("Log_p - Valores Preditos")
plt.title("MLP Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linha de identidade
plt.savefig("MLP_logP_Bin.png", dpi=300, bbox_inches='tight')

# Exibir as 10 primeiras previsões e valores reais
print('Primeiras 10 previsões =', y_pred[:10])
print('Primeiros 10 valores reais =', y_test.values[:10])
