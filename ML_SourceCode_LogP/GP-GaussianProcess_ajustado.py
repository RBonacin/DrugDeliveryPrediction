# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
dataset = pd.read_csv('binaryfingerprintsLogP_ALL.csv', encoding='latin-1', sep=',')

# Remoção de outliers da variável alvo 'logP'
Q1 = dataset['logP'].quantile(0.25)
Q3 = dataset['logP'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
dataset = dataset[(dataset['logP'] >= lower_bound) & (dataset['logP'] <= upper_bound)]

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

# Definir kernels para busca
kernels = [
    C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1)),
    C(1.0, (1e-4, 1e1)) * Matern(nu=2.5),
    Matern(nu=2.5)
]

# Definir parâmetros para busca
param_grid = {
    'kernel': kernels,
    'n_restarts_optimizer': [10, 20, 30]
}

# Realizar busca de hiperparâmetros
gp_model = GaussianProcessRegressor(random_state=42)
grid_search = GridSearchCV(gp_model, param_grid, cv=5, scoring='neg_mean_squared_error',n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor valor dos hiperparâmetros
best_params = grid_search.best_params_
print(f"Melhores hiperparâmetros encontrados: {best_params}")

# Obter melhor modelo
best_model = grid_search.best_estimator_


print("Modelo GP treinado.")

# Treinar melhor modelo
best_model.fit(X_train, y_train)


# Prever nos dados de teste
y_pred, std_pred = best_model.predict(X_test, return_std=True)

# Avaliar modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Exibir métricas
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")

# Plotar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Log_p - Valores Reais")
plt.ylabel("Log_p - Valores Preditos")
plt.title("Gaussian Process Regression")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.savefig("GaussianProcess_logP_Bin.png", dpi=300, bbox_inches='tight')
