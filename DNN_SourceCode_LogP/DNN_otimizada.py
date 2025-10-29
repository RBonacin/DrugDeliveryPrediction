import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


# Carregar o arquivo CSV
dataset = pd.read_csv('binaryfingerprintsLogP_ALL.csv', encoding='latin-1', sep=',')



# Remover outliers da variável alvo 'logP'
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

# 🔹 Função para construção do modelo com hiperparâmetros variáveis
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(hp.Int('units_1', min_value=64, max_value=256, step=32),
                                 activation='relu',
                                 input_shape=(X_train.shape[1],)))

    # Adicionar camadas ocultas variáveis (1 a 10 camadas)
    for i in range(hp.Int('num_layers', 1, dnn_layers_ss)):
        model.add(keras.layers.Dense(hp.Int(f'units_{i+2}', min_value=32, max_value=128, step=32),
                                     activation='relu'))

    model.add(keras.layers.Dense(1))  # Saída única para regressão

    # Escolher taxa de aprendizado ideal
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.01, 0.001, 0.0001])),
                  loss='mse',
                  metrics=['mae'])
    return model

# 🔹 Criar o otimizador de hiperparâmetros
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    dnn_layers_ss=[1,2,3,4,5],
    max_trials=100,  # Número de modelos a testar
    directory='~/helderDL/DNN/All',
    project_name='ANN_hyperparam_tuning2'
)

# 🔹 Procurar pelos melhores hiperparâmetros
tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

# 🔹 Obter o melhor modelo encontrado
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# 🔹 Treinar o modelo final com os melhores hiperparâmetros
history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)

# Avaliação do modelo
eval_mse, eval_mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Melhor MSE: {eval_mse}")
print(f"Melhor MAE: {eval_mae}")


# Previsão nos dados de teste
y_pred = best_model.predict(X_test).flatten()

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(eval_mse)

print(f"R² Score: {r2}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 🔹 Plotar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Log_p - Valores Reais")
plt.ylabel("Log_p - Valores Preditos")
plt.title("Rede Neural Otimizada")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linha de identidade
plt.savefig("DNN_Opt_logP_Bin.png", dpi=300, bbox_inches='tight')

# Exibir as 10 primeiras previsões e valores reais
print('Primeiras 10 previsões =', y_pred[:10])
print('Primeiros 10 valores reais =', y_test.values[:10])
