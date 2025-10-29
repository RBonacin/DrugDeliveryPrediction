import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from keras_tuner import RandomSearch
from keras_tuner import Objective
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD


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

X_train_reg = X_train 

# 🔹 Função para construção do modelo com hiperparâmetros variáveis
def build_model(hyperparams):
    model = Sequential()
    model.add(layers.Input(shape=(X_train_reg.shape[1],)))
    model.add(layers.Dense(units=hyperparams.Int("units_l1", 16, 50, step=16),
                           use_bias=hyperparams.Boolean("bias_l1"),
                           activation=hyperparams.Choice("act_l1", ["relu", "tanh"])
                          ))
    model.add(layers.Dense(units=hyperparams.Int("units_l2", 16, 50, step=16),
                           use_bias=hyperparams.Boolean("bias_l2"),
                           activation=hyperparams.Choice("act_l2", ["relu", "tanh"])
                          ))
    model.add(layers.Dense(1))

    optim=hyperparams.Choice("optimizer",["sgd","rmsprop","adam"])
    model.compile(optim, loss="mean_squared_error", metrics=["mean_squared_error"])

    return model


# 🔹 Criar o otimizador de hiperparâmetros
tuner =  RandomSearch(hypermodel=build_model,
                      objective="val_mean_squared_error",
                      #objective=Objective(name="val_mean_squared_error",direction="min"),
                      max_trials=5,
                      #seed=123,
                      project_name="Regression",
                      overwrite=True
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
