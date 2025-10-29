# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn 
import math
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split #Splitting data into training and testing sets
from sklearn.preprocessing import RobustScaler
from imblearn.metrics import specificity_score
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import layers
from keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


# Create a CNN  Model
def create_CNN_model():
    model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Saída única para regressão
    ])
    #Compiling model
    #model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-04), loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
    return model


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

model=create_CNN_model()
model.fit(X_train, y_train,epochs=30)
y_pred = model.predict(X_test)
#pred = np.round_(predictions)
#f1 = sklearn.metrics.f1_score(y_test,predictions)
#print('F1Score: ',f1)# Seleção das features (todas exceto a última coluna)

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

# Previsão nos dados de teste
y_pred = model.predict(X_test).flatten()
  
# Plotar resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Log_p - Valores Reais")
plt.ylabel("Log_p - Valores Preditos")
plt.title("Rede Neural Artificial (ANN)")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linha de identidade
plt.savefig("DNN_logP_Bin.png", dpi=300, bbox_inches='tight')
  








