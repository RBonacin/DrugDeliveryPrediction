import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt


# Carregar o arquivo CSV
dataset = pd.read_csv('binaryfingerprints_BIO_FEW.csv', encoding='latin-1', sep=',')



# Remover outliers da variável alvo 'logP'
#Q1 = dataset['logP'].quantile(0.25)
#Q3 = dataset['logP'].quantile(0.75)
#IQR = Q3 - Q1
#lower_bound = Q1 - 1.5 * IQR
#upper_bound = Q3 + 1.5 * IQR
#dataset = dataset[(dataset['logP'] >= lower_bound) & (dataset['logP'] <= upper_bound)]

# Seleção das features (todas exceto a última coluna)
features = dataset.columns[:-1]

# Definição das variáveis X (features) e y (variável alvo)
X = dataset[features]
y = dataset['Bioavailability']

# Normalização das features usando RobustScaler
#scaler = RobustScaler()
#X_scaled = scaler.fit_transform(X)

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Função para construção do modelo com hiperparâmetros variáveis
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(hp.Int('units_1', min_value=64, max_value=256, step=32),
                                 activation='relu',
                                 input_shape=(X_train.shape[1],)))

    # Adicionar camadas ocultas variáveis (1 a 3 camadas)
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(keras.layers.Dense(hp.Int(f'units_{i+2}', min_value=32, max_value=128, step=32),
                                     activation='relu'))

    model.add(keras.layers.Dense(1))  # Saída única para classificação

    # Escolher taxa de aprendizado ideal
    model.compile(optimizer=keras.optimizers.Adam(
                      learning_rate=hp.Choice('learning_rate', [0.01, 0.001, 0.0001])),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model

# 🔹 Criar o otimizador de hiperparâmetros
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=10,  # Número de modelos a testar
    directory='~/helderDL/DNN/BioFew',
    project_name='ANN_hyperparam_tuning'
)

# 🔹 Procurar pelos melhores hiperparâmetros
tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

# 🔹 Obter o melhor modelo encontrado
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# 🔹 Treinar o modelo final com os melhores hiperparâmetros
history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=1)


results = best_model.evaluate(X_test, y_test, verbose=0)
print("test loss, test acc:", results)

# Previsão nos dados de teste
y_pred = best_model.predict(X_test).flatten()
y_pred = np.round_(y_pred)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)
#auc = roc_auc_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))

# Exibir as métricas
print(f"Accuracy: {accuracy}")
#print(f"F1-Score: {f1}")
#print(f"AUC: {auc}")

print("Relatório de Classificação:\n", classification_report(y_test, y_pred_classes))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_classes))
