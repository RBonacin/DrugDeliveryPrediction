import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
dataset = pd.read_csv('binaryfingerprints_BIO_ALL.csv', encoding='latin-1', sep=',')

# Remover outliers da variável alvo (não recomendado para variáveis categóricas, mas mantido se houver valores extremos numéricos)
#Q1 = dataset['biodisponibilidade'].quantile(0.25)
#Q3 = dataset['biodisponibilidade'].quantile(0.75)
#IQR = Q3 - Q1
#lower_bound = Q1 - 1.5 * IQR
#upper_bound = Q3 + 1.5 * IQR
#dataset = dataset[(dataset['biodisponibilidade'] >= lower_bound) & (dataset['biodisponibilidade'] <= upper_bound)]

# Seleção das features (todas exceto a última coluna)
features = dataset.columns[:-1]

# Definição das variáveis X e y
X = dataset[features].values
y = dataset['Bioavailability'].values

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(hp.Int('units_1', 64, 256, step=32),
                                 activation='relu',
                                 input_shape=(X_train.shape[1],)))

    for i in range(hp.Int('num_layers', 1, 6)):
        model.add(keras.layers.Dense(hp.Int(f'units_{i+2}', 32, 128, step=32),
                                     activation='relu'))

    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Saída binária

    model.compile(optimizer=keras.optimizers.Adam(
                      learning_rate=hp.Choice('learning_rate', [0.1, 0.01, 0.001, 0.0001])),
                  loss='binary_crossentropy',
                  metrics=['auc', keras.metrics.AUC(name='auc')])
    return model

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_auc',
    max_trials=20,
    directory='~/helderDL/DNN/BioAll',
    project_name='ANN_biodisp_classification5'
)


tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)


best_hps = tuner.get_best_hyperparameters(1)[0]
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=100, batch_size=32, verbose=1)


# best_model.evaluate(X_test, y_test, verbose=0)
#print(f"Acurácia: {eval_accuracy:.4f}")
#print(f"AUC: {eval_auc:.4f}")

y_pred_probs = best_model.predict(X_test).flatten()
y_pred_classes = (y_pred_probs > 0.5).astype(int)


print("Relatório de Classificação:\n", classification_report(y_test, y_pred_classes))
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_classes))
print("\n Accuracy : ", accuracy_score(y_test, y_pred_classes))
print("\n F1-Score : ", f1_score(y_test, y_pred_classes))
print("\n AUC : ", roc_auc_score(y_test, y_pred_classes))


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, label="Reais", alpha=0.6)
plt.scatter(range(len(y_test)), y_pred_classes, label="Preditos", alpha=0.6)
plt.title("Classificação - Biodisponibilidade")
plt.ylabel("Classe")
plt.xlabel("Amostras")
plt.legend()
plt.savefig("DNN_Class_Biodisp.png", dpi=300, bbox_inches='tight')

# Exibir primeiras previsões
print('Primeiras 10 probabilidades =', y_pred_probs[:10])
print('Primeiras 10 previsões =', y_pred_classes[:10])
print('Primeiros 10 valores reais =', y_test[:10])
