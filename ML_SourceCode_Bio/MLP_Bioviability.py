import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from itertools import product

# Montar o Google Drive
#from google.colab import drive
#drive.mount('/content/drive')

# Carregar o arquivo CSV
dataset = pd.read_csv('binaryfingerprints_BIO_ALL.csv', encoding='latin-1', sep=',')

print("Primeiras linhas do dataset:")
print(dataset.head())
print("Estatísticas descritivas do dataset:")
print(dataset.describe())

# Remoção de outliers da variável alvo 'logP'
#Q1 = dataset['Bioavailability'].quantile(0.25)
#Q3 = dataset['Bioavailability'].quantile(0.75)
#IQR = Q3 - Q1
#lower_bound = Q1 - 1.5 * IQR
#upper_bound = Q3 + 1.5 * IQR

#dataset = dataset[(dataset['logP'] >= lower_bound) & (dataset['logP'] <= upper_bound)]
print(f"Total de linhas no DataFrame: {dataset.shape[0]}")

# Seleção das features (todas exceto a última coluna)
features = dataset.columns[:-1]

# Definição das variáveis X (features) e y (variável alvo)
X = dataset[features]
y = dataset['Bioavailability']

# Normalização das features usando RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir a grade de hiperparâmetros para MLP
first_layer_neurons = np.arange(80, 160, 20)
second_layer_neurons = np.arange(60, 120, 20)
hidden_layer_sizes = list(product(first_layer_neurons, second_layer_neurons))

parameters = {'solver': ['adam','lbfgs'], 'max_iter': [400,600], 'alpha': 10.0 ** -np.arange(1,10,2), 'hidden_layer_sizes':hidden_layer_sizes, 'random_state':[3,6], "activation": ["logistic","relu"]}

#parameters = {'solver': ['adam','lbfgs','sgd'], 'max_iter': [400,600,1000], 'alpha': 10.0 ** -np.arange(1,10,2), 'hidden_layer_sizes':hidden_layer_sizes, 'random_state':[3,6], "activation": ["logistic","relu"]}

# Criação do modelo MLP
mlp_model = MLPClassifier(random_state=1)

# Realizar a busca com validação cruzada
grid_search = GridSearchCV(mlp_model, param_grid=parameters, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
mlp_model = grid_search

# Melhor valor dos hiperparâmetros
best_params = grid_search.best_params_
print(f"Melhores hiperparâmetros encontrados: {best_params}")

# Treinar o modelo final com os melhores hiperparâmetros
#mlp_model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
#svm_model.fit(X_train, y_train)

print("Modelo MLP treinado.")

# Previsão nos dados de teste
y_pred = mlp_model.predict(X_test)
#y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))

# Exibir as métricas
print(f"Accuracy: {accuracy}")
print(f"F1-Score: {f1}")
print(f"AUC: {auc}")

# Plotar resultados
#plt.figure(figsize=(10, 6))
#plt.scatter(y_test, y_pred)
#plt.xlabel("Bioavailability - Valores Reais")
#plt.ylabel("Bioavailability - Valores Preditos")
#plt.title("SVM")
#plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Linha de identidade
#plt.savefig("SVM_Bioavailability_Bin.png", dpi=300, bbox_inches='tight')

# Exibir as 10 primeiras previsões e valores reais
print('Primeiras 10 previsões =', y_pred[:10])
print('Primeiros 10 valores reais =', y_test.values[:10])
