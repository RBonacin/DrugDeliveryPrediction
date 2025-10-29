
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore
from rdkit.Chem import AllChem


# Carregar o arquivo CSV
df = pd.read_csv('resultados_smiles_logP.csv', encoding='latin-1', sep=';')

# Verificar e converter a coluna 'SMILES' para strings, substituindo NaN por strings vazias
df['SMILES'] = df['SMILES'].astype(str).replace('nan', '')


for i in range(2048):
    print('fp'+str(i)+',',end='')

print('logP')
    
# Convertendo a coluna 'SMILES' em uma lista
smiles_list = df['SMILES'].tolist()
logP_list = df['logP'].tolist()


fpgen = AllChem.GetMorganGenerator(radius=2)

for i in range(len(smiles_list)):
    print(smiles_list[i])
    mol = Chem.MolFromSmiles(smiles_list[i])
    fp1 = fpgen.GetFingerprint(mol)
    strBin = fp1.ToBitString()
    strBin = strBin.replace('0','0,')
    strBin= strBin.replace('1','1,')
    print(strBin + str(logP_list[i]))



