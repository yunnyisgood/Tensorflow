import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import math

#fingerfrint name to use
ffpp = "pattern"

train = pd.read_csv('../_data/samsung_sc_discovery/train.csv')
dev = pd.read_csv('../_data/samsung_sc_discovery/dev.csv')
test = pd.read_csv('../_data/samsung_sc_discovery/test.csv')
ss  = pd.read_csv('../_data/samsung_sc_discovery/sample_submission.csv')

train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']

train_fps = []#train fingerprints
train_y = [] #train y(label)

for index, row in train.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])
    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    train_fps.append(fp)
    train_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

#fingerfrint object to ndarray
np_train_fps = []
for fp in train_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_train_fps.append(arr)

np_train_fps_array = np.array(np_train_fps)

print(np_train_fps_array.shape)
print(len(train_y))

test_fps = []#test fingerprints
test_y = [] #test y(label)

for index, row in test.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])

    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    test_fps.append(fp)
    test_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

np_test_fps = []
for fp in test_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_test_fps.append(arr)

np_test_fps_array = np.array(np_test_fps)

print(np_test_fps_array.shape)
print(np.array(test_y).shape)
print(len(test_y))

print(np_train_fps_array.shape)
print(np.array(train_y))

np.save('../_data/sc_train_x.npy', arr=np_train_fps_array)
np.save('../_data/sc_train_y.npy', arr=np.array(train_y))
np.save('../_data/sc_test_x.npy', arr=np_test_fps_array)

'''


'''