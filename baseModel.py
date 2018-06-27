# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:19:29 2018

@author: adityabiswas
"""



import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import h5py

usingInterpreter = True
if usingInterpreter:
    root = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects')
else:
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path.append(root)

from Helper.preprocessing import imputeMeans, standardize
from Helper.utilities import save, load, getPatientIndices

dataloc = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'AKI_base.h5')
data = pd.read_hdf(dataloc, 'data')
pIndex = getPatientIndices(data['PAT_ENC_CSN_ID'].values)
m, k = len(data), len(pIndex)

varGroupsfolder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data')
varGroups = load(loc = varGroupsfolder, name = 'varGroups2')

# data should already be in numpy format with a common dtype across columns
def forwardImpute(data, pIndex):
    if len(np.shape(data)) == 1:
        data = data[:,None]
    for i in tqdm(range(k)):
        start, stop, length = pIndex[i,:]
        lastRecord = data[start,:]
        for j in range(1,length):
            whereRecorded = np.isfinite(data[start+j,:])
            data[start+j,~whereRecorded] = lastRecord[~whereRecorded]
            lastRecord = data[start+j,:]
    return data
    

#################################################################################
baseVars = ['AGE', 'RACE', 'MALE', 'bun', 'creatinine', 'potassium', 'bicarbonate', 'chloride', 'hemoglobin',
             'sodium', 'bipap', 'diuretic', 'antibiotic', 'narcotic', 'statin', 'transfuserbc',
             'cardiaccath', 'icu', 'ventorder', 'nsaid', 'acearbrenin', 'surgicalward',
             'contraststudy', 'wbcc', 'plateletcount', 'minCreat48', 'creatFirst']

#name = 'pressor'
#temp = flatten([value for value in dropVars.values()])
#print(searchColumns(data.columns.values, name))
#print(searchColumns(temp, name))


everLoc = os.path.join('G:', os.sep, 'EHR_preprocessed', 'temporalFeatures', 'ever.h5')
h5_store = h5py.File(everLoc, 'r')
data[varGroups['med']] = np.array(h5_store.get('medications'))
data[varGroups['pro']] = np.array(h5_store.get('procedures'))

data['RACE'] = data['RACE'] == 'Black or African American'

X = data[baseVars].values.astype(np.float32)
X = forwardImpute(X, pIndex)
X = imputeMeans(X)
X = standardize(X)

cutoff = pIndex[int(np.round(len(pIndex)*0.7)),0]
Xtrain, Xtest = X[:cutoff,:], X[cutoff:,:]
Y = data['future48_AKI2_overlap'].values
Ytrain, Ytest = Y[:cutoff], Y[cutoff:]
IDtrain = data['PAT_ENC_CSN_ID'].values[:cutoff]

selectTrain, selectTest = np.isfinite(Ytrain), np.isfinite(Ytest)
Xtrain, Ytrain, IDtrain = Xtrain[selectTrain,:], Ytrain[selectTrain], IDtrain[selectTrain]
Xtest, Ytest = Xtest[selectTest,:], Ytest[selectTest]

pIndexSub = getPatientIndices(IDtrain)
sampleWeights = np.zeros(len(Xtrain))
for i in tqdm(range(len(pIndexSub))):
    start, stop, length = pIndexSub[i,:]
    sampleWeights[start:stop] = 1/length

#X = np.concatenate((valueFeatures, timeFeatures, data2[['los']].values, 
#                    data2[['creatinine']].values), axis = 1)

##############################################################################


from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from Helper.utilities import showCoef
model = LR(class_weight = 'balanced', C = 1e-1)

model.fit(Xtrain, Ytrain)#,sample_weight = sampleWeights)
P = model.predict_proba(Xtest)[:,1]
model.coef_
print(AUC(Ytest, P))#, sample_weight = sampleWeights))

## performance is around 0.83 currently