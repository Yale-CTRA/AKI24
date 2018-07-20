# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:08:31 2018

@author: adityabiswas
"""

import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import h5py
from matplotlib import pyplot as plt
import seaborn as sns


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

# data should already be in numpy format
def forwardImpute(data, pIndex):
    if len(np.shape(data)) == 1:
        data = data[:,None]
    for i in tqdm(range(len(pIndex))):
        start, stop, length = pIndex[i,:]
        lastRecord = data[start,:]
        for j in range(1,length):
            whereRecorded = np.isfinite(data[start+j,:])
            data[start+j,~whereRecorded] = lastRecord[~whereRecorded]
            lastRecord = data[start+j,:]
    return data

# data should be 1d numpy array.  returns boolean index selecting everything before the first 1
def selectUntilFirst(data, pIndex):
    select = np.ones(len(data), dtype = np.bool)
    for i in tqdm(range(len(pIndex))):
        start, stop, length = pIndex[i,:]
        dataSub = data[start:stop]
        idx = np.where(dataSub == 1)[0]
        if len(idx) > 0:
            select[start+idx[0]:stop] = False
    return select
        
def removeRedunancy(data, pIndex):
    select = np.ones(len(data), dtype = np.bool)
    for i in tqdm(range(len(pIndex))):
        start, stop, length = pIndex[i,:]
        dataSub = data[start:stop,:]
        reference = dataSub[0,:]
        for j in range(1,length):
            current = dataSub[j,:]
            if np.all(current == reference):
                select[start+j] = False
            else:
                reference = current
    return select
        

def qualityMetrics(Y, T, P, pIndex, plot = False):
    P = P[:,None]
    decisionBoundaries = np.linspace(0, 1, 51)[None,:]
    k = np.shape(decisionBoundaries)[1]
    boolEvents = np.zeros(len(pIndex), dtype = np.bool)
    for i in tqdm(range(len(pIndex))):
        start, stop, length = pIndex[i,:]
        if np.sum(Y[start:stop]) > 0:
            boolEvents[i] = True
    countEvents = np.sum(boolEvents)
    countNonEvents = len(pIndex) - countEvents
    
    resultsEvent = np.zeros((countEvents, k), dtype = np.bool)
    resultsNonEvent = np.zeros((countNonEvents, k), dtype = np.bool)
    
    counterEvent, counterNonEvent = 0, 0
    for i in tqdm(range(len(pIndex))):
        start, stop, length = pIndex[i,:]
        Pcurrent = np.repeat(P[start:stop,:], k, axis = 1)
        if boolEvents[i]: # run code for event
            d = np.repeat(decisionBoundaries, len(Pcurrent), axis = 0)
            allCrossings = Pcurrent >= d
            
            crossings = np.any(allCrossings, axis = 0)
            resultsEvent[counterEvent,:] = crossings
            counterEvent += 1
        else: # run code for non-event
            d = np.repeat(decisionBoundaries, length, axis = 0)
            allCrossings = Pcurrent >= d
            crossings = np.any(allCrossings, axis = 0)
            resultsNonEvent[counterNonEvent,:] = crossings
            counterNonEvent += 1
    
    resultsEvent = np.mean(resultsEvent, axis = 0)
    resultsNonEvent = 1 - np.mean(resultsNonEvent, axis = 0)
    
    # plot line graphs of classification accuracy for both event and non-events
    if plot:
        fig, ax = plt.subplots()
        ax.plot(decisionBoundaries[0,:], resultsEvent, 'k', linewidth = 1.5, label = 'Event')
        ax.plot(decisionBoundaries[0,:], resultsNonEvent, 'c--', linewidth = 1.5, label = 'Non-Event')
        legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", 
                           borderaxespad=0.)
        for label in legend.get_texts():
            label.set_fontsize('large')
        ax.set_xlabel('Decision Boundaries', fontsize=18)
        ax.set_ylabel('Classification Accuracy', fontsize=18)
        ax.tick_params(labelsize = 14)
        plt.show()
    
    # find where the two lines have the maximum product
    maxProd = np.max(resultsEvent*resultsNonEvent)
    idx = np.where(resultsEvent*resultsNonEvent == maxProd)[0][0]
    actualDecision = decisionBoundaries[0,idx]
    
    # plot distribution of timeTillEvent for event patients at chosen decision boundary
    if plot:
        counterEvent = 0
        T_events = np.ones(countEvents, dtype = np.float32)*72
        for i in tqdm(range(len(pIndex))):
            if boolEvents[i]:
                start, stop, length = pIndex[i,:]
                idxs = np.where(P[start:stop,0] > actualDecision)[0]
                if len(idxs) > 0:
                    T_events[counterEvent] = T[start:stop][idxs[0]]
                counterEvent += 1
        sns.distplot(T_events)
    
    return maxProd, T_events
            
    
    

    


#################################################################################
baseVars = ['AGE', 'RACE', 'MALE', 'bun', 'creatinine', 'potassium', 'bicarbonate', 'chloride', 'hemoglobin',
             'sodium', 'bipap', 'diuretic', 'antibiotic', 'narcotic', 'statin', 'transfuserbc',
             'cardiaccath', 'icu', 'ventorder', 'nsaid', 'acearbrenin', 'surgicalward',
             'contraststudy', 'wbcc', 'plateletcount', 'minCreat48', 'creatFirst']

def generateNames(name, includeTime = False):
    x = ['min', 'max', 'diff', 'mean', 'var']
    y = [i + 'Slope' for i in x] 
    z = [i + 'Curve' for i in x]
    names = x + y + z
    if includeTime:
        w = ['count', 'freq', 'sinceLastTime', 'minTimeDiff', 'maxTimeDiff', 'meanTimeDiff']
        names += [name + '2' for name in names]  ## need to fix later
    return [name + '_' + i for i in names]

#name = 'pressor'
#temp = flatten([value for value in dropVars.values()])
#print(searchColumns(data.columns.values, name))
#print(searchColumns(temp, name))

# fix race, fix the ever vars
everLoc = os.path.join('G:', os.sep, 'EHR_preprocessed', 'temporalFeatures', 'ever.h5')
h5_store = h5py.File(everLoc, 'r')
data[varGroups['med']] = np.array(h5_store.get('medications'))
data[varGroups['pro']] = np.array(h5_store.get('procedures'))
data['RACE'] = data['RACE'] == 'Black or African American'

# add the temporal vars for creatinine
def getTemporalData(name, timeWindow, includeTimes = False):
    location = os.path.join('G:', os.sep, 'EHR_preprocessed', 'temporalFeatures', name + '.h5')
    h5 = h5py.File(location, 'r')
    values = np.array(h5.get('value'+ '_' + timeWindow))
    if includeTimes:
        times = np.array(h5.get('time'+ '_' + timeWindow))
        values = np.concatenate((values, times), axis = 1)
    return values

timeWindow = '3days'
creatValues = getTemporalData('creatinine', timeWindow, True)
percentValues = getTemporalData('creatPercent', timeWindow)
bunValues = getTemporalData('bun', timeWindow)
bicarbValues = getTemporalData('bicarbonate', timeWindow, True)
chlorideValues = getTemporalData('chloride', timeWindow)
hemoValues = getTemporalData('hemoglobin', timeWindow, True)
sodiumValues = getTemporalData('sodium', timeWindow)

timeVarNames = generateNames('creatinine', True)
timeVarNames += generateNames('creatPercent')
timeVarNames += generateNames('bun')
timeVarNames += generateNames('bicarbonate', True)
timeVarNames += generateNames('chloride')
timeVarNames += generateNames('hemoglobin', True)
timeVarNames += generateNames('sodium')


# join data
#allInputVars = varGroups['lab'] + varGroups['med'] + varGroups['dem'] + varGroups['loc'] + \
#                   varGroups['pro'] + varGroups['vit']
X = data[baseVars].values.astype(np.float32)
X = np.concatenate([X, creatValues, percentValues, bunValues, bicarbValues,
                    chlorideValues, hemoValues, sodiumValues], axis = 1)

# remove all instances after aki stage 2 appears
# remove places theres's nothing to predict
# create outcomes, IDS, redo pIndex
select = selectUntilFirst(data['AKI2'].values, pIndex)
select = np.logical_and(removeRedunancy(X[:,:len(baseVars)], pIndex), select)
Y = data['future48_AKI2_overlap'].values
select = np.logical_and(select, np.isfinite(Y))

X = X[select,:]
Y = Y[select]
ID = data['PAT_ENC_CSN_ID'].values[select]
pIndex = getPatientIndices(ID)

# select, fill in nans, standardize
X = forwardImpute(X, pIndex)
X = imputeMeans(X)
X = standardize(X)

    
dataLoc = os.path.join('G:', os.sep, 'EHR_preprocessed', 'modelData', 'all.h5')
h5_store = h5py.File(dataLoc, 'w')
h5_store.create_dataset('X', data=X)
h5_store.create_dataset('Y', data=Y)
h5_store.create_dataset('ID', data=ID)
h5_store.create_dataset('select', data=select)
h5_store.close()


###########################################################################################

dataLoc = os.path.join('G:', os.sep, 'EHR_preprocessed', 'modelData', 'base_Creat.h5')
h5_store = h5py.File(dataLoc, 'r')
X = np.array(h5_store.get('X'))
Y = np.array(h5_store.get('Y'))
ID = np.array(h5_store.get('ID'))
select = np.array(h5_store.get('select'))
h5_store.close()

#T = data.loc[select,'timeTillAKI2'].values

pIndex = getPatientIndices(ID)
cutoff = pIndex[int(np.round(len(pIndex)*0.7)),0]
Xtrain, Xtest = X[:cutoff,:], X[cutoff:,:]
Ytrain, Ytest = Y[:cutoff], Y[cutoff:]
Train, Ttest = T[:cutoff], T[cutoff:]
IDtrain, IDtest = ID[:cutoff], ID[cutoff:]

pIndexTrain = getPatientIndices(IDtrain)
pIndexTest = getPatientIndices(IDtest)
sampleWeights = np.zeros(len(Xtrain))
for i in tqdm(range(len(pIndexTrain))):
    start, stop, length = pIndexTrain[i,:]
    sampleWeights[start:stop] = 1/length

##############################################################################################

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from Helper.utilities import showCoef


model = LR(class_weight = 'balanced', C = 1e-1)
model.fit(Xtrain, Ytrain, sample_weight = sampleWeights)
showCoef(model.coef_[0], baseVars + timeVarNames)


P = model.predict_proba(Xtest)[:,1]
print(AUC(Ytest, P))
## performance of 0.781 for base + allVars (1week)
## performance of 0.775 for base + creatVars (1week) ## DB 0.640
## performance of 0.749 for base
## performance of 0.683 for AKI stage 1

##################################################################################################
import xgboost as xgb

dtrain = xgb.DMatrix(Xtrain, label=Ytrain, weight = sampleWeights,
                     feature_names = baseVars + timeVarNames)
dtest = xgb.DMatrix(Xtest, label=Ytest, feature_names = baseVars + timeVarNames)

param = {'max_depth': 5, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic'}
param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 30
bst = xgb.train(param, dtrain, num_round, evallist)
P_xgb = bst.predict(dtest)
print(qualityMetrics(Ytest, Ttest, P_xgb, pIndexTest, plot = True))
bst.get_score()
## performance of 0.834 for base + allVars (1week)
## DB 0.693