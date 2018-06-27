# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:09:36 2018

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

from Helper.utilities import save, load, getPatientIndices


def search(data, name, timeName, reference, window, offset = 0, direction = 'forward'):
    """
    Inputs:
        data: pandas dataframe that has already been restricted to a particular patient encounter
        name: column name of variable thats being searched (for selecting non-NaNs)
        reference: is index of timestamp to which the search should be made with respect to
        window: time period of search
        offset: absolute value of amount of time offset from reference timestamp to begin window
        direction: whether to search into future or past. takes a string
    
    Returns: boolean index over encounter of where data of name col was recorded within search window
    """
    vals = data[name].values
    times = data[timeName].values
    finite = np.isfinite(vals)
    if direction == 'forward':
        start = times[reference] + offset
        select = np.logical_and(finite, np.logical_and(times >= start, times <= start + window))
    elif direction == 'backward':
        start = times[reference] - offset
        select = np.logical_and(finite, np.logical_and(times <= start, times >= start - window))
    else:
        raise ValueError("Please choose either 'forward' or 'backward' for direction parameter")
    return select



# returns engineered features that dont depend on lab values, only timings
# relevant for groups of vars that are measured in panels
def timeOnlyFE(times, interval, currentTime):
    count = len(times)
    if currentTime > 0:
        freq = count/min(interval, currentTime)
    else:
        freq = count
    
    if count > 0:    
        timeSinceLast = currentTime - times[-1]
    else:
        timeSinceLast = min(interval, currentTime)

    if count > 1:
        deltas = times[1:] - times[:-1]
        minTimeDiff, maxTimeDiff, meanTimeDiff = np.min(deltas), np.max(deltas), np.mean(deltas)
    else:
        minTimeDiff, maxTimeDiff, meanTimeDiff = np.nan, np.nan, np.nan
    return np.array([count, freq, timeSinceLast, minTimeDiff, maxTimeDiff, meanTimeDiff], 
                    dtype = np.float64)
        

    
# creates engineered features based on continuous event-based measurements
# min, max, largest total change, mean, and std for 0th, 1st, and 2nd derivative information
# len(times) must be at least 1
# returns 15 new values
def valuesFE(times, values):
    times = times/24 # helps prevent underflow issues
    count = len(times)
    if count > 1:
        # min, max, mu, and var for actual values
        m, M = np.min(values), np.max(values)
        deltas = times[1:] - times[:-1]
        totalTime = np.sum(deltas)
        avgVals = 0.5*(values[1:] + values[:-1])
        mean = np.sum(avgVals*deltas)/totalTime
        var = np.sum(np.power(avgVals - mean, 2)*deltas)/totalTime
        
        # min, max, mean, var first derivative approx
        diffVals = values[1:] - values[:-1]
        slopes = 10*diffVals/deltas  # helps prevent underflow issues
        minSlope, maxSlope = np.min(slopes), np.max(slopes)
        meanSlope = np.sum(slopes*deltas)/totalTime
        if count == 2:
            varSlope = 0
        else:
            varSlope = np.sum(np.power(slopes - meanSlope, 2)*deltas)/totalTime
        
              
        # min, max, mean, var second derivative approx
        if count > 2:
            curves = 10*(slopes[1:] - slopes[:-1])  # helps prevent underflow issues
            deltas2 = deltas[1:] + deltas[:-1]
            minCurve, maxCurve = np.min(curves), np.max(curves)
            totalDeltas2 = np.sum(deltas2)
            meanCurve = np.sum(curves*deltas2)/totalDeltas2
            if count == 3:
                varCurve = 0
            else:
                varCurve = np.sum(np.power(curves - meanCurve, 2)*deltas2)/totalDeltas2
        else:
            minCurve, maxCurve, meanCurve, varCurve = 0, 0, 0, 0
    else:
        m, M, mean, var = values[0], values[0], values[0], 0
        minSlope, maxSlope, meanSlope, varSlope = 0, 0, 0, 0
        minCurve, maxCurve, meanCurve, varCurve = 0, 0, 0, 0
    

    return np.array([m, M, M-m, mean, var, 
                     minSlope, maxSlope, maxSlope-minSlope, meanSlope, varSlope,
                     minCurve, maxCurve, maxCurve-minCurve, meanCurve, varCurve], dtype = np.float64)
    

    
def createTimeVaryingFeatures(data, pIndex, featureName, timeName, window = 7*24, 
                              includeTimeFeatures = False):
    m, k = len(data), len(pIndex)
    #preallocate
    valueFeatures = np.empty((m, 15), dtype = np.float64)
    valueFeatures.fill(np.nan)
    if includeTimeFeatures:
        timeFeatures = np.empty((m, 6), dtype = np.float64)
        timeFeatures.fill(np.nan)
    
    #start loop
    for i in tqdm(range(k)):
        start, stop, length = pIndex[i,:]
        encounter = data.iloc[start:stop,:]
        for j in range(length):
            times = encounter[timeName].values
            
            # find data to create features on
            select = search(encounter, featureName, timeName, j, window, 
                            offset = 0, direction = 'backward')
            timeSubset = times[select]
            creatinineSubset = encounter[featureName].values[select]
            
            # create features
            if includeTimeFeatures:
                currentTime = times[j]
                timeFeatures[start+j,:] = timeOnlyFE(timeSubset, interval = window, 
                                                    currentTime = currentTime)
            if len(timeSubset) > 0:
                valueFeatures[start+j,:] = valuesFE(timeSubset, creatinineSubset)

    
    if includeTimeFeatures:
        return valueFeatures, timeFeatures
    else:
        return valueFeatures

#expects data to be boolean numpy array
def createEver(data, pIndex):
    everData = np.zeros(np.shape(data), dtype = np.bool)
    n = np.shape(data)[1]
    for i in tqdm(range(len(pIndex))):
        start, stop, length = pIndex[i,:]
        current = np.zeros(n, dtype = np.bool)
        for j in range(length):
            current = np.logical_or(current, data[start+j,:])
            everData[start+j,:] = current
    return everData

###########################################################################################
### LOAD RELEVANT INFO
dataloc = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'AKI_base.h5')
data = pd.read_hdf(dataloc, 'data')
varGroupsfolder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data')
varGroups = load(loc = varGroupsfolder, name = 'varGroups2')


pIndex = getPatientIndices(data['PAT_ENC_CSN_ID'].values)
m, k = len(data), len(pIndex)



########################################################
### CREATING TIME VARYING FEATURES FOR ALL RELEVANT LABS
    

for var in ['creatinine', 'creatPercent', 'bun', 'potassium', 'bicarbonate', 
            'chloride', 'hemoglobin', 'sodium']:
    dataLoc = os.path.join('G:', os.sep, 'EHR_preprocessed', 'temporalFeatures', var + '.h5')
    h5_store = h5py.File(dataLoc, 'w')
    if var in['creatinine', 'bicarbonate', 'hemoglobin']:
        includeTimeFeatures = True
    else:
        includeTimeFeatures = False
        
    for window, windowName in zip([24 + 4, 3*24 + 4, 7*24 + 4], ['1day', '3days', '1week']):
        pair = createTimeVaryingFeatures(data, pIndex, var, 'los',
                                    window = window, includeTimeFeatures = includeTimeFeatures)
        if includeTimeFeatures:
            valueFeatures, timeFeatures = pair
            h5_store.create_dataset('value' + '_' + windowName, data=valueFeatures)
            h5_store.create_dataset('time' + '_' + windowName, data=timeFeatures)
        else:
            valueFeatures = pair
            h5_store.create_dataset('value' + '_' + windowName, data=valueFeatures)
        
    h5_store.close()
            


################################################################################
### CREATE EVER VARS FOR MEDS AND PROCEDURES

everLoc = os.path.join('G:', os.sep, 'EHR_preprocessed', 'temporalFeatures', 'ever.h5')
h5_store = h5py.File(everLoc, 'w')

everMeds = createEver(data[varGroups['med']].values, pIndex)
everProcedures = createEver(data[varGroups['pro']].values, pIndex)
h5_store.create_dataset('medications', data=everMeds)
h5_store.create_dataset('procedures', data=everProcedures)

h5_store.close()
















       