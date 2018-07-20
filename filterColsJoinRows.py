# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 14:12:14 2018

@author: adityabiswas
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
from copy import deepcopy as copy

usingInterpreter = True
if usingInterpreter:
    root = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects')
else:
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path.append(root)

from Helper.utilities import save, load, getPatientIndices
from Helper.clean import mixed2float
from Helper.preprocessing import convertISO8601


########################################################################################
flatten = lambda l: [item for sublist in l for item in sublist]

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

def group(data, boundaries):
    """
    Input: a dictionary with keys correspondng to variable group names
                        and values that are length-2 lists
           List members correspond to first and final variable name in that group
           Assumes that groups are contiguous
    Returns: pair of dictionaries: one mapping to variable names and other to corresponding slices
    """
    varGroups = {}
    sliceGroups = {}
    for key in boundaries:
        indices = data.columns.get_indexer(boundaries[key])
        sliceGroups[key] = slice(indices[0], indices[1]+1)
        varGroups[key] = list(data.columns[sliceGroups[key]])
    return varGroups, sliceGroups


def shiftIndices(m, startLoc, endLoc, insertLoc):
    # m is length of data
    segment = np.arange(startLoc, endLoc)
    if insertLoc <= startLoc: # shift segment up
        pre = np.arange(insertLoc)
        middle = np.arange(insertLoc, startLoc)
        post = np.arange(endLoc, m)
        idx = np.concatenate([pre, segment, middle, post])
    else: # shift segment down
        pre = np.arange(startLoc)
        middle = np.arange(endLoc, insertLoc)
        post = np.arange(insertLoc, m)
        idx = np.concatenate([pre, middle, segment, post])
    return idx

def getDuplicates(IDs):
    seenIDs = [IDs[0]]
    duplicateIDs = []
    current = IDs[0]
    for i in tqdm(range(1,len(data))):
        d = IDs[i]
        if current != d:
            current = d
            if not (d in seenIDs):
                seenIDs.append(d)
            else:
                duplicateIDs.append(d)
    return duplicateIDs

getIntIndex = lambda col, val: np.arange(len(col))[col==val]

def reorderByEncounter(data, name):
    m = len(data)
    newOrder = np.arange(m)
    IDs = copy(data[name].values)
    
    duplicateIDs = getDuplicates(IDs)
    print('\n', len(duplicateIDs), ' encounters split')
    
    # keep doing passes until everything is fixed
    while len(duplicateIDs) > 0:
        for i, ID in tqdm(enumerate(duplicateIDs)):
            intLocs = getIntIndex(IDs, ID)
            trans = np.where((intLocs[1:] - intLocs[:-1]) != 1)[0]
            if len(trans) > 0: # check must be done since previous iters might fix later ones
                trans = trans[0]
                idx = shiftIndices(data, intLocs[trans+1], intLocs[-1]+1, intLocs[trans]+1)
                IDs = IDs[idx]
                newOrder = newOrder[idx]
        duplicateIDs = getDuplicates(IDs)
     
    return data.iloc[newOrder,:]

##############################################################################################
## read in only very first section of data to do column organization and var deletion
data = pd.read_csv(os.path.join('G:', os.sep, 'EHR_preprocessed', 'final_part0.csv'))
data = reorderByEncounter(data, 'PAT_ENC_CSN_ID')

pIndex = getPatientIndices(data['PAT_ENC_CSN_ID'].values)
m, k = len(data), len(pIndex)

# add urine_output, creat, and new vars separately. do info and demo vars manually
# labs include vitals
boundaries = {'loc': ['emergencyroom','pacu'], 
              'lab': ['baseex','cdfeia'],
              'vit': ['systolic','arterial_diastolic'],
              'med': ['lorazepam','immunomodulators'], 
              'pro': ['telemetry','hospice'],
              'grp': ['ELX_GRP_1', 'TOT_GRP']}
varGroups, _ = group(data, boundaries)
varGroups['lab'] = varGroups['lab'] + ['urine_output', 'creatinine', 'proteingap']
varGroups['vit'] = varGroups['vit'] + ['MAP', 'pulse_pressure']
varGroups['inf'] = ['PAT_MRN_ID', 'PAT_ENC_CSN_ID', 'time', 'HOSP_ADMSN_TIME', 'HOSP_DISCH_TIME',
                 'HOSP_ADMSN_TYPE', 'DISCHARGE_DISPSN', 'DISCH_DISP_C', 'PAT_STATUS', 'DEATH_DATE']
varGroups['dem'] = ['AGE', 'MALE',  'RACE', 'HISPANIC', 'HEIGHT', 'WEIGHT', 'BMI']

# sanity check
assert sum([len(varGroups[key]) for key in varGroups]) == len(data.columns)

# fill procedure variables with 0s if nan
# turn meds into simple event records
data.loc[:,varGroups['pro']] = data.loc[:,varGroups['pro']].fillna(0)
print('Turned procedures binary')
data.loc[:,varGroups['med']] = np.isfinite(data.loc[:,varGroups['med']].values)
print('Turned medications binary')



## WILL REMOVE LOCATION VARS LATER IN FILE AFTER DATA IS LOADED
## first must convert all labs to numericals
convertVars = list(data[varGroups['lab']].columns[data[varGroups['lab']].dtypes == np.dtype('O')])

for var in convertVars:
    newCol = np.empty(len(data), dtype = np.float16)
    newCol.fill(np.nan)
    select = ~(data[var].values.astype(np.str) == 'nan')
    newCol[select] = 1
    data[var] = newCol
print('Converted ordinal laboratory data to binary')

# discover the proportions of patient encounters at which variables are measured at least once

countFuncs = {'lab': lambda x: x.count() > 0, 
              'vit': lambda x: x.count() > 0, 
              'loc': lambda x: np.any(x.values == 1, axis = 0),
              'med': lambda x: np.sum(x.values, axis = 0) > 0, 
              'pro': lambda x: np.sum(x.values, axis = 0) > 0}

measured = {key: np.zeros((k, len(varGroups[key])), dtype = np.bool) for key in countFuncs}

for i in tqdm(range(k)):
    start, stop, _ = pIndex[i,:]
    encounter = data.iloc[start:stop]
    for key, func in countFuncs.items():
        names = varGroups[key]
        measured[key][i,:] = func(encounter[names])
        
print('Measured variable recording rates')
   
proportions = {}     
for key in countFuncs:
    proportions[key] = pd.Series(np.sum(measured[key], axis = 0)/k, index = varGroups[key])


# figure out which variables to drop
# labs and vitals less than 10%
# meds and procedures less than 5%
# location less than 5%
thresholds = {'lab': 0.1, 'vit': 0.1, 'loc': 0.05, 'med': 0.05, 'pro': 0.05}
dropVars = {}
for key, value in thresholds.items():
    dropVars[key] = list(np.array(varGroups[key])[proportions[key].values < value])


# actually drop
# fix varGroup to reflect deletions BUT DONT DO LOCATION VARS. DO LATER
    
for key in thresholds:
    if key != 'loc':
        value = varGroups[key]
        varGroups[key] = list(set(value) - set(dropVars[key]))

# save columns we're keeping in new file
varGroups['inf'] = ['PAT_MRN_ID', 'PAT_ENC_CSN_ID', 'time']

varGroupsfolder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data')
save(obj = varGroups, loc = varGroupsfolder, name = 'varGroups')
save(obj = dropVars, loc = varGroupsfolder, name = 'dropVars')

print('Pickled first iteration of varGroups dictionary')


##################################################################################################
## must use dropVars dict created earlier for location info
varGroupsfolder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data')

#### load and join all data
folder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data')
varGroups = load(loc = folder, name = 'varGroups')
colNames = flatten(varGroups.values())

data = [None for i in range(4)]
for i in tqdm(range(4)):
    loc = os.path.join('G:', os.sep, 'EHR_preprocessed', 'final_part' + str(i) + '.csv')
    data[i] = pd.read_csv(loc, usecols = colNames)
    data[i].set_index('PAT_ENC_CSN_ID', inplace = True, drop = False)
    
print('Loaded all 4 data parts')
data = pd.concat(data, axis = 0)
print('Concatenated all data together')

############################################################################################
## ecgheart rate was missed in preprocessing earlier and needs to be done here
## also bloodcult, urinecx is unnecessary to keep even though its apparently often recorded
## abogrouping needs to be encoded as an A and B marker
## ordinal vars need to be converted to numeric
data['ecgheartrate'] = mixed2float(data['ecgheartrate'].values)

whereA, whereB = data['abogrouping'].values == 'A', data['abogrouping'].values == 'B'
whereAB = data['abogrouping'].values == 'AB'
whereNaN = data['abogrouping'].values.astype(np.dtype('unicode')) == 'nan'
markerA = 1.0*np.logical_or(whereA, whereAB)
markerA[whereNaN] = np.nan
markerB = 1.0*np.logical_or(whereB, whereAB)
markerB[whereNaN] = np.nan

data.insert(data.columns.get_loc('abogrouping'), 'bloodtypeA', markerA)
data.insert(data.columns.get_loc('abogrouping'), 'bloodtypeB', markerB)


data.drop(columns = ['bloodcult', 'urinecx', 'abogrouping'], inplace = True)
varGroups['lab'] = list(set(varGroups['lab']) - set(['bloodcult', 'urinecx', 'abogrouping']))
varGroups['lab'] += ['bloodtypeA', 'bloodtypeB']


def convertOrdinal(col):
    convert = {'negative': 0, 'trace': 1, '1+': 2, '2+': 3, '3+': 4}
    newCol = np.empty(len(col), dtype = np.float16)
    newCol.fill(np.nan)
    recorded = ~(col.astype(np.dtype('unicode')) == 'nan')
    for i in range(len(col)):
        if recorded[i]:
            newCol[i] = convert[col[i]]
    return newCol

convertVars = list(data[varGroups['lab']].columns[data[varGroups['lab']].dtypes == np.dtype('O')])
for var in tqdm(convertVars):
    data[var] = convertOrdinal(data[var].values)

print('Converted ordinal vars to numerical')

# fill procedure variables with 0s if nan
# turn meds into simple event records
data[varGroups['pro']] = data[varGroups['pro']].fillna(0)
print('Turned procedures binary')
data[varGroups['med']] = np.isfinite(data[varGroups['med']].values)
print('Turned medications binary')


## remove useless vars
removeRowFuncDict = {'lab': lambda x: np.all(np.isnan(x), axis = 1), 
                      'vit': lambda x: np.all(np.isnan(x), axis = 1),
                      'loc': lambda x: ~np.any(x == 1, axis = 1),
                      'med': lambda x: np.sum(x, axis = 1) == 0, 
                      'pro': lambda x: np.sum(x, axis = 1) == 0}

remove = np.zeros((len(data), 5), dtype = np.bool)
for idx, key in enumerate(removeRowFuncDict):
    remove[:,idx] = removeRowFuncDict[key](data[varGroups[key]].values)
remove = np.all(remove, axis = 1)

data = data.loc[~remove,:]
print('Removed all rows without new info recorded')

dataloc = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'Data.h5')
data.to_hdf(dataloc, 'data')
save(obj = varGroups, loc = varGroupsfolder, name = 'varGroups')



#####################################################################################################
#data = pd.read_hdf(loc, 'data')
#load(loc = varGroupsfolder, name = 'varGroups')

## resort all data such that patient encounters are properly placed together
data = reorderByEncounter(data, 'PAT_ENC_CSN_ID')
pIndex = getPatientIndices(data['PAT_ENC_CSN_ID'].values)
m, k = len(data), len(pIndex)

data.to_hdf(dataloc, 'data')

##############################################################################################
#data = pd.read_hdf(loc, 'data')
#varGroups = load(loc = varGroupsfolder, name = 'varGroups')

## add new location vars and remove ones we decided to drop in earlier part of code
keepward = "emergencyroom, icu, micu, sicu, ccu, cticu".split(sep = ', ')
medicalward = "genmed, neuro, hemeonc, cardiology".split(sep = ', ')
surgicalward = "periop, transplant, surgonc, maternity, surgery".split(sep = ', ')
otherward = list(set(varGroups['loc']) - set(keepward + medicalward + surgicalward))

for name, varList in zip(['medicalward', 'surgicalward', 'otherward'],
                         [medicalward, surgicalward, otherward]):
    col = np.any(data[varList].values == 1, axis = 1)*1.0
    col[~np.any(np.isfinite(data[varGroups['loc']].values), axis = 1)] = np.nan
    data.insert(loc = data.columns.get_loc('icu'), column = name, value = col)


dropVars = load(loc = varGroupsfolder, name = 'dropVars')
varGroups['loc'] = list(set(varGroups['loc']) - set(dropVars['loc'])) + \
                            ['medicalward', 'surgicalward', 'otherward']
varGroups['loc'] = list(set(varGroups['loc'] + keepward))
dropVars['loc'] = list(set(dropVars['loc']) - set(varGroups['loc']))
data.drop(columns = dropVars['loc'], inplace = True)


## carry forward imputation for location vars
colidx = data.columns.get_indexer(varGroups['loc'])
oldLocCols = data.iloc[:,colidx].values.astype(np.float16)
newLocCols = np.zeros((len(data), len(varGroups['loc'])), dtype = np.bool)
for i in tqdm(range(k)):
    start, stop, length = pIndex[i,:]
    lastRecord = oldLocCols[start,:] == 1
    for j in range(1,length):
        current = oldLocCols[start+j,:]
        if np.any(np.isfinite(current)): #set this row to the new set of imputation values
            lastRecord = current == 1
        newLocCols[start+j,:] = lastRecord
        
data[varGroups['loc']] = newLocCols

cols = varGroups['med'] + varGroups['pro']
data[cols] = data[cols].values.astype(np.bool)

dataloc = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'Data.h5')
save(obj = varGroups, loc = varGroupsfolder, name = 'varGroups')
data.to_hdf(dataloc, 'data')

###################################################################################################
dataloc = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'Data.h5')
data = pd.read_hdf(dataloc, 'data')
varGroupsfolder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data')
varGroups = load(loc = varGroupsfolder, name = 'varGroups')
pIndex = getPatientIndices(data['PAT_ENC_CSN_ID'].values)
k = len(pIndex)

# Formatting times:
#     1. Convert times from EPIC/SAS format to np.datetime64 arrays
#     2. Correct sorting for chronological order within each encounter (there are randos missorted)
#     3. Convert time to hours passed from first timestamp with respect to each encounter (los)

## check encounter id 92971224, 97713018, 95096582, 97175134, 95672277 for some examples
## first time recorded is very off - like 13 years off
## all have 00:00:00 for time of day in first record
## i = 4685 for first id
## proposed solution: none of these people have a location recorded during these times, so
## delete all starting rows w/o a location

select = np.zeros(len(data), dtype = np.bool)


time = copy(data['time'].values).astype(np.unicode)
timeFunc = np.vectorize(lambda x: x.replace(' ', 'T')[:-3])
time = timeFunc(time).astype(np.datetime64)

# sorting (must sort original data too)
print('Sorting patient encounter data by chronological order')
newIndex = np.zeros(m, dtype = np.int64)
for i in tqdm(range(k)):
    start, stop, _ = pIndex[i,:]
    indices = np.argsort(time[pIndex[i,0]:pIndex[i,1]]) + start
    newIndex[start:stop] = indices
    
time = time[newIndex]
data = data.iloc[newIndex,:]

# convert times to hours
print('Converting timestamps to length of stay in hours')
newTime = np.zeros(m, dtype = np.float32)
for i in tqdm(range(k)):
    start, stop, length = pIndex[i,:]
    timeZero = time[start]
    for j in range(1,length):
        newTime[start+j] = (time[start+j] - timeZero)/np.timedelta64(1, 'h')
        if newTime[start+j] > 50000:
            raise ValueError
    
# set new time var (length of stay in hours)
data['time'] = time
data.insert(loc = data.columns.get_loc('time'), column = 'los', value = newTime)
varGroups['inf'] = varGroups['inf'] + ['los']
save(obj = varGroups, loc = varGroupsfolder, name = 'varGroups')
data['PAT_MRN_ID'] = data['PAT_MRN_ID'].values.astype(np.unicode)
data['RACE'] = data['RACE'].values.astype(np.unicode)

dataloc = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'Data.h5')
data.to_hdf(dataloc, 'data')


##############################################################################################
#### AT THIS POINT, THIS IS A P GOOD DATASET TO USE FOR MANY PURPOSES
#### FROM HERE ON, THE DATASET WILL MOVE TOWARDS AKITOMORROW
##############################################################################################


# Creating aki markers and targets (28 cols):
#     1a. Find minimum creatinine in past 2 & 7 days (2 cols)
#     1b. Calculate AKI stage 1 & 2 markers at each time point based on info from 1a (2 cols)
#     2. Find maximum creatinine, presence of stage 1, and presence of stage 2 
#            at 6, 12, 24, and 48 hours using segmented and overlapping windows (3*4*2=24 cols)
#    
# Notes:
#     small margin of time is added to windows for discrepencies in daily measurement times
#     NaNs will be present when there is no information to go on


## load data if necessary
dataloc = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'Data.h5')
data = pd.read_hdf(dataloc, 'data')
varGroupsfolder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data')
varGroups = load(loc = varGroupsfolder, name = 'varGroups')
pIndex = getPatientIndices(data['PAT_ENC_CSN_ID'].values)
m, k = len(data), len(pIndex)

# finding min creats
print('Finding minimum recorded creatinine in past 2 and 7 days')
minCreats = np.empty((m, 2), dtype = np.float16) #cols: min from past 2 and 7 days, respectively
minCreats.fill(np.nan)
creatIndex = data.columns.get_loc('creatinine')
for i in tqdm(range(k)):
    start, stop, length = pIndex[i,:]
    minCreats[start,:] = data.iloc[start,creatIndex]
    encounter = data.iloc[start:stop,:]
    for j in range(1,length):           # this can be done more efficienctly
        ind2 = search(encounter, 'creatinine', timeName = 'los', reference = j, 
                      window = 2*24+4, direction = 'backward')
        ind7 = search(encounter, 'creatinine', timeName = 'los', reference = j, 
                      window = 7*24+4, direction = 'backward')
        min2 = np.min(encounter.loc[ind2, 'creatinine'].values) if np.any(ind2) else np.nan
        min7 = np.min(encounter.loc[ind7, 'creatinine'].values) if np.any(ind7) else np.nan
        minCreats[start+j,:] = [min2, min7]


# recording aki markers
akiNow = np.empty((m, 2), dtype = np.float16)  # cols: stage 1 and 2, respectively
akiNow.fill(np.nan)
finiteMins, finiteCreats = np.isfinite(minCreats), np.isfinite(data['creatinine'].values)
for i in tqdm(range(k)):
    start, stop, length = pIndex[i,:]
    
    for j in range(1,length):
        if finiteCreats[start+j]: # check if current timestamp has creat recorded
            
            # find absolute and percentage changes
            absChange2, percentChange7 = np.nan, np.nan
            if finiteMins[start+j,0]: # exists a min creat from past 2 days
                absChange2 = data.iloc[start+j,creatIndex] - minCreats[start+j,0]
            if finiteMins[start+j,1]: # exists a min creat from past 7 days
                percentChange7 = (data.iloc[start+j,creatIndex] - minCreats[start+j,1])/minCreats[start+j,1]
                
            # record results using thresholds
            akiNow[start+j,0] = percentChange7 >= 0.5 or absChange2 >= 0.3
            akiNow[start+j,1] = percentChange7 >= 1
            
            
# sanity check for aki stage 1, stage 2 encounter event rates                       
counter0, counter1 = 0, 0
for i in tqdm(range(k)):
    start, stop, _ = pIndex[i,:]
    enoughCreats = np.sum(finiteCreats[start:stop]) > 1
    if np.any(akiNow[start:stop,0]==1) and enoughCreats:
        counter0 += 1
    if np.any(akiNow[start:stop,1]==1) and enoughCreats:
        counter1 += 1

print('\n', np.round(100*counter0/k, 1), '% of encounters have presence of Stage 1 AKI')
print(np.round(100*counter1/k, 1), '% of encounters have presence of Stage 2 AKI')




# merge mincreat and akinow to full dataset
data['AKI1'] = akiNow[:,0]
data['AKI2'] = akiNow[:,1]
data.insert(loc = data.columns.get_loc('creatinine'), column = 'minCreat48',
            value = minCreats[:,0])
data.insert(loc = data.columns.get_loc('creatinine'), column = 'minCreat7',
            value = minCreats[:,1])

varGroups['lab'] = varGroups['lab'] + ['minCreat48', 'minCreat7', 'AKI1', 'AKI2']
save(obj = varGroups, loc = varGroupsfolder, name = 'varGroups2')



# create targets (4 cols for each outcome group (3) corresponding to 6, 12, 24, and 48 hrs)
# example: segmented and overlap would correspond to windows of [6-12] and [0-12] hrs, respectively
outcomes_names = ['AKI1', 'AKI2', 'creatinine']
outcomes_overlap = [np.empty((m,4), dtype = np.float16) for i in range(3)]
outcomes_segment = [np.empty((m,4), dtype = np.float16) for i in range(3)]
timeTillAKI = np.empty((m,2), dtype = np.float16)
timeTillAKI.fill(np.nan)
for i in range(3):
    outcomes_overlap[i].fill(np.nan)
    outcomes_segment[i].fill(np.nan)

# create normal outcomes
margin = 0.5
hourListPre = [6, 12, 24, 48]
hourList = [hourListPre[i] + margin*(hourListPre[i]/hourListPre[0]) for i in range(4)]
offsetList = [0, 6, 12, 24] # note, start times for offset are normal (e.g. 24-52)
for i in tqdm(range(k)):
    start, stop, length = pIndex[i,:]
    encounter = data.iloc[start:stop,:]
    for j in range(length):
        for h in range(len(hourList)):
            # checking stage1 indices suffices for all 3
            ind_overlap = search(encounter, outcomes_names[0], timeName = 'los', reference = j, 
                                 window = hourList[h], offset = 0, direction = 'forward')
            ind_segment = search(encounter, outcomes_names[0], timeName = 'los', reference = j, 
                                 window = hourList[h], offset = offsetList[h], direction = 'forward')
            if np.any(ind_overlap):
                outcomes_overlap[0][start+j,h] = np.any(encounter.loc[ind_overlap, outcomes_names[0]])
                outcomes_overlap[1][start+j,h] = np.any(encounter.loc[ind_overlap, outcomes_names[1]])
                outcomes_overlap[2][start+j,h] = np.max(encounter.loc[ind_overlap, outcomes_names[2]])
            if np.any(ind_segment):
                outcomes_segment[0][start+j,h] = np.any(encounter.loc[ind_segment, outcomes_names[0]])
                outcomes_segment[1][start+j,h] = np.any(encounter.loc[ind_segment, outcomes_names[1]])
                outcomes_segment[2][start+j,h] = np.max(encounter.loc[ind_segment, outcomes_names[2]])

# create time till AKI outcomes
for i in tqdm(range(k)):
    start, stop, length = pIndex[i,:]
    encounter = data.iloc[start:stop,:]
    timesCurrent = encounter['los'].values
    AKIbool = encounter[['AKI1', 'AKI2']].values == 1
    if np.any(AKIbool[:,0]):
        idx0 = np.where(AKIbool[:,0])[0][0]
        timeTillAKI[start:stop,0] = timesCurrent[idx0] - timesCurrent
    if np.any(AKIbool[:,1]):
        idx1 = np.where(AKIbool[:,1])[0][0]
        timeTillAKI[start:stop,1] = timesCurrent[idx1] - timesCurrent
    
    

# merge targets to full dataset
varGroups['fut'] = []
for i, hour in enumerate(hourListPre):
    for j in range(len(outcomes_names)):
        string = 'future' + str(hour) + '_' + outcomes_names[j] + '_'
        data[string + 'overlap'] = outcomes_overlap[j][:,i]
        data[string + 'segment'] = outcomes_segment[j][:,i]
        varGroups['fut'] = varGroups['fut'] + [string + 'overlap', string + 'segment']

data['timeTillAKI1'] = timeTillAKI[:,0]
data['timeTillAKI2'] = timeTillAKI[:,1]

varGroups['fut'] += ['timeTillAKI1', 'timeTillAKI2']

# add two more vars: creat percent and first recorded creat
data.insert(loc = data.columns.get_loc('creatinine'), column = 'creatPercent',
            value = (data['creatinine'] - data['minCreat7'])/data['minCreat7'])

creatFirst = np.empty(len(data), dtype = np.float32)
creatFirst.fill(np.nan)
creatValues = data['creatinine'].values
for i in tqdm(range(len(pIndex))):
    start, stop, length = pIndex[i,:]
    subCreat = creatValues[start:stop]
    recorded = np.isfinite(subCreat)
    creatsCurrent = subCreat[recorded]
    if len(creatsCurrent) > 0:
        val = creatsCurrent[0]
        idx = np.where(val == subCreat)[0][0]
        creatFirst[start+idx:stop] = val

data.insert(loc = data.columns.get_loc('creatinine'), column = 'creatFirst', value = creatFirst)

varGroups['lab'] = varGroups['lab'] + ['creatPercent', 'creatFirst']



# save
save(obj = varGroups, loc = varGroupsfolder, name = 'varGroups2')
dataloc = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'AKI_base.h5')
data.to_hdf(dataloc, 'data')







