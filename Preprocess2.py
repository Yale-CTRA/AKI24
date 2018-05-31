import pandas as pd
import numpy as np
from copy import deepcopy as copy
from tqdm import tqdm

import os
import sys

usingInterpreter = True
if usingInterpreter:
    root = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects')
else:
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path.append(root)

from Helper.clean import compress, keepNumerical, mixed2float
from Helper.preprocessing import oneHot, convertISO8601, stringCollapse


def readData(file_name, chunkSize = 1e6, numChunks = None):
    """
    reads data in chunks and combines it
    option of only partially reading in the file using numChunks arg w/ small chunkSize
    """
    csv_iter = pd.read_csv(file_name, chunksize = chunkSize, iterator = True)
    data = None
    if numChunks == 1:
        for idx, chunk in enumerate(csv_iter):
            if idx == 0:
                data = chunk
            else:
                break
    else:
        data = []
        for idx, chunk in enumerate(csv_iter):
            data.append(chunk)
            if numChunks is not None:
                if idx+1 == numChunks:
                    break
        data = pd.concat(data, axis = 0, copy = False)
    return data


def getPatientIndices(data):
    """
    Input: 1-d numpy array of patient encounter IDs that is total length of data (repeats necessary)
    Output: (k x 3) matrix, X, where 1st and 2nd columns are start and stop indices, 
            third column is number of timestamps, and k is num encounters
    Usage for jth patient: data.iloc[ X[j,0]:X[j,1] , : ]
    """
    m = len(data)
    X = np.zeros((len(np.unique(data)), 3), dtype = np.int64)
    
    #fix boundaries
    X[0,0] = 0
    X[-1,1] = m

    # do loop to record indices
    counter = 0
    current = data[0]
    for i in range(m):
        if data[i] != current:
            current = data[i]
            X[counter,1] = i
            X[counter+1,0] = i
            counter += 1
    X[:,2] = X[:,1] - X[:,0]
    return X

def move(data, var, idx):
    """
    Moves a var from its position to a new specified index and returns dataframe
    Probably a more computationally efficient way to do this
    """
    _ = data[var]
    data.drop(labels = var, axis = 1, inplace = True)
    data.insert(idx, var, _)
    return data

###################################################################################################
###################################################################################################

# load data and create patient encounter indexer (pIndex)
dataLoc = os.path.join('E:', os.sep, 'sas library', 'final.csv')
data = readData(dataLoc, chunkSize = 1e6, numChunks = 1)
data.set_index('PAT_ENC_CSN_ID', inplace = True)

pIndex = getPatientIndices(data.index)
m, k = len(data), len(pIndex)

# drop vars we know are useless from the getgo
# speechaswall and transfu included cuz very low recording rate and are the only ones
#       in the procedures category with mixed types from shitty data entry
dropCols = ['PAT_NAME', 'BIRTH_DATE']
data.drop(labels = dropCols, axis = 1, inplace = True)


# move location of creatinine and urine_output to be with the rest of labs
idx = data.columns.get_loc('baseex')
data = move(data, 'urine_output', idx)
data = move(data, 'creatinine', idx)


###################################################################################################
############# CLEAN VARS
###################################################################################################
# variables that might need fixing: list(data.columns[data.dtypes == np.dtype('O')])
# find unique: set(data.loc[~(data[var].astype(np.str) == 'nan'), var])

## fix demographic vars
data.rename(columns={'AGE_AT_ENCOUNTER':'AGE', 'ETHNICITY': 'HISPANIC', 
                     'SEX': 'MALE'}, inplace=True)

ethnicity = np.empty(m, dtype = np.float16)
ethnicity.fill(np.nan)
ethnicity[data['HISPANIC'].values == 'Hispanic or Latino'] = 1
ethnicity[data['HISPANIC'].values == 'Non-Hispanic'] = 0
data['HISPANIC'] = ethnicity

collapseList = ['Other/Not Listed', 'Unknown']
data['RACE'] = stringCollapse(data['RACE'].values, collapseList, 'Other')
data.loc[data['RACE']=='Patient Refused', 'RACE'] = np.nan

male = np.empty(m, dtype = np.float16)
male.fill(np.nan)
male[data['MALE'] == 'M'] = 1
male[data['MALE'] == 'F'] = 0
data['MALE'] = male


## convert time vars
deathDate = data['DEATH_DATE'].apply(lambda x: str(x).replace(' ', 'T')).values
deathDate[deathDate == 'nan'] = np.nan
data['DEATH_DATE'] = deathDate
data['time'] = convertISO8601(data['time'].values)

# sorting patient encounters
print('Sorting patient encounter data by chronological order')
newIndex = np.zeros(m, dtype = np.int64)
for i in tqdm(range(k)):
    start, stop, _ = pIndex[i,:]
    indices = np.argsort(data['time'].values[pIndex[i,0]:pIndex[i,1]]) + start
    newIndex[start:stop] = indices
data = data.iloc[newIndex,:]


### fix vars that can be processed through mixed2float
data.rename(columns={'rbcmorph':'rbcnormal'}, inplace=True)
rbcnormal = copy(data['rbcnormal'].values)
rbcnormal = stringCollapse(rbcnormal, ['Normal'], 0, inverse = True)
rbcnormal[rbcnormal == 'NORMAL'] = 1
data['rbcnormal'] = rbcnormal.astype(np.float16)

selectedCols = data.loc[:,'baseex':'ethanol'].columns
selectedCols = list(selectedCols[[data[var].dtype == np.dtype('O') for var in selectedCols]])
for var in tqdm(selectedCols):
    data[var] = mixed2float(data[var].values)
    
    
## fix urine vars
    

###################################################################################################
############# ADD VARS
###################################################################################################
## pulse pressure, mean arterial pressure, body mass index, length of stay
## bun/creatinine ratio, anion gap, protein gap





