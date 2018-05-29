
import pandas as pd
import numpy as np
from copy import deepcopy as copy
from tqdm import tqdm

import os
import sys
root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
## for use when in the interpreter
#root = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects')
sys.path.append(root)

from Helper.clean import compress, keepNumerical, mixed2float
from Helper.preprocessing import oneHot, convertISO8601


def readData(file_name, chunkSize = 1e6, numChunks = None):
    """
    reads data in chunks and combines it
    option of only partially reading in the file using numChunks arg w/ small chunkSize
    """

    csv_iter = pd.read_csv(file_name, chunksize = chunkSize, iterator = True)
    data = None
    if numChunks == 1:
        for _, chunk in enumerate(csv_iter):
            data = chunk
    else:
        data = []
        for idx, chunk in enumerate(csv_iter):
            data.append(chunk)
            if numChunks is not None:
                if idx+1 == numChunks:
                    break
        data = pd.concat(data, axis = 0, copy = False)
    return data


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
        print("Please choose either 'forward' or 'backward' for direction parameter")
        select = np.zeros(len(vals), dtype = np.bool)
    return select


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

##############################################################################################################################
##############################################################################################################################

# load data and create patient encounter indexer (pIndex)
dataLoc = os.path.join('E:', os.sep, 'sas library', 'final.csv')
data = readData(dataLoc, chunkSize = 2e6, numChunks = 1)
data.set_index('PAT_ENC_CSN_ID', inplace = True)

pIndex = getPatientIndices(data.index)
m, k = len(data), len(pIndex)


# drop vars we know are useless from the getgo
# speechaswall and transfu included cuz very low recording rate and are the only ones
#       in the procedures category with mixed types from shitty data entry
dropCols = ['PAT_NAME', 'BIRTH_DATE', 'DEATH_DATE', 'DISCH_DISP_C', 'DISCHARGE_DISPSN',
            'PAT_STATUS', 'TOT_GRP', 'speechswall', 'transfu'] + ['ELX_GRP_' + str(i+1) for i in range(31)]
data.drop(labels = dropCols, axis = 1, inplace = True)


# move location of creatinine and urine_output to be with the rest of labs
idx = data.columns.get_loc('baseex')
data = move(data, 'urine_output', idx)
data = move(data, 'creatinine', idx)

# create clinically meaningful variable groups
boundaries = {'loc': ['emergencyroom','pacu'], 'lab': ['creatinine','arterial_diastolic'],
              'med': ['lorazepam','immunomodulators'], 'pro': ['telemetry','hospice']}
varGroups, _ = group(data, boundaries)


# fill procedure variables with 0s if nan
# turn meds into measured / not measured
data.loc[:,varGroups['pro']] = data.loc[:,varGroups['pro']].fillna(0)
data.loc[:,varGroups['med']] = np.isfinite(data.loc[:,varGroups['med']].values)


# group location variables together in clinically meaningful ways to reduce sparseness



# discover the proportions of patient encounters at which variables are measured at least once
print('Measuring variable recording rates')
measured = {key: np.zeros((k, len(value))) for key, value in varGroups.items()}
countFuncs = {'lab': lambda x: x.count() > 0, 
              'loc': lambda x: np.any(x.values == 1, axis = 0),
              'med': lambda x: np.sum(x.values, axis = 0) > 0, 
              'pro': lambda x: np.sum(x.values, axis = 0) > 0}
for i in tqdm(range(k)):
    start, stop, _ = pIndex[i,:]
    encounter = data.iloc[start:stop]
    for key, value in varGroups.items():
        measured[key][i,:] = countFuncs[key](encounter[value])
   
proportions = {}     
for key, value in varGroups.items():
    proportions[key] = pd.Series(np.sum(measured[key], axis = 0)/k, index = value)


# figure out which variables to drop
# labs less than 10%
# meds, locations, and procedures less than 5%
thresholds = {'lab': 0.1, 'loc': 0.05, 'med': 0.05, 'pro': 0.05}
dropVars = {}
for key, value in varGroups.items():
    dropVars[key] = list(np.array(value)[proportions[key].values < thresholds[key]])


# actually drop
# fix varGroup to reflect deletions
flatten = lambda l: [item for sublist in l for item in sublist]
data.drop(labels = flatten(dropVars.values()), axis = 1, inplace = True)
    
_ = {}
for key, value in varGroups.items():
    _[key] = list(set(value) - set(dropVars[key]))
varGroups = _
    

###################################################################################################

# Formatting times:
#     1. Convert times from EPIC/SAS format to np.datetime64 arrays
#     2. Correct sorting for chronological order within each encounter (there are randos missorted)
#     3. Convert time to hours passed from first timestamp with respect to each encounter (los)

# to np.datetime64
time = convertISO8601(copy(data['time'].values))

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
newTime = np.zeros(m, dtype = np.float16)
for i in tqdm(range(k)):
    start, stop, length = pIndex[i,:]
    timeZero = time[start]
    for j in range(1,length):
        newTime[start+j] = (time[start+j] - timeZero)/np.timedelta64(1, 'h')
    
# set new time var (length of stay in hours)
data.loc[:,'los'] = newTime


# Creating aki markers and targets (28 cols):
#     1a. Find minimum creatinine in past 2 & 7 days (2 cols)
#     1b. Calculate AKI stage 1 & 2 markers at each time point based on info from 1a (2 cols)
#     2. Find maximum creatinine, presence of stage 1, and presence of stage 2 
#            at 6, 12, 24, and 48 hours using segmented and overlapping windows (3*4*2=24 cols)
#    
# Notes:
#     small margin of time is added to windows for discrepencies in daily measurement times
#     NaNs will be present when there is no information to go on


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
data['minCreat48'] = minCreats[:,0]
data['minCreat7'] = minCreats[:,1]


# create targets (4 cols for each outcome group (3) corresponding to 6, 12, 24, and 28 hrs)
# example: segmented and overlap would correspond to windows of [6-12] and [0-12] hrs, respectively
outcomes_names = ['AKI1', 'AKI2', 'creatinine']
outcomes_overlap = [np.empty((m,4), dtype = np.float16) for i in range(3)]
outcomes_segment = [np.empty((m,4), dtype = np.float16) for i in range(3)]
for i in range(3):
    outcomes_overlap[i].fill(np.nan)
    outcomes_segment[i].fill(np.nan)

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
                                 window = hourList[l], offset = 0, direction = 'forward')
            ind_segment = search(encounter, outcomes_names[0], timeName = 'los', reference = j, 
                                 window = hourList[l], offset = offsetList[l], direction = 'forward')
            if np.any(ind_overlap):
                outcomes_overlap[0][start+j,h] = np.any(encounter.loc[ind_overlap, outcomes_names[0]])
                outcomes_overlap[1][start+j,h] = np.any(encounter.loc[ind_overlap, outcomes_names[1]])
                outcomes_overlap[2][start+j,h] = np.max(encounter.loc[ind_overlap, outcomes_names[2]])
            if np.any(ind_segment):
                outcomes_segment[0][start+j,h] = np.any(encounter.loc[ind_segment, outcomes_names[0]])
                outcomes_segment[1][start+j,h] = np.any(encounter.loc[ind_segment, outcomes_names[1]])
                outcomes_segment[2][start+j,h] = np.max(encounter.loc[ind_segment, outcomes_names[2]])
    


# merge targets to full dataset
for i, hour in enumerate(hourListPre):
    for j in range(len(outcome_names)):
        string = 'future' + str(hour) + '_' + outcome_names[j] + '_'
        data[string + 'overlap'] = outcomes_overlap[j][:,i]
        data[string + 'segment'] = outcoms_segment[j][:,i]


## finished making targets
###############################################################################################
## start fixing mixed types from shitty data entry!! yay! /s
## everything from here below is barely currently worked on / old, irrelevant code












def unique(data):
    select = ~(data.astype(np.str) == 'nan')
    return list(set(data[select]))


## delete columns
dropCols = ['PAT_NAME', 'BIRTH_DATE', 'DEATH_DATE', 'DISCH_DISP_C', 'DISCHARGE_DISPSN',
            'PAT_STATUS', 'TOT_GRP'] + ['ELX_GRP_' + str(i+1) for i in range(31)]




mixedRegLabs = ['baseex', 'co2', 'hco3poc', 'o2sat', 'pco2', 'troponin',
               'ptt', 'ast', 'bilidirect', 'bilitotal', 'buncreatratio', 'magnesium',
               'phosphorous', 'potassium', 'abslymphcount', 'basophils', 'eos',
               'lymphabsdiff', 'monocyteabs', 'neutrophils', 'rbcmorph']

mixedWeirdLabs = ['rbcmorph', 'ethanol']
mixedUrineLabs = ['uabili', 'uaclarity', 'uacolor', 'uaglucose', 'uaketones', 'ualeukest',
                  'uanitrite', 'uaprotein', 'uahycasts', 'uarbcs', 'uawbcs]

mixedCols = [10,11,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,38,39,40,41,42,44,47,
              48,50,51,52,53,55,56,57,58,60,62,63,64,65,66,67,68,69,70,71,73,74,75,76,77,78,79,80,
              81,87,88,89,90,91,96,99,100,103,105,106,107,109,110,111,112,113,114,115,120,123,124,
              125,126,133,134,135,136,138,139,140,141,142,144,147,148,150,151,153,154,155,156,157,
              160,162,163,164,165,169,170,171,173,174,175,176,178,179,180,181,182,186,191,192,193,
              194,195,196,203,205,206,207,208,209,210,212,215,664,672]


temp = data['time'].values


















start = time.time()
temp = convertISO8601(copy(data['time'].values))
end = time.time()
temp = temp.astype(np.datetime64)






def build_indexer(m, train_per):
    train_index = np.repeat(False, m)
    train_index[:int(round(m*train_per))] = True
    return train_index
    
    
    
def standardize(data, train_index, exclude = []):
    vars_to_standardize = list(set(data.columns) - set(data.columns[data.dtypes == np.bool]) - set(exclude))
    scaler = StandardScaler()
    data.loc[train_index, vars_to_standardize] = scaler.fit_transform(data.loc[train_index, vars_to_standardize])
    data.loc[~train_index, vars_to_standardize] = scaler.transform(data.loc[~train_index, vars_to_standardize])
    return data


class TimeSeriesContainer(object):
    def __init__(self, data, train_index, info, batchSize = 32):
        super().__init__()
        self.id, self.predictors, self.target = info
        self.train_index = train_index
        self.batchSize = batchSize
        self.train_len = sum(train_index)
        self.eval_len = len(train_index) - self.train_len
        
        self.data = data.set_index(self.id, drop = True, inplace = False)
        self.train = self.__toPanel(self.data.ix[train_index,:], 'Training Set')
        self.eval = self.__toPanel(self.data.ix[~train_index,:], 'Evaluation Set')
        self.current = self.train
        
        self.index = 0
        self.stop = False
    
    
    # magic methods
    def __len__(self):
        if self.current == self.train:
            return self.train_len
        else:
            return self.eval_len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        end = self.index + self.batchSize
        if self.stop:
            self.index = 0
            self.stop = False
            raise StopIteration
        elif end >= len(self.current):
            self.stop = True
            batch_X = [x[self.predictors].values.astype('float32') for x in self.current[self.index:]]
            batch_Y = [y[self.target].values.astype('uint8') for y in self.current[self.index:]]
            return (batch_X, batch_Y)
        else:
            batch_X = [x[self.predictors].values.astype('float32') for x in self.current[self.index:end]]
            batch_Y = [y[self.target].values.astype('uint8') for y in self.current[self.index:end]]
            self.index += self.batchSize
            return (batch_X, batch_Y)
    
    ## mode switching for iteration
    def train_mode(self):
        self.current = self.train
    def eval_mode(self):
        self.current = self.eval
    
    # utility functions
    def shuffle(self):
        permutation = list(range(len(self.train)))
        random.shuffle(permutation)
        self.train = [self.train[i] for i in permutation]
  
    def set_batchSize(self, batchSize):
        self.batchSize = batchSize
    def set_predictors(self, predictors):
        self.predictors = predictors
    def set_target(self, target):
        self.target = target
    
    # internal function
    # parses data into feature and target lists where every item is a particular patient's time data
    def __toPanel(self, data, label):
        IDs = data.index.values
        slice_list = [None]*len(np.unique(IDs))
        last, counter = 0, 0
        for i in range(1, len(data)):
            if IDs[i-1] != IDs[i]:
                slice_list[counter] = slice(last,i)
                counter += 1
                last = i
        slice_list[-1] = slice(last, len(IDs))
        panel = [data.iloc[slice_list[s],:] for s in range(len(slice_list))]
        print('Panel creation completed for ' + label + '.')
        return panel
    

    

def findFirstRows(IDs):
    m = len(IDs)
    firstIndexer = np.repeat(False, m)
    
    lastSeen = IDs[0]
    for i in range(m):
        current = IDs[i]
        if lastSeen != current:
            firstIndexer[i-1] = True
        lastSeen = current
    firstIndexer[-1] = True
    
    return firstIndexer
    
        
    
firstIndexer = findFirstRows(data['pat_enc_csn_id'].values)
data.set_index('pat_enc_csn_id', inplace = True)
data = data.loc[:,[np.isfinite(value) for value in list(data.iloc[0,:])]]
data = data.loc[firstIndexer,:]





varDict

futureVars = ['akiinndext24', 'akiinnext48', 'countcreat48', 'dcadmitted', 'dcama', 'dcdiedinhouse', 'dchome', 'dchomehospice',
              'dchosptransfer', 'dcinpthospice', 'dcother', 'dcplannedreadmit', 'dctosnf']
temporalVars = ['bicarbonate', 'bun', 'bunbl','chloride', 'creatinine', 'creatinebl', ]
staticVars = set(data.columns.values) - set(futureVars + temporalVars)


def main():
    
    os.chdir('/home/aditya/Projects/AKI Time Series/Data/')
    data_file_name = 'aki tomorrow all data.dta'
    store_file_name = 'aki tomorrow all data preprocessed.pickle'
    
    ID = 'pat_enc_csn_id'
    target = 'akiinnext24'
    train_per = 0.7
    data = load(data_file_name, 2e7, 10)
    print("Finished loading and cleaning data.")
    
    
    
    
    keep = data.columns[:np.where(data.columns.values == 'training')[0][0]].tolist() + [target]
    keep.remove('dcdiedinhouse')
    data = data.loc[:,keep]
    predictors = list(set(data.columns.tolist()) - set([ID, target]))

    train_index = build_indexer(len(data), train_per)
    # data was prestandardized
    #data = standardize(data, train_index, exclude = [ID])
    #print("Finished standardizing data.")
    
    panels = TimeSeriesContainer(data, train_index, (ID, predictors, target), batchSize = 32)
    
    store = open(store_file_name, 'wb')
    pickle.dump(panels, store)
    store.close()
    
    print('Processed data stored in:\n' + os.getcwd() + '/' + store_file_name)
        
if __name__ == '__main__':
    main()
    
    
    