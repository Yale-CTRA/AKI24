# Start March 25, 2017
# Last March 27, 2017

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
import pickle

def load(file_name, num_rows = 1e6, num_chunks = 10, columns = None):
    assert num_rows/num_chunks > 100
    chunksize = int(num_rows/num_chunks)
    stata_iter = pd.read_stata(file_name, chunksize = chunksize, iterator = True, columns = columns)
    data = [None for i in range(num_chunks)]
    for idx, chunk in enumerate(stata_iter):
        if idx == num_chunks: 
            break
        data[idx] = clean(chunk)
    data = pd.concat(data, axis = 0, copy = False)
    return data


def keep_only_nums(data):
    dtype_set = {np.int8, np.int16, np.int32, np.float32, np.float64, np.bool}
    return data.loc[:,data.dtypes.apply(lambda x: any([issubclass(x.type, dtype) for dtype in dtype_set]))]

def remove_nan_cols(data):
    # only examine first row for nans
    return data.loc[:,data.iloc[0,:].apply(lambda x: not np.isnan(x))]

def clean(data):
    ## REMOVE BAD COLS
    data = remove_nan_cols(keep_only_nums(data))
    
    ## COMPRESS DTYPES
    # examine only the first 100 rows
    new_dtypes = data.dtypes.copy()
    new_dtypes[data.iloc[:10000,:].apply(lambda x: len(np.unique(x)) <= 2, axis = 0)] = np.dtype('bool')
    new_dtypes[new_dtypes.apply(lambda x: issubclass(x.type, np.float64))]= np.dtype('float32')
    data = data.astype(new_dtypes.to_dict(), copy = True)
    return data
    

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
    
    
    