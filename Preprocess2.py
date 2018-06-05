import pandas as pd
import numpy as np
from copy import deepcopy as copy
from tqdm import tqdm
import re

import os
import sys

usingInterpreter = True
if usingInterpreter:
    root = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects')
else:
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
sys.path.append(root)

from Helper.clean import compress, keepNumerical, mixed2float, mixedCategoricalClean
from Helper.preprocessing import oneHot, convertISO8601, stringCollapse
from Helper.utilities import saveUnique


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
dataLoc = os.path.join('G:', os.sep, 'sas library', 'final.csv')
#data = readData(dataLoc, chunkSize = 2e6, numChunks = 1)
data = pd.read_csv(dataLoc)
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


# save unique values and counts for all object vars for error checking later
folder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data')
names = list(data.columns[data.dtypes == np.dtype('O')])
saveUnique(data, names, folder, 'fullData')

###################################################################################################
############# CLEAN VARS
###################################################################################################
# variables that might need fixing: list(data.columns[data.dtypes == np.dtype('O')])
# find unique: set(data.loc[~(data[var].astype(np.str) == 'nan'), var])

#############################
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

######################################################################
### fix vars that can be processed through mixed2float
## albumin and cholesterolldl and rbcmorph must be done separately

data.rename(columns={'rbcmorph':'rbcnormal'}, inplace=True)
rbcnormal = copy(data['rbcnormal'].values)
rbcnormal = stringCollapse(rbcnormal, ['Normal'], 0, inverse = True)
rbcnormal[rbcnormal == 'NORMAL'] = 1
data['rbcnormal'] = rbcnormal.astype(np.float16)

def onlyNums(x, regEx):
   for i in range(len(x)):
       if isinstance(x[i], str):
           num = re.findall(regEx, str(x[i]))
           if num[0] == '<' or num[0] == '>':
               x[i] = np.nan
   return x

data['albumin'] = onlyNums(data['albumin'].values, '<')
data['cholesterolldl'] = onlyNums(data['cholesterolldl'].values, '>|<')


selectedColumns = data.loc[:,'baseex':'ethanol'].columns
selectedColumns = list(selectedColumns[[data[var].dtype == np.dtype('O') for var in selectedColumns]])

    
selectedColumns += ['cktotal', 'ecgp', 'ecgpr', 'bicarbonate', 'caionplasma', 'bnp', 'fio2',
                   'cmvquant', 'serinepr3', 'gbm', 'thrombintime', 'cyclosporin', 'pcotart',
                   'pthmm', 'rapamycin', 'urinecalrandom', 'bkquant', 'albumin', 'cholesterolldl']
def accumulateRanges(data, pairs):
    cols = []
    for pair in pairs:
        cols += list(data.loc[:,pair[0]:pair[1]].columns)
pairs = [('globulin','platelet'),
         ('vancomycinlevel','vitamind25'),
         ('hscrp','vancorandom'),
         ('uricacid','kur'),
         ('c4complement', 'vpo2'),
         ('gentamicin', 'tacrolevel'),
         ('uunrandom', 'pthintact'),
         ('albuminurine', 'valproiclevel'),
         ('lithiumlevel', 'rf')]    
selectedColumns += accumulateRanges(data, pairs)
    

for var in tqdm(selectedColumns):
    data[var] = mixed2float(data[var].values)
    
    
################################################################################
## fix ordinal vars + remaining urine vars (will throw one warning: invalid value in greater)
    
def fixOrdinalVars(data):
    
    m = len(data)
    # basic conversion converts any of the common namings to our categories
    # throws TypeError if not found, which should be caught and dealth with
    conversionDict = {'none': 'negative', 'neg': 'negative', 
                      'few': 'trace', 'rare': 'trace', 'occasional': 'trace',
                      'small': '1+',  'positive': '1+', 
                      'moderate': '2+', 
                      'large': '3+', '4+': '3+', 'many': '3+', 'marked': '3+'}
    codings = ['negative', 'trace', '1+', '2+', '3+']
    basicConversion = lambda val: val if val in codings else conversionDict[val]
    

    def intervalConversion(val, boundaryMap):
        """
        boundaryMap is a dictionary mapping categories to upper boundary for that interval
        boundaryMap needs to be specified for trace, 1+, and 2+ only
        0 is assumed to map to negative, and no match maps to 3+
        """
        if val == 0:
            newVal = 'negative'
        elif val <= boundaryMap['trace']:
            newVal = 'trace'
        elif val <= boundaryMap['1+']:
            newVal = '1+'
        elif val <= boundaryMap['2+']:
            newVal = '2+'
        else:
            newVal = '3+'
        return newVal
    
    def fullConversion(val, boundaryMap):
        try:
            return basicConversion(col[i])
        except KeyError:
            try:
                val = float(val)
                return intervalConversion(val, boundaryMap)
            except (ValueError, KeyError):
                return np.nan
    
    
    conversionUrineVars= ['uabili', 'uaprotein', 'uaglucose', 'uawbcs', 'uarbcs', 'uahycasts',
                 'uanitrite', 'uaketones', 'ualeukest', 'uablood', 'eosurine',
                 'uawbccclumps', 'uacaoxalate', 'uacapyrophosphate', 'uawbccasts', 'uauricacid']
    # otherUrineVars = ['uaclarity', 'uacolor',  'uaappearance']
    # uaph, uaspecgrav stays as continuous
    # uaspecgrav has mixed units, which needs to be fixed
    
    # uanitrite, ualeukest, and uablood excluded since no numerical values. will throw KeyError
    allBoundaryMaps = {'uabili': {'trace': 0.2, '1+': 1, '2+': 2},
                       'uaprotein': {'trace': 30, '1+': 100, '2+': 300},
                       'uaglucose': {'trace': 30, '1+': 100, '2+': 300},
                       'uawbcs': {'trace': 2, '1+': 10, '2+': 50},
                       'uawbccasts': {'trace': 2, '1+': 10, '2+': 50},
                       'uarbcs': {'trace': 2, '1+': 10, '2+': 50},
                       'uahycasts': {'trace': 2, '1+': 5, '2+': 10},
                       'uaketones': {'trace': 40, '1+': 100, '2+': 200},
                       'uanitrite': {},
                       'ualeukest': {},
                       'uablood': {},
                       'eosurine': {},
                       'uawbcclumps': {},
                       'uacaoxalate': {},
                       'uacapyrophosphate': {},
                       'uauricacid': {}}
    
    cleaningFuncMap = {'uabili': mixedCategoricalClean,
                       'uaprotein': mixedCategoricalClean,
                       'uaglucose': mixedCategoricalClean,
                       'uawbcs': mixed2float,
                       'uawbccasts': mixed2float,
                       'uarbcs': mixed2float,
                       'uahycasts': mixed2float,
                       'uaketones': mixedCategoricalClean,
                       'uanitrite': mixedCategoricalClean,
                       'ualeukest': mixedCategoricalClean,
                       'uablood': mixedCategoricalClean,
                       'eosurine': mixedCategoricalClean,
                       'uawbcclumps': mixedCategoricalClean,
                       'uacaoxalate': mixedCategoricalClean,
                       'uacapyrophosphate': mixedCategoricalClean,
                       'uauricacid': mixedCategoricalClean}
    
    # remove 'Hemolyzed ' and 'Non-' strings first in uablood and ualeukest
    def hemoRemove(val):
        val = str(val)
        if val != 'nan':
            return val.replace('Non-','').replace('Hemolyzed','')
        else:
            return np.nan
        
    if 'uablood' in data.columns:
        data['uablood'] = data['uablood'].apply(hemoRemove)
    if 'ualeukest' in data.columns:
        data['ualeukest'] = data['ualeukest'].apply(hemoRemove)

            

    
    # first do conversionUrineVars
    for var in conversionUrineVars:
        if var in data.columns:
            new = np.empty(m, dtype = data[var].dtype)
            new.fill(np.nan)
            col = cleaningFuncMap[var](data[var].values)
            select = ~(col.astype(np.str) == 'nan')
            for i in range(m):
                if select[i]:
                    try:
                        new[i] = fullConversion(col[i], allBoundaryMaps[var])
                    except KeyError:
                        pass
            data[var] = new
    
    return data
    
    
data = fixOrdinalVars(data)


# now do uaclarity, uaappearance, uacolor    
data.rename(columns={'uaclarity':'uaclaritynormal',
                     'uaappearance':'uaappearancenormal',
                     'uacolor':'uacolornormal'}, inplace=True)


data['uaclarity'] = mixedCategoricalClean(data['uaclarity'].values)
data['uaclarity'] = stringCollapse(data['uaclarity'].values, ['clear'], 1)
data['uaclarity'] = stringCollapse(data['uaclarity'].values, 1, 0, inverse = True)

data['uaappearance'] = mixedCategoricalClean(data['uaappearance'].values)
data['uaappearance'] = stringCollapse(data['uaappearance'].values, ['clear'], 1)
data['uaappearance'] = stringCollapse(data['uaappearance'].values, 1, 0, inverse = True)


data['uacolor'] = mixedCategoricalClean(data['uacolor'].values)
collapseList = ['yellow', 'colorless', 'pale yellow']
data['uacolor'] = stringCollapse(data['uacolor'].values, collapseList, 1)
data['uacolor'] = stringCollapse(data['uacolor'].values, 1, 0, inverse = True)
data['uacolor'] = data['uacolor'].astype(np.float16)


# now do uaph and uaspecgrav
uaspecgrav = mixed2float(data['uaspecgrav'].values)
select = uaspecgrav > 1000
uaspecgrav[select] = uaspecgrav[select]/1000
data['uaspecgrav'] = uaspecgrav

data['uaph'] = mixed2float(data['uaph'].values)



#######################################################################
##### do virus vars and other weird categoricals

def bloodTypes(x):
   select = ~(x.astype(np.str) == 'nan')
   for i in range(len(x)):
       if select[i]:
           blood = re.findall('^\w+', x[i])
           if blood[0] in ['A', 'B', 'AB', 'O']:
               x[i] = blood[0]
           else:
               x[i] = np.nan
   return x

data['abogrouping'] = bloodTypes(data['abogrouping'].values)


# do flurtpcr and fluabantigen
def fluFunc(col):
    func = np.vectorize(lambda x: x.upper() if str(x) != 'nan' else x)
    col = func(col)
    col = stringCollapse(col, ['FLU A DET', 'FLU A DETECTED', 'FLU A POS', 'FLU A POSITIVE'], 
                                'INFLUENZA A')
    col = stringCollapse(col, ['FLU B DET', 'FLU B DETECTED', 'FLU B POS', 'FLU B POSITIVE'], 
                                 'INFLUENZA B')
    col = stringCollapse(col, ['NOT DETECTED', 'INFLUENZA A', 'INFLUENZA B'], 
                                 np.nan, inverse = True)
    return col

data['flurtpcr'] = fluFunc(data['flurtpcr'].values)
data['fluabantigen'] = fluFunc(data['fluabantigen'].values)


# special function that checks pos/neg, det/not det but also has indeterminate category
# kept as strings
def virusConversion(col):
    select = ~(col.astype(np.str == 'nan'))
    for i in range(len(col)):
        if select[i]:
            string = str(col[i]).upper()
            if string == 'CANCELLED':
                col[i] = np.nan
            elif 'NEG' in string or 'NOT' in string:
                col[i] = 'NEGATIVE'
            elif 'POS' in string or ('DET' in string and 'INDET' not in string):
                col[i] = 'POSITIVE'
            else:
                col[i] = 'INDETERM.'
    return col


virusVars = ['adenovirusdfa', 'hcvab', 'hepbcore', 'hepbsag', 'adenoviruspcr', 'cdfgdhantigen',
             'cdfrapidtoxin', 'cmvpcrplasma', 'cdfcytoassay']

for var in virusVars:
    data[var] = virusConversion(data[var].values)
                
            


#######################################################################
## do rest of binary vars: ana and cdfantigen must be done separately

def anaFunc(col):
    select = ~(col.astype(np.str == 'nan'))
    for i in range(len(col)):
        if select[i]:
            string = col[i].lower()
            if 'pos' in string:
                col[i] = 1
            elif 'neg' in string:
                col[i] = 0
            elif ':' in string:
                colLoc = string.find(':')
                try:
                    val = float(string[colLoc+1:])
                    if val >= 80:
                        col[i] = 1
                    else:
                        col[i] = 0
                except ValueError:
                    col[i] = np.nan
    col = col.astype(np.float16)
    return col


data['ana'] = anaFunc(data['ana'].values)


def cdfantigenFunc(x):
   for i in range(len(x)):
       if isinstance(x[i], str):
           x[i] = x[i].lower()
           if 'positive' in x[i]:
               x[i] = 1
           elif 'negative' in x[i]:
               x[i] = 0
           else:
               x[i] = np.nan
   return x

data['cdfantigen'] = cdfantigenFunc(data['cdfantigen'].values)


def binaryConversion(col):
    select = ~(col.astype(np.str == 'nan'))
    for i in range(len(col)):
        if select[i]:
            string = str(col[i]).upper()
            if 'NEG' in string or 'NOT' in string:
                col[i] = 0
            elif 'POS' in string or 'DET' in string:
                col[i] = 1
            else:
                col[i] = np.nan
    return col.astype(np.float16)

binaryVars = ['rhtype', 'hiv1ant', 'hiv2ant', 'hivab', 'cmvpcrplasma', 
              'myourine', 'cdfpcr', 'bkplasma', 'hbeantibody', 'hbeantigen', 'bkurine']

for var in binaryVars:
    data[var] = binaryConversion(data[var].values)         


def specialBinary(l):
    select = ~(l.astype(np.str == 'nan'))
    for i in range(len(l)):
        if select[i]:
            val = re.sub(r'\,+', '', str(l[i]).lower())
            if 'neg' in val or '<' in val or 'not' in val:
                l[i] = 0
            elif 'pos' in val or '>' in val or 'det' in val:
                l[i] = 1
            else:
                try:
                    val = float(val)
                    l[i] = 1
                except TypeError:
                    l[i] = np.nan
    return l.astype(np.float16)


specialVars = ['ddimer', 'hcvquant', 'hiv1rna', 'hepbquant']

for var in specialVars:
    data[var] = specialBinary(data[var].values)

###################################################################################################
############# ADD VARS
###################################################################################################
## pulse pressure, mean arterial pressure, body mass index, length of stay
## bun/creatinine ratio, anion gap, protein gap





