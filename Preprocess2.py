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
from Helper.utilities import saveUnique, save, load, searchColumns


###################################################################################################
###################################################################################################

# load data and create patient encounter indexer (pIndex)
dataLoc = os.path.join('G:', os.sep, 'sas library', 'anotherfinal.csv')
#data = readData(dataLoc, chunkSize = 2e6, numChunks = 1)
csv_iter = pd.read_csv(dataLoc, chunksize = 9e6, iterator = True)
#data = pd.read_csv(dataLoc, nrows = 1e4)
idx_iter = 0

for idx_iter, data in enumerate(csv_iter):

    data.set_index('PAT_ENC_CSN_ID', inplace = True)
    print('loaded data')
    
    #pIndex = getPatientIndices(data.index)
    #m, k = len(data), len(pIndex)
    m = len(data)
    
    # drop vars we know are useless from the getgo
    dropCols = ['PAT_NAME', 'BIRTH_DATE']
    data.drop(labels = dropCols, axis = 1, inplace = True)

    
    # save unique values and counts for all object vars for error checking later
    folder = os.path.join(os.path.expanduser('~'), 'Documents', 'Projects', 'AKI24', 'Data', 'unique')
    countNames = list(data.columns[data.dtypes == np.dtype('O')].values[1:])
    countNames = list(set(countNames) - set(['time', 'HOSP_ADMSN_TIME',
                      'HOSP_DISCH_TIME', 'DEATH_DATE']))
    saveUnique(data, countNames, folder, 'dataPre_part' + str(idx_iter), verbose = False)
    countNames = np.array(countNames).astype('U25')
    print('counting completed and saved')

    
    def replaceName(countNames, name, newName):
        if name in list(countNames):
            idx = np.where(countNames == name)[0][0]
            countNames[idx] = newName
        return countNames
        
    ###################################################################################################
    ############# CLEAN VARS
    ###################################################################################################
    # variables that might need fixing: list(data.columns[data.dtypes == np.dtype('O')])
    # find unique: set(data.loc[~(data[var].astype(np.str) == 'nan'), var])
    
    
    #############################
    ## fix demographic vars
    data.rename(columns={'AGE_AT_ENCOUNTER':'AGE', 'ETHNICITY': 'HISPANIC', 
                         'SEX': 'MALE'}, inplace=True)
    
    countNames = replaceName(countNames, 'AGE_AT_ENCOUNTER', 'AGE')
    countNames = replaceName(countNames, 'ETHNICITY', 'HISPANIC')
    countNames = replaceName(countNames, 'SEX', 'MALE')

    
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
    
    print('demographics fixed')

    
    
    ## convert time vars
    deathDate = data['DEATH_DATE'].apply(lambda x: str(x).replace(' ', 'T')).values
    deathDate[deathDate == 'nan'] = np.nan
    data['DEATH_DATE'] = deathDate
    data['time'] = convertISO8601(data['time'].values)
    data['HOSP_ADMSN_TIME'] = convertISO8601(data['HOSP_ADMSN_TIME'].values)
    data['HOSP_DISCH_TIME'] = convertISO8601(data['HOSP_DISCH_TIME'].values)

    # sorting patient encounters
#    print('Sorting patient encounter data by chronological order')
#    newIndex = np.zeros(m, dtype = np.int64)
#    for i in tqdm(range(k)):
#        start, stop, _ = pIndex[i,:]
#        indices = np.argsort(data['time'].values[pIndex[i,0]:pIndex[i,1]]) + start
#        newIndex[start:stop] = indices
#    data = data.iloc[newIndex,:]
    
    print('times converted')

    
    ######################################################################
    ### fix vars that can be processed through mixed2float
    ## albumin and cholesterolldl and rbcmorph must be done separately

    
    def onlyNums(x, regEx):
       for i in range(len(x)):
           if isinstance(x[i], str):
               num = re.findall(regEx, str(x[i]))
               if len(num) > 0:
                   x[i] = np.nan
       return x
    
    data['albumin'] = onlyNums(data['albumin'].values, '<')
    data['cholesterolldl'] = onlyNums(data['cholesterolldl'].values, '>|<')
    
    
    
    selectedColumns = ['lymphabsdiff', 'baseex', 'acratiourine', 'myoglobin', 'o2sat', 'fio2',
                       'c4complement', 'bicarbonate', 'uricacid', 'ckmb', 'gentamicintrough',
                       'venousbicarb', 'kur', 'albuminurine', 'ucreat24h', 'egfr', 'abslymphcount',
                       'bnp', 'mchc', 'hba1c', 'tacrolevel', 'klratio', 'chlorideurine', 
                       'valproiclevel', 'rbcc', 'pcratio', 'lithiumlevel', 'wbcc', 'uklratio', 
                       'ferritin', 'protein_total', 'alkphos', 'uprotein', 'triglycerides', 
                       'cktotal', 'gbm', 'gentamicinrandom', 'cholesterolhdl', 'pco2', 'ph', 'vpo2',
                       'gentamicinpeak', 'pthintact', 'flckappa', 'pcotart', 'glucose', 'osmolurine',
                       'uunrandom', 'po2', 'phvenous', 'serinepr3', 'vitamind125', 'eos', 'icalcium',
                       'hct', 'po2mv', 'agratio', 'bun', 'troponin', 'monocyteabs', 'flclambda', 
                       'pco2art', 'chloride', 'pt', 'phmv', 'ckmbpoc', 'neutrophils', 'vancorandom',
                       'buncreatratio', 'magnesium', 'fibrinogen', 'ptt', 'prealbumin', 
                       'cholesterol', 'vancotrough', 'ecgpr', 'plateletcount', 'inr', 'sedrate', 
                       'rf', 'vitamind25', 'albumin', 'venouso2sat', 'usodium', 'apo2', 'rdw', 
                       'basophils', 'ironsat', 'phosphorous', 'anc', 'ecgp', 'pthmm', 'pco2venous', 
                       'hco3poc', 'ecgqrsaxis', 'ecgqrsint', 'mpv', 'ammonia', 'co2', 'tibc', 
                       'lactate', 'leadblood', 'lactatedehyd', 'sodium', 'lactatedehyd', 
                       'gentamicin', 'cholesterolldl', 'hemoglobin', 'ethanol', 'iron', 'alt',
                       'globulin', 'ast', 'pco2mv', 'arterialbicarb', 'potassium', 'pharterial',
                       'amylase', 'bilitotal', 'pcotvenous', 'caionplasma', 'aniongap', 'ggt',
                       'urinecalrandom', 'mcv', 'rapamycin', 'vancomycinlevel', 'bilidirect', 
                       'bands', 'ucreatrandom', 'hepbsab', 'mch', 'cyclosporin', 'calcium',
                       'mixedvenouso2sat', 'hscrp', 'ecgheartrate']
                       
                       
    selectedColumns = list(np.array(selectedColumns)[[data[var].dtype == np.dtype('O') \
                           for var in selectedColumns]])

    
    for var in tqdm(selectedColumns):
        data[var] = mixed2float(data[var].values)
    
    print('floats converted')
        
        
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
            0 is assumed to map to negative, and no-match maps to 3+
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
                     'uawbcclumps', 'uacaoxalate', 'uacapyrophosphate', 'uawbccast', 'uauricacid']
        # otherUrineVars = ['uaclarity', 'uacolor',  'uaappearance']
        # uaph, uaspecgrav stays as continuous
        # uaspecgrav has mixed units, which needs to be fixed
        
        # uanitrite, ualeukest, and uablood excluded since no numerical values. will throw KeyError
        allBoundaryMaps = {'uabili': {'trace': 0.2, '1+': 1, '2+': 2},
                           'uaprotein': {'trace': 30, '1+': 100, '2+': 300},
                           'uaglucose': {'trace': 30, '1+': 100, '2+': 300},
                           'uawbcs': {'trace': 2, '1+': 10, '2+': 50},
                           'uawbccast': {'trace': 2, '1+': 10, '2+': 50},
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
                           'uawbccast': mixed2float,
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
    print('ordinals converted')
    
    
    # now do uaclarity, uaappearance, uacolor    
    data.rename(columns={'uaclarity':'uaclaritynormal',
                         'uaappearance':'uaappearancenormal',
                         'uacolor':'uacolornormal'}, inplace=True)
            
    
    countNames = replaceName(countNames, 'uaclarity', 'uaclaritynormal')
    countNames = replaceName(countNames, 'uaappearance', 'uaappearancenormal')
    countNames = replaceName(countNames, 'uacolor', 'uacolornormal')
    

    
    data['uaclaritynormal'] = mixedCategoricalClean(data['uaclaritynormal'].values)
    data['uaclaritynormal'] = stringCollapse(data['uaclaritynormal'].values, ['clear'], 1)
    data['uaclaritynormal'] = stringCollapse(data['uaclaritynormal'].values, [1], 0, inverse = True)
    data['uaclaritynormal'] = data['uaclaritynormal'].astype(np.float16)

    
    data['uaappearancenormal'] = mixedCategoricalClean(data['uaappearancenormal'].values)
    data['uaappearancenormal'] = stringCollapse(data['uaappearancenormal'].values, ['clear'], 1)
    data['uaappearancenormal'] = stringCollapse(data['uaappearancenormal'].values, [1], 0, inverse = True)
    data['uaappearancenormal'] = data['uaappearancenormal'].astype(np.float16)
    
    data['uacolornormal'] = mixedCategoricalClean(data['uacolornormal'].values)
    collapseList = ['yellow', 'colorless', 'pale yellow']
    data['uacolornormal'] = stringCollapse(data['uacolornormal'].values, collapseList, 1)
    data['uacolornormal'] = stringCollapse(data['uacolornormal'].values, [1], 0, inverse = True)
    data['uacolornormal'] = data['uacolornormal'].astype(np.float16)
    
    
    # now do uaph and uaspecgrav
    uaspecgrav = mixed2float(data['uaspecgrav'].values)
    select = uaspecgrav > 1000
    uaspecgrav[select] = uaspecgrav[select]/1000
    data['uaspecgrav'] = uaspecgrav
    
    data['uaph'] = mixed2float(data['uaph'].values)
    
    print('remaining urine vars converted')
    
    
    
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
    
    
    # do flurtpcr and fluabantigen and fludfa
    def fluFunc(col):
        for i in range(len(col)):
            if isinstance(col[i], np.str):
                col[i] = col[i].upper()
        col = stringCollapse(col, ['FLU A DET', 'FLU A DETECTED', 'FLU A POS', 'FLU A POSITIVE'], 
                                    'INFLUENZA A')
        col = stringCollapse(col, ['FLU B DET', 'FLU B DETECTED', 'FLU B POS', 'FLU B POSITIVE'], 
                                     'INFLUENZA B')
        col = stringCollapse(col, ['NEGATIVE', 'NEG', 'FLU A NEG', 'FLU B NEG', 'FLU A NEGATIVE', 
                                   'FLU B NEGATIVE'], 'NOT DETECTED')
        col = stringCollapse(col, ['NOT DETECTED', 'INFLUENZA A', 'INFLUENZA B'], 
                                     np.nan, inverse = True)
        return col
    
    data['flurtpcr'] = fluFunc(data['flurtpcr'].values)
    data['fluabantigen'] = fluFunc(data['fluabantigen'].values)
    data['fludfa'] = fluFunc(data['fludfa'].values)

    
    
    # special function that checks pos/neg, det/not det but also has indeterminate category
    # kept as strings
    def virusConversion(col):
        select = ~(col.astype(np.str) == 'nan')
        for i in range(len(col)):
            if select[i]:
                string = str(col[i]).upper()
                if string == 'CANCELLED':
                    col[i] = np.nan
                elif 'NEG' in string or 'NOT' in string or 'NON' in string:
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
                    
    print('virus vars and weird categoricals converted')
    
    
    #######################################################################
    ## do rest of binary vars: ana, rbcmorph, cdfantigen, and speechswall must be done separately
    
    data.rename(columns={'rbcmorph':'rbcnormal'}, inplace=True)
    countNames = replaceName(countNames, 'rbcmorph', 'rbcnormal')

    rbcnormal = mixedCategoricalClean(data['rbcnormal'].values)
    rbcnormal = stringCollapse(rbcnormal, ['normal', 'n'], 1)
    rbcnormal = stringCollapse(rbcnormal, [1], 0, inverse = True)
    data['rbcnormal'] = rbcnormal.astype(np.float16)
    
    
    def anaFunc(col):
        select = ~(col.astype(np.str) == 'nan')
        for i in range(len(col)):
            if select[i]:
                string = str(col[i]).lower()
                try:
                    if 'pos' in string:
                        col[i] = 1
                    elif 'neg' in string:
                        col[i] = 0
                    elif ':' in string:
                        colLoc = string.find(':')
                        val = float(string[colLoc+1:])
                        if val >= 80:
                            col[i] = 1
                        else:
                            col[i] = 0
                    else:
                        val = float(string)
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
                   x[i] = 'POSITIVE'
               elif 'negative' in x[i]:
                   x[i] = 'NEGATIVE'
               elif 'indet' in x[i]:
                   x[i] = 'INDETERM.'
               else:
                   x[i] = np.nan
       return x
                   
    
    data['cdfantigen'] = cdfantigenFunc(data['cdfantigen'].values)
    
    def speechswallFunc(x):
        select = ~(x.astype(np.str) == 'nan')
        for i in range(len(x)):
            if select[i]:
                x[i] = 1
        return x.astype(np.float16)
                   
    
    data['speechswall'] = speechswallFunc(data['speechswall'].values)
    
    
    def binaryConversion(col):
        select = ~(col.astype(np.str) == 'nan')
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
        select = ~(l.astype(np.str) == 'nan')
        for i in range(len(l)):
            if select[i]:
                val = re.sub(r'\,+', '', str(l[i]).lower())
                try:
                    if 'neg' in val or '<' in val or 'not' in val:
                        l[i] = 0
                    elif 'pos' in val or '>' in val or 'det' in val:
                        l[i] = 1
                    else:
                        val = float(val)
                        l[i] = 1
                except (TypeError, ValueError):
                    l[i] = np.nan
        return l.astype(np.float16)
    
    
    specialVars = ['ddimer', 'hcvquant', 'bkquant', 'cmvquant', 'hiv1rna', 'hepbquant']
    
    for var in specialVars:
        data[var] = specialBinary(data[var].values)
    
    print('binary vars converted')
    
     
    ###################################################################################################
    ############# ADD VARS
    ###################################################################################################
    ## pulse pressure, mean arterial pressure, body mass index, length of stay
    ## bun/creatinine ratio, anion gap, protein gap
    
    
    data["proteingap"] = data["protein_total"] - data["albumin"]
    data["MAP"] = 1/3 * (data["systolic"]) + 2/3 * (data["diastolic"])
    data["pulse_pressure"] = data["systolic"] - data["diastolic"]
    data["BMI"] = data["WEIGHT"] / ((data["HEIGHT"] * 0.0254) ** 2)

    print('engineered variables added')


    ## save final stuff
    saveUnique(data, countNames, folder, 'dataPost_part' + str(idx_iter), verbose = False)
    data.to_csv(os.path.join('G:', os.sep, 'EHR_preprocessed', 
                             'final_part' + str(idx_iter) + '.csv'))
    
    print('Iteration: ', str(idx_iter), ' completed')



