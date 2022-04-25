# Hillenbrand Database: fetch the data and select parts of it 


import pandas as pd

    
def fetch_hillenbrand(genders='all',vowels='all',columns=['gender','vowel','f0','F1','F2','F3'],Debug=False):

    '''
    The function fetch_hillenbrand() loads the Hillenbrand dataset in a similar way as the datasets in sklearn.
    There are extra arguments that lets one select parts of the database 

    The Hillenbrand dataset is a 1995 repeat and extension of the classic Peterson-Barney(1953) experiment
    in which Formants are established as compact and highly discriminative features for vowel recognition
    (c) 1995 James Hillenbrand
    https://homepages.wmich.edu/~hillenbr/voweldata.html
    
    The interface provided here reads from a copy of the data at ESAT stored in a more 
    convenient csv format and in which the 0 values (not available) are replaced by #N/A

    =================   ==============
    Classes 
            (genders)      4 (m,w,b,g)
            (vowels)      12 (ae,ah,aw,eh,er,ei,ih,iy,oa,oo,uh,uw)
            (spkr)        151 (100 adults, 51 for children)
    Samples per class     12 vowels x 151 speakers
    Samples total         1668
    Index                 fid  is file-id which is = gid+spk#+vid
    Dimensionality        22   
                             3 categorical features (vid, gid, sid)  [for vowel-id,gender-id and speaker-id] (each combination is unique)
                            19 numerical features: (dur,f0,F1,F2,F3,F4,F1-1,F2-1,F3-1,F1-2,F2-2,F3-2,F1-3,F2-3,F3-3,Start,End,Center1,Center2) 
    Features            real, positive  
    Missing Features    partial missing data is given as NaN, mainly F3 and F4 values
    =================   ==============
    
    With this interface you can load specific parts of the database (subset of speakers, vowels and features)
    that you want to use
   
    Parameters
    ----------
        genders:  list of selected genders  (default=all, options are 'adults','children','male','female' or list of 'm','f','b','g')
        vowels:   list of selected vowels   (default=all, options are 'vowels6', 'vowels3' or list)
        columns:  list of selected columns  (default=['gid','vid','f0','F1','F2','F3'])
        Debug:    False(def) or True
        
    Returns
    -------
        pandas dataframe with selected data
    
    '''
    
    # STEP 1: READ ALL THE DATA
    ###########################
    # the dataframe 'columns' contain the fields in our dataset, the first 3 fields are identifiers 
    # for vowel, gender and speaker, the remaining 19 are datafields
    # print(hildata.columns)
    # the dataframe 'index' allows for accessing the records by name, it is the file-id in the first column
    # print(hildata.index)
    #
    hil_filepath = 'http://homes.esat.kuleuven.be/~spchlab/data/hillenbrand/vowdata.csv'    
    hildata = pd.read_csv(hil_filepath,index_col=0)  
    hildata.rename(columns={'vid':'vowel'},inplace=True)
    hildata.rename(columns={'gid':'gender'},inplace=True)
    hildata.rename(columns={'sid':'spkid'},inplace=True)
    hildata.rename(index={'fid':'fileid'},inplace=True)
    return(select_hillenbrand(hildata,genders=genders,vowels=vowels,columns=columns))
    
    
def select_hillenbrand(df,genders=[],vowels=[],columns='all'):

    allcolumns = list(df.columns.values)
        
    # select the appropriate data records
    #############################################
    if type(genders) is str:
        if genders == 'adults':
            genders = ['m','w']
        elif genders == 'children':
            genders = ['b','g']
        elif genders == 'male':
            genders = ['m','b']
        elif genders == 'female':
            genders = ['w','g']
        elif genders == 'all':
            genders = list(df['gender'].unique())
        
    if type(vowels) is str:
        if vowels == 'vowels6':
            vowels = ['aw','eh','er','ih','iy','uw']
        elif vowels == 'vowels3':
            vowels = ['aw','iy','uw']
        elif vowels == 'pb':
            vowels = ['iy','ih','eh','ae','ah','aw','oo','uw','uh','er']
        elif (vowels == 'all') :
            vowels = list(df['vowel'].unique())
        
    if type(columns) is not list: columns = allcolumns

    if len(genders) == 0 : # don't select on gender
        df1 = df
    else:
        df1 =  df.loc[ (df['gender'].isin(genders)) ]
        
    if len(vowels) == 0 : # don't select on vowel
        df2 = df1
    else:
        df2 =  df1.loc[ df1['vowel'].isin(vowels) ]           
    
    return( df2[ columns] )