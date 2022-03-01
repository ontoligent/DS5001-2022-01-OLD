import pandas as pd
import numpy as np
from scipy.linalg import norm, eigh

def create_bow(CORPUS:pd.DataFrame, bag:list, item_type:str='term_str'):
    """
    Create a bag-of-words representation from a table tokens.
    Arguments:
        CORPUS (pd.DataFrame): a DataFrame with an OHCO index and ``['term_str','max_pos']`` in columns.
        bag (list): a slice of the OHCO index identifying the bag level, e.g. ``['book_id', 'chap_id']`` for chapter bags.
        item_type (string): the name of the column containing the normalized token string.
    Returns:
        A DataFrame with an the bag-level OHCO and ``['n']`` column for number of tokens in each bag.
    """
    BOW = CORPUS.groupby(bag+[item_type])[item_type].count().to_frame('n')
    return BOW

def get_tfidf(BOW:pd.DataFrame, tf_method:str='max', idf_method:str='standard'):
    """
    Get a TFIDF-weighted document-term matrix from a bag-of-words table.
    Arguments:
        BOW (pd.DataFrame): A DataFrame produced by ``create_bow()``
        tf_method (string): The term frequency count method. Options: sum, max, log, raw, and bool. Defaults to max.
        idf_method (strings): The inversre document frequency count method. Options: standard, textbook, sklearn, sklearn_smooth. Defaults to standard.
    Returns:
        TFIDF (pd.DataFrame): A DataFrame with an unnormalized, zero-filled document-term matrix of TFIDF weights.
        DFIDF (pd.DataFrame): A Series with a vocabulary as index and DFIDF as value.
    """
            
    DTCM = BOW.n.unstack() # Create Doc-Term Count Matrix
    
    if tf_method == 'sum':
        TF = (DTCM.T / DTCM.T.sum()).T
    elif tf_method == 'max':
        TF = (DTCM.T / DTCM.T.max()).T
    elif tf_method == 'log':
        TF = (np.log2(DTCM.T + 1)).T
    elif tf_method == 'raw':
        TF = DTCM
    elif tf_method == 'bool':
        TF = DTCM.astype('bool').astype('int')
    else:
        raise ValueError(f"TF method {tf_method} not found.")

    DF = DTCM.count() # Assumes NULLs 
    N_docs = len(DTCM)
    
    if idf_method == 'standard':
        IDF = np.log2(N_docs/DF) # This what the students were asked to use
    elif idf_method == 'textbook':
        IDF = np.log2(N_docs/(DF + 1))
    elif idf_method == 'sklearn':
        IDF = np.log2(N_docs/DF) + 1
    elif idf_method == 'sklearn_smooth':
        IDF = np.log2((N_docs + 1)/(DF + 1)) + 1
    else:
        raise ValueError(f"DF method {df_method} not found.")
    
    TFIDF = TF * IDF
    TFIDF = TFIDF.fillna(0)

    DFIDF = DF * IDF

    return TFIDF, DFIDF

def get_pca(TFIDF, 
            k:int=10, 
            norm_docs:bool=True, 
            center_by_mean:bool=True, 
            center_by_variance:bool=False):
    """
    Get principal components and loadings from a TFIDF matrix. Typically, this will be one with a
    a reduced feature spacing, i.e. of the top ``n`` significant terms.
    Arguments:
        k (int): The number of components to return. Defaults to 10.
        norm_docs (bool): Whether to apply L2 normalization or not. Defaults to True.
        center_by_mean (bool): Whether to center term vectors by mean. Defaults to True.
        center_by_variance (bool): Whether to center term vectors by standard deviation. Defaults to False.
    Returns:
        LOADINGS (pd.DataFrame): A DataFrame of terms by principal components. 
        DCM (pd.DataFrame): A DataFrame of documents by principal components.
        COMPINF (pd.DataFrame): A DataFrame of information about each component.
    """
    
    if TFIDF.isna().sum().sum():
        TFIDF = TFIDF.fillna(0)
    
    if norm_docs:
        TFIDF = TFIDF.apply(lambda x: x / norm(x), 1).fillna(0)
    
    if center_by_mean:
        TFIDF = TFIDF - TFIDF.mean()
        
    if center_by_variance:
        TFIDF = TFIDF / TFIDF.std()        

    COV = TFIDF.cov()

    eig_vals, eig_vecs = eigh(COV)
    EIG_VEC = pd.DataFrame(eig_vecs, index=COV.index, columns=COV.index)
    EIG_VAL = pd.DataFrame(eig_vals, index=COV.index, columns=['eig_val'])
    EIG_VAL.index.name = 'term_str'
        
    EIG_IDX = EIG_VAL.eig_val.sort_values(ascending=False).head(k)
    
    COMPS = EIG_VEC[EIG_IDX.index].T
    COMPS.index = [i for i in range(COMPS.shape[0])]
    COMPS.index.name = 'pc_id'

    LOADINGS = COMPS.T

    DCM = TFIDF.dot(LOADINGS)
    
    COMPINF = pd.DataFrame(index=COMPS.index)

    for i in range(k):
        for j in [0, 1]:
            top_terms = ' '.join(LOADINGS.sort_values(i, ascending=bool(j)).head(5).index.to_list())
            COMPINF.loc[i, j] = top_terms
    COMPINF = COMPINF.rename(columns={0:'pos', 1:'neg'})
    
    COMPINF['eig_val'] = EIG_IDX.reset_index(drop=True).to_frame()
    COMPINF['exp_var'] = COMPINF.eig_val / COMPINF.eig_val.sum()
    
    return LOADINGS, DCM, COMPINF
    