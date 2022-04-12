from re import X
import pandas as pd
import numpy as np
from scipy.linalg import norm, eigh


class PCA():
    """
    Get principal components and loadings from a matrix X, such as count matrix. 

    Get/Set Attributes:
        k (int): The number of components to return. Defaults to 10.
        norm_docs (bool): Whether to apply L2 normalization or not. Defaults to True.
        center_by_mean (bool): Whether to center term vectors by mean. Defaults to True.
        center_by_variance (bool): Whether to center term vectors by standard deviation. Defaults to False.

    Generated Attributes:
        LOADINGS (pd.DataFrame): A DataFrame of features by principal components. 
        OCM (pd.DataFrame): A DataFrame of observations by principal components.
        COMPS (pd.DataFrame): A DataFrame of information about each component.
    """

    k:int=10 
    norm_rows:bool=True
    center_by_mean:bool=False
    center_by_variance:bool=False
    method:str='standard' # 'svd'
    n_top_terms:int=5


    def __init__(self, X:pd.DataFrame) -> None:
        self.X = X
        if self.X.isna().sum().sum():
            self.X = self.X.fillna(0)

    def compute_pca(self):
        self._generate_covariance_matrix()
        if self.method == 'standard':
            self._compute_by_eigendecomposition()
        elif self.method == 'svd':
            self._compute_by_svd()
        else:
            raise ValueError(f"Unknown method {self.method}. Try 'standard' or 'svd'.")
        self._get_top_terms()
    
    def _generate_covariance_matrix(self):
        """
        Get the covariance matrix of features from the input matrix X.
        Apply norming and centering if wanted. Note that PCA as LSA does
        not apply centering by mean or variance.
        """
        if self.norm_rows:
            self.X = self.X.apply(lambda x: x / norm(x), 1).fillna(0)    
        if self.center_by_mean:
            self.X = self.X - self.X.mean()        
        if self.center_by_variance:
            self.X = self.X / self.X.std()        
        self.COV = self.X.cov()

    def _compute_by_svd(self):
        """
        Use SVD to compute objects.
        """
        u, d, vt = np.linalg.svd(self.X)

        self.OCM = pd.DataFrame(u[:,:self.k], index=self.X.index).iloc[:,:self.k] 
        self.COMPS = pd.DataFrame(d[:self.k], columns = ['weight'])
        self.COMPS.index.name = 'pc_id'
        self.LOADINGS = pd.DataFrame(vt.T[:, :self.k], index=self.X.columns)
        self.LOADINGS.columns.name = 'pc_id'
        self.LOADINGS.index.name = 'category_id'

    def _compute_by_eigendecomposition(self):
        """
        Use Eigendecomposition to compute objects.
        """
        eig_vals, eig_vecs = eigh(self.COV)
        EIG_VEC = pd.DataFrame(eig_vecs, index=self.COV.index, columns=self.COV.index)
        EIG_VAL = pd.DataFrame(eig_vals, index=self.COV.index, columns=['eig_val'])
        EIG_IDX = EIG_VAL.eig_val.sort_values(ascending=False).head(self.k)

        self.LOADINGS = EIG_VEC[EIG_IDX.index]
        self.LOADINGS.columns = [i for i in range(self.LOADINGS.shape[1])]
        self.LOADINGS.columns.name = 'pc_id'
        self.LOADINGS.index.name = 'category_id'
        
        self.OCM = self.X.dot(self.LOADINGS)

        self.COMPS = pd.DataFrame(index=self.LOADINGS.columns)
        self.COMPS['eig_val'] = EIG_IDX.reset_index(drop=True).to_frame()
        self.COMPS['exp_var'] = self.COMPS.eig_val / self.COMPS.eig_val.sum()

    def _get_top_terms(self):
        """
        Generate topic-like lists from LOADINGS
        """
        for i in range(self.k):
            for j, pole in enumerate(['neg','pos']):
                top_terms = ' '.join(self.LOADINGS.sort_values(i, ascending=bool(j))\
                    .head(self.n_top_terms).index.to_list())
                self.COMPS.loc[i, pole] = top_terms

        # for i in range(self.k):
        #     for j in [0, 1]:
        #         top_terms = ' '.join(self.LOADINGS.sort_values(i, ascending=bool(j)).head(self.n_top_terms).index.to_list())
        #         self.COMPS.loc[i, j] = top_terms
        # self.COMPS = self.COMPS.rename(columns={0:'pos', 1:'neg'})
    
    
    
    if __name__ == '__main__':
        pass

        test_file = ""