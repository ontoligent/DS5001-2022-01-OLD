import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt            

class HAC:
    """This class takes an arbitrary vector space and represents it 
    as a hierarhical agglomerative cluster tree. The number of observations
    should be sufficiently small to allow being plotted."""

    w:int = 10
    labelsize:int = 14
    orientation:str = 'left'
    dist_metric:str = 'cosine' # ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
    linkage_method:str = 'ward'
    norm:str = 'l2' # l1, l2, max
    
    def __init__(self, M, labels=None):
        self.M = M
        self.h = M.shape[0]
        if labels:
            self.labels = labels            
        else:
            self.labels = M.index.tolist()

    def get_sims(self):
        self.SIMS = pdist(normalize(self.M, norm=self.norm), metric=self.dist_metric)

    def get_tree(self):
        self.TREE = sch.linkage(self.SIMS, method=self.linkage_method)        
        
    def plot_tree(self):
        plt.figure()
        plt.subplots(figsize=(self.w, self.h / 3))
        sch.dendrogram(self.TREE, labels=self.labels, orientation=self.orientation);
        plt.tick_params(axis='both', which='major', labelsize=self.labelsize)
        
    def plot(self):
        self.get_sims()
        self.get_tree()
        self.plot_tree()
