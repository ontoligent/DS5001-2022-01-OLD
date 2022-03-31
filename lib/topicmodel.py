from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA, NMF

class TopicModel():            

    """
    A class to convert a BOW table into a topic model consisting of THETA, PHI, and TOPIC tables.
    Attributes:
        bow_count_col (str): The name of the token count column in the BOW table. Defaults to 'n'.
        n_term (int): The number of vocabalary terms to use in the topic model. Defaults to None.
        n_topics (int): The number of topics to generate. Defaults to 20.
        n_top_terms (int): The number of top terms to use to represent each topic. Will compute based on entropy if no value given. Defaults to None.
        engine_type (str): The topic modeling engine to use. May be 'LDA' or 'NMF'. Defaults to 'LDA'.
        alpha (float) = The document-topic prior, e.g. .1.  Defaults to None.
        beta (float) = The topic-term prior, e.g. .01.  Defaults to None.
        
        # LDA Params
        max_iter:int = 20
        learning_offset:float = 50.
        random_state:int = 0
        
        # NMF Params
        nmf_init:str = 'nndsvd'
        nmf_max_iter:int = 1000
        nmf_random_state:int = 1

        kw = {} # Extra parameters

    """

    # General
    bow_count_col:str = 'n'
    n_terms:int = None
    n_topics:int = 20
    n_top_terms:int = None
    engine_type:str = 'LDA' # Also NMF
    alpha:float = None # doc_topic_prior
    beta:float = None  # topic_word_prior
    
    # LDA Params
    max_iter:int = 20
    learning_offset:float = 50.
    random_state:int = 0
    
    # NMF Params
    nmf_init:str = 'nndsvd'
    nmf_max_iter:int = 1000
    nmf_random_state:int = 1

    kw = {} # Extra parameters
    
    def __init__(self, BOW:pd.DataFrame):
        """
        Initialize by passing a bag-of-words table with an OHCO index and 'n' feature of word counts.
        """
        self.BOW = BOW
        
    def create_X(self):
        
        # Convert BOW to DTM (X)
        X = self.BOW[self.bow_count_col].unstack(fill_value=0)

        # Reduce feature space if asked
        V = X[X > 0].sum().to_frame('df')
        if self.n_terms:
            V['idf'] = np.log2(len(X)/V.df)
            V['dfidf'] = V.df * V.idf
            SIGS = V.sort_values('dfidf', ascending=False).head(self.n_terms).index
            self.X = X[SIGS]
        else:
            self.X = X
        self.V = V        
        
    def get_model(self):
        
        if self.engine_type == 'LDA':
            self.engine = LDA(n_components=self.n_topics, 
                                max_iter=self.max_iter, 
                                doc_topic_prior=self.alpha,
                                topic_word_prior=self.beta,
                                learning_offset=self.learning_offset, 
                                random_state=self.random_state,
                                **self.kw)

        elif self.engine_type == 'NMF':
            self.engine = NMF(n_components=self.n_topics, 
                                max_iter=self.nmf_max_iter,
                                init=self.nmf_init, 
                                random_state=self.nmf_random_state, 
                                **self.kw)
                
        self.THETA = pd.DataFrame(self.engine.fit_transform(self.X.values), index=self.X.index)
        self.THETA.columns.name = 'topic_id'
        
        self.PHI = pd.DataFrame(self.engine.components_, columns=self.X.columns)
        self.PHI.index.name = 'topic_id'
        self.PHI.columns.name = 'term_str'
        
        self.TOPIC = self.PHI.sum(1).to_frame('phi_sum')
        self.TOPIC['theta_sum'] = self.THETA.sum()

    def describe_topics(self):
        
        # Compute topic entropy over PHI to get n for top terms
        PHI_P = (self.PHI.T / self.PHI.T.sum())
        PHI_I = np.log2(1/PHI_P)
        self.TOPIC['h'] = round((PHI_I * PHI_P).sum().sort_values(ascending=False), 2)
        if not self.n_top_terms:
            self.n_top_terms = round(self.TOPIC.h.mean())

        # Compute relevant terms
        self.get_relevant_terms(0)
        
        # Get top terms
        self.TOPIC['top_terms'] = self.PHI.apply(lambda x: ' '.join(x.sort_values(ascending=False).head(self.n_top_terms).index), 1)        
        self.TOPIC['label'] = self.TOPIC.apply(lambda x: f"{x.name}: {x.top_terms_rel}", 1)
        

    def get_relevant_terms(self, 𝜆 = .5):
        """
        Compute relevance of topic terms as defined by Sievert and Shirley 2014.
        C. Sievert and K. Shirley, “LDAvis: A Method for Visualizing and Interpreting Topics,” 
        in Proceedings of the workshop on interactive language learning, visualization, and interfaces, 2014, pp. 63–70.
        """
        Ptw = self.PHI.apply(lambda x: x / x.sum(), 1) # L1 norm of PHI rows, i.e. p(w|t)
        Pw = self.PHI.sum() / self.PHI.sum().sum() # Marginal probs of terms in PHI, i.e. p(w)
        self.REL = 𝜆 * np.log2(Ptw) + (1-𝜆) * np.log2(Ptw / Pw)
        self.TOPIC['top_terms_rel'] = self.REL.apply(lambda x: ' '.join(x.sort_values(ascending=False).head(self.n_top_terms).index), 1)

    def get_model_stats(self):
        self.entropy = self.TOPIC.h.sum()
        self.redundancy = 1 - self.entropy / np.log2(self.n_topics)
    
    def get_doc_stats(self):
        self.DOC = (self.THETA.T * np.log2(1/self.THETA.T)).sum().to_frame('entropy')
        self.DOC['max_topic'] = self.THETA.idxmax(1)
        
    def plot_topics(self):
        self.TOPIC.sort_values('theta_sum', ascending=True)\
            .plot.barh(y='theta_sum', x='label', figsize=(5, self.n_topics/2))