import pandas as pd
import numpy as np
from gensim.models import word2vec
from sklearn.manifold import TSNE
import plotly_express as px

class W2V():
    """
    Apply word2vec to a TOKEN table with OHCO index. Returns a word embedding model and a tSNE projection, 
    as well as method to visualize the results. Assumes that TOKEN is annotated with POS column called 'pos'.
    """
    # Arguments for Gensim's word2vec()
    w2v_args = dict(
        min_count = 20,
        vector_size = 100,
        window = 2
    )
    
    # Arguments for SciKit Learn's TSNE()
    tsne_args = dict(
        learning_rate = 200.,
        perplexity = 40,
        n_components = 2,
        init = 'random',
        n_iter = 1000,
        random_state = 23    
    )
    

    def __init__(self, tokens, window_bag, doc_bag):
        """
        Initialize object.
        Arguments:
            tokens (pd.DataFrame): An ETA compliant token table.
            window_bag (list): OCHO slice for bag to use for window. Should be sentences.
            doc_bag (list): OCHO slice for bag to use to compute term significance, e.g. OHCO slide for chapters.
        """
        self.TOKENS:pd.DataFrame = tokens
        self.OHCO = self.TOKENS.index.names
        self.WBAG = window_bag 
        self.DOCBAG = doc_bag 
        print("W2V Bag:", self.WBAG[-1])
        print("DOC Bag:", self.DOCBAG[-1])
        
    def generate_model(self):
        """
        Run all the methods to generate both the word embeddings and the tSNE projection.
        """
        print("Extracting vocabulary")
        self._extract_vocab()
        print('Gathering sentences')
        self._get_sents()
        print("Learning word vectors")
        self._get_model()
        print("Estimating tSNE coordinates")
        self._get_tsne_coords()
        print("Done", u'\u2713')
        
    def _extract_vocab(self):
        """
        Extract vocabulary table VOCAB form TOKENS and compute pos_max, pos_group, and dfidf for terms.
        """
        self.VOCAB = self.TOKENS.term_str.value_counts().to_frame('n')
        self.VOCAB.index.name = 'term_str'
        
        # Todo: Check if TOKENS has `pos`
        self.VOCAB['pos_max'] = self.TOKENS.value_counts(['term_str','pos']).unstack().idxmax(1)
        self.VOCAB['pos_group'] = self.VOCAB.pos_max.str.slice(0,2)
        
        # Add DFIDF 
        self.DOCS = self.TOKENS.groupby(self.DOCBAG +['term_str']).term_str.count()\
            .unstack(fill_value=0).astype('bool').astype('int')
        self.VOCAB['df'] = self.DOCS.sum()
        N = len(self.DOCS)
        self.VOCAB['dfidf'] = self.VOCAB.df * np.log2(N/self.VOCAB.df)
        
    def _get_sents(self):
        """
        Gather senteces (window bags) into lists of tokens for use by Gensim.
        """
        self.SENTS = self.TOKENS.groupby(self.WBAG)\
            .term_str.apply(lambda  x:  x.tolist())\
            .reset_index()['term_str'].tolist()
        
    def _get_model(self):
        """
        Learn word embeddings from SENTS.
        """
        self.model = word2vec.Word2Vec(self.SENTS, **self.w2v_args)
        self.VEC = pd.DataFrame(self.model.wv.get_normed_vectors(), 
                                    index=self.model.wv.index_to_key)
        self.VEC.index.name = 'term_str'
        self.VEC = self.VEC.sort_index()
        
    def _get_tsne_coords(self):
        """
        Project embeddings onto 2D tSNE space.
        """
        self.tsne_engine = TSNE(**self.tsne_args)
        self.tsne_model = self.tsne_engine.fit_transform(self.VEC)        
        self.TSNE = pd.DataFrame(self.tsne_model, columns=['x','y'], index=self.VEC.index)\
                .join(self.VOCAB, how='left')[['x','y','n','dfidf','pos_group']]
        
    def plot_tsne(self, n=1000, method='dfidf'):
        """
        Plot tSNE coordinates. 
        Arguments:
            n (int): Number of terms to plot. Not used if method = 'all'. Be careful with that.
            method (str): Choose whether and how to filter terms to plot. 
                'dfidf' top n terms by dfidf; 'sample' n terms by sample; 'all' all terms. 
                Defaults to 'dfidf.'
        """
        if method == 'dfidf':
            X = self.TSNE.sort_values('dfidf', ascending=False).head(n).reset_index()
        elif method == 'sample':
            if n < len(self.TSNE):
                X = self.TSNE.sample(n).reset_index()
            else:
                X = self.TSNE.reset_index()
        elif method == 'all':
            X = self.TSNE.reset_index()
        else:
            raise ValueError("Unknown method. Use 'difidf' or 'sample'.")        
        px.scatter(X, 'x', 'y', 
                   text='term_str', 
                   color='pos_group', 
                   hover_name='term_str',
                   size='dfidf',
                   height=1000, width=1200)\
            .update_traces(
                mode='markers+text', 
                textfont=dict(color='black', size=14, family='Arial'),
                textposition='top center').show()
        
    def complete_analogy(self, A, B, C, n=2):
        """
        Estimate the fourth term of an analogy. Passes arguments to model.wv.most_similar(positive=[B, C], negative=[A]).
        See document for Gensim's model.wv.most_similar() for more info.
        """
        try:
            cols = ['term', 'sim']
            return pd.DataFrame(self.model.wv.most_similar(positive=[B, C], negative=[A])[0:n], columns=cols)
        except KeyError as e:
            print('Error:', e)
            return None

    def get_most_similar(self, positive, negative=None):
        """
        Get the most similar terms for a given term. Passes arguments to model.wv.most_similar(positive, negative).
        See document for Gensim's model.wv.most_similar() for more info.
        """
        return pd.DataFrame(self.model.wv.most_similar(positive, negative), columns=['term', 'sim'])        