import pandas as pd
import numpy as np
from gensim.models import word2vec
from sklearn.manifold import TSNE
import plotly_express as px

class W2V():
    
    w2v_args = dict(
        min_count = 20,
        vector_size = 100,
        window = 2
    )
    
    tsne_args = dict(
        learning_rate = 200.,
        perplexity = 40,
        n_components = 2,
        init = 'random',
        n_iter = 1000,
        random_state = 23    
    )
    

    def __init__(self, tokens, window_bag, doc_bag):
        self.TOKENS:pd.DataFrame = tokens
        self.OHCO = self.TOKENS.index.names
        self.WBAG = window_bag
        self.DOCBAG = doc_bag
        print("W2V Bag:", self.WBAG[-1])
        print("DOC Bag:", self.DOCBAG[-1])
        
    def generate_model(self):
        print("Extracting vocabulary")
        self._extract_vocab()
        print('Getting sentences')
        self._get_sents()
        print("Getting word vectors")
        self._get_model()
        print("Getting tSNE coordinates")
        self._get_tsne_coords()
        print("Done", u'\u2713')
        
    def _extract_vocab(self):
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
        self.SENTS = self.TOKENS.groupby(self.WBAG)\
            .term_str.apply(lambda  x:  x.tolist())\
            .reset_index()['term_str'].tolist()
        
    def _get_model(self):
        self.model = word2vec.Word2Vec(self.SENTS, **self.w2v_args)
        
        self.VEC = pd.DataFrame(self.model.wv.get_normed_vectors(), 
                                    index=self.model.wv.index_to_key)
        self.VEC.index.name = 'term_str'
        self.VEC = self.VEC.sort_index()
        
    def _get_tsne_coords(self):
        self.tsne_engine = TSNE(**self.tsne_args)
        self.tsne_model = self.tsne_engine.fit_transform(self.VEC)        
        self.TSNE = pd.DataFrame(self.tsne_model, columns=['x','y'], index=self.VEC.index)\
                .join(self.VOCAB, how='left')[['x','y','n','dfidf','pos_group']]
        
    def plot_tsne(self, n=1000, method='dfidf'):
        if method == 'dfidf':
            X = self.TSNE.sort_values('dfidf', ascending=False).head(n).reset_index()
        elif method == 'sample':
            if n < len(self.TSNE):
                X = self.TSNE.sample(n).reset_index()
            else:
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
        try:
            cols = ['term', 'sim']
            return pd.DataFrame(self.model.wv.most_similar(positive=[B, C], negative=[A])[0:n], columns=cols)
        except KeyError as e:
            print('Error:', e)
            return None

    def get_most_similar(self, positive, negative=None):
        return pd.DataFrame(self.model.wv.most_similar(positive, negative), columns=['term', 'sim'])        