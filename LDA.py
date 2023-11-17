from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from pprint import pprint
import numpy as np
from tqdm import tqdm

def runModel(corpus:list[str], save_path:str, num_topics:int, passes:int=10)->LdaModel:
    '''
    corpus: a list of cut documents
        Sample:
        corpus = ["Text of document 1", "Text of document 2", "Text of document 3", ...]
    num_topics
    passes=10
    '''
    

    # Tokenize and preprocess the data
    tokenized_data = [simple_preprocess(doc) for doc in corpus]

    # Create a Gensim dictionary
    dictionary = corpora.Dictionary(tokenized_data)

    # save dictionary, add a suffix to filename
    dictionary.save(save_path + ".dictionary")

    # Create a Gensim corpus
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_data]

    # Train the LDA model
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

    # Print the topics and their words
    pprint(lda_model.print_topics())

    # save model
    lda_model.save(save_path)
    return lda_model


def bootstrapSample(corpus:list[str], save_path:str, num_topics:int, passes:int=10, num_iterations:int=100, sample_percent:float=0.9)->None:
    '''
    Run bootstrap sampling
    - corpus: list of cut documents in 1 year
    - save_path: path of saved results of bootstrap samples
    - num_topics
    - passes
    - num_iterations: number of bootstrap samples
    - sample_percent: percentage of corpus fitted to each bootstrap sample LDA
    
    return None
    '''

    for i in tqdm(range(num_iterations)):
        sample_indices = np.random.choice(len(corpus), size=int(sample_percent * len(corpus)), replace=True)
        sampled_corpus = [corpus[i] for i in sample_indices]

        # Create dictionary and bag of words
        tokenized_data = [simple_preprocess(doc) for doc in sampled_corpus]
        dictionary = corpora.Dictionary(tokenized_data)
        # save dictionary, add a suffix to filename
        dictionary.save(save_path + ".dictionary")

        bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_data]

        lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary)
        lda_model.save(save_path + "_iter" + str(i))  # save sample
    
    return