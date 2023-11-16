from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from pprint import pprint


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