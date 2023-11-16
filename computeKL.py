# compute K-L divergence with gensim
# source: https://markroxor.github.io/gensim/static/notebooks/distance_metrics.html


from gensim.corpora import Dictionary
import numpy as np

from gensim.matutils import kullback_leibler

def kl_divergence(model1, model2, num_topics, reference_doc, news_report, dictionary_refer, dictionary_news):
    """
    Calculate Kullback-Leibler divergence between two LDA models.

    Parameters:
    - model1, model2: LdaModel instances
    - num_topics: Number of topics in the models
    - num_words: Number of words to consider for each topic
    - reference_doc
    - news_report
    - dictionary_refer
    - dictionary_news

    Returns:
    - KL: Kullback-Leibler divergence
    """
    
    # convert to bag of words
    bow_refer = dictionary_refer.doc2bow(reference_doc)
    bow_news = dictionary_news.doc2bow(news_report)
    
    # get topic distributions
    reference_topics = model1[bow_refer]
    report_topics = model2[bow_news]

    
    KL = kullback_leibler(reference_topics, report_topics)

    return KL

