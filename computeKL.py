# compute K-L divergence with gensim
# source: https://markroxor.github.io/gensim/static/notebooks/distance_metrics.html


from gensim.corpora import Dictionary
import numpy as np

from gensim.matutils import kullback_leibler

def kl_divergence(model, reference_doc:str, news_report:list[str], dictionary_news):
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

    reference document model is not needed
    """
    
    # convert to bag of words
    news_id2word = dictionary_news
    bow_refer = news_id2word.doc2bow(reference_doc.split())
    print(news_report[0].split())
    bow_news = news_id2word.doc2bow(news_report[0].split()) # to test the function, calculate the first document in that year only
    
    # update id2word parameter of LDA model, or else IndexError occurs
    # model2(id2word = dictionary)

    # get topic distributions
    # modifier parameters of .get_document_topics as follows to ensure probabilities sum up to 1
    reference_topics = model.get_document_topics(bow_refer, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
    report_topics = model.get_document_topics(bow_news, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)


    print("reference_topics:", reference_topics)
    print("report_topics:", report_topics)
    # print(model.get_document_topics(bow_news))
    
    # print(bow_refer)
    # print(bow_news)
    KL = kullback_leibler(reference_topics, report_topics)

    # compute KL of word distributions instead

    return KL

