# compute K-L divergence with gensim

import pandas as pd
from gensim.matutils import kullback_leibler

def kl_divergence(df, model, reference_doc:str, news_report:list[str], dictionary_news, year, sample_index:list[int]=None):
    """
    Calculate Kullback-Leibler divergence between reference_doc and every element of news_report.

    Parameters:
    - df: DataFrame saving K-L divergence per year per document
    - model: LdaModel instances of InsurTech news reports in each year 
    - reference_doc: raw text
    - news_report: list of lists
    - dictionary_news: saved dictionary of news LDA model in each year
    - year: used to index df
    - sample_index: indice of documents used in bootstrap sample model. If none, loop all documents.

    Returns: None.
            Value of df is changed.
    """
    
    # convert to bag of words
    news_id2word = dictionary_news
    bow_refer = news_id2word.doc2bow(reference_doc.split())    # use .split to tokenize words
    # set minimum_probability to a very small value to ensure probabilities sum up to 1
    reference_topics = model.get_document_topics(bow_refer, minimum_probability=0.000000000001)
    # print("reference_topics:", reference_topics)
    if sample_index:
        news_report = [news_report[index] for index in sample_index]
    
    for i, report in zip(sample_index, news_report):
        bow_news = news_id2word.doc2bow(report.split()) 
        report_topics = model.get_document_topics(bow_news, minimum_probability=0.000000000001)
        KL = kullback_leibler(reference_topics, report_topics)
        # print("report_topics:", report_topics)

        # calculate the reciprocal
        KL = 1 / KL
        
        if pd.isna(df.at[i, year]):
            # initialize
            df.at[i, year] = KL
        else:
            df.at[i, year] += KL # sum up different values from all bootstrap sample models

    return
