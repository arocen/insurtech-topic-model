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
        # print("Debug: sample_index is ", sample_index)
        # print("Debug: len(news_report) is", len(news_report))

        news_report = [news_report[index] for index in sample_index]
    
    for i, report in zip(sample_index, news_report):
        bow_news = news_id2word.doc2bow(report.split()) 
        report_topics = model.get_document_topics(bow_news, minimum_probability=0.000000000001)
        KL = kullback_leibler(reference_topics, report_topics)
        # print("report_topics:", report_topics)

        # calculate the reciprocal
        # KL = 1 / KL # comment this to use arithmetic average instead
        
        if pd.isna(df.at[i, year]):
            # initialize
            df.at[i, year] = KL
        else:
            df.at[i, year] += KL # sum up different values from all bootstrap sample models

    return


def kl_divergence_without_year(df, model, reference_doc:str, reports:list[str], dictionary_reports, sample_index:list[int]=None, column_name="all_corpus"):
    """
    Calculate Kullback-Leibler divergence between reference_doc and every element of reports.

    Parameters:
    - df: DataFrame saving K-L divergence per document
    - model: a LdaModel instance of bootstrap sampled InsurTech analyse reports
    - reference_doc: raw text
    - reports: list of lists
    - dictionary_reports: saved dictionary of a LDA model
    - sample_index: indice of documents used in bootstrap sample model. If none, loop all documents.
    - column_name: column name where K-L divergence is saved in df

    Returns: None.
            Value of df is changed.
    """
    reports_id2word =  dictionary_reports
    bow_refer = reports_id2word.doc2bow(reference_doc.split())
    reference_topics = model.get_document_topics(bow_refer, minimum_probability=0.000000000001)

    if sample_index:
        reports = [reports[index] for index in sample_index]
    
    for i, report in zip(sample_index, reports):
        bow_news = reports_id2word.doc2bow(report.split()) 
        report_topics = model.get_document_topics(bow_news, minimum_probability=0.000000000001)
        KL = kullback_leibler(reference_topics, report_topics)
        # print("report_topics:", report_topics)

        # calculate the reciprocal
        # KL = 1 / KL # comment this to use arithmetic average instead
        
        if pd.isna(df.at[i, column_name]):
            # initialize
            df.at[i, column_name] = KL
        else:
            df.at[i, column_name] += KL # sum up different values from all bootstrap sample models

    return

def kl_divergence_without_year_simple_average(ith_model, df, model, reference_doc:str, reports:list[str], dictionary_reports, sample_index:list[int]=None, column_name="all_corpus"):
    '''
    Average KL are calculated among sample reports of a model, instead of per report.

    - ith_model: ith model. So result value of KL should be written to row i.
    '''
    reports_id2word =  dictionary_reports
    bow_refer = reports_id2word.doc2bow(reference_doc.split())
    reference_topics = model.get_document_topics(bow_refer, minimum_probability=0.000000000001)

    if sample_index:
        reports = [reports[index] for index in sample_index]
    
    KL = 0
    for report in reports:
        bow_news = reports_id2word.doc2bow(report.split()) 
        report_topics = model.get_document_topics(bow_news, minimum_probability=0.000000000001)
        
        KL += kullback_leibler(reference_topics, report_topics)
    
    # Get average
    doc_num = len(sample_index)
    avg_KL = KL / doc_num

    if pd.isna(df.at[ith_model, column_name]):
        df.at[ith_model, column_name] = avg_KL
    else:
        print("Error: Something must be wrong since you write different results to a same cell.")

    return


def kl_divergence_mutal_refer(ith_model, df, model, reference_doc1:str, reference_doc2:str, dictionary_reports, column_name):
    '''
    Calculate KL divergence of topic distributions of 2 input reference documents, based on trained LDA model.
    '''
    reports_id2word =  dictionary_reports
    bow_refer1 = reports_id2word.doc2bow(reference_doc1.split())
    reference_topics1 = model.get_document_topics(bow_refer1, minimum_probability=0.000000000001)

    bow_refer2 = reports_id2word.doc2bow(reference_doc2.split())
    reference_topics2 = model.get_document_topics(bow_refer2, minimum_probability=0.000000000001)

    
    KL = kullback_leibler(reference_topics1, reference_topics2)
    # print(KL)

    if pd.isna(df.at[ith_model, column_name]):
        df.at[ith_model, column_name] = KL
    else:
        print("Error: Something must be wrong since you are writing results from different models to a same cell.")

    return