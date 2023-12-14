# Run LDA with single report in 1 year and calculate KL dicergence.

from dotenv import load_dotenv
import os
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.matutils import kullback_leibler

load_dotenv()


cut_renbao_2015 = os.environ.get("cut_renbao_2015")
cut_renbao_2016 = os.environ.get("cut_renbao_2016")
model_save_folder_2015 = os.environ.get("renbao_2015_model")
model_save_folder_2016 = os.environ.get("renbao_2016_model")

def singleDocLda(doc_path, model_save_folder, model_save_name, num_topics=15):
    '''Run LDA with single report in 1 year.'''
    with open(doc_path, "r", encoding="utf-8") as f:
        doc = f.read()
    tokenized_data = [simple_preprocess(doc)]
    dictionary = corpora.Dictionary(tokenized_data)

    # save dictionary, add a suffix to filename
    dictionary.save(model_save_folder + ".dictionary")

    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_data]

    lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary)
    lda_model.save(os.path.join(model_save_folder, model_save_name))
    return


def calcKL(model_path, doc_path, refer_corpus_path):
    '''Calculate KL divergence based on LDA models with single report in 1 year.'''
    
    # load corpus
    with open(doc_path, "r", encoding="utf-8") as f:
        report_corpus = f.read()
        
    # load reference doc
    with open(refer_corpus_path, "r", encoding="utf-8") as f:
        reference_corpus = f.read()

    # load model
    model = LdaModel.load(model_path)

    # convert to bag of words
    dictionary_report = corpora.Dictionary.load(model_path + ".id2word")
    report_id2word = dictionary_report
    bow_refer = report_id2word.doc2bow(reference_corpus.split())    # use .split to tokenize words
    # set minimum_probability to a very small value to ensure probabilities sum up to 1
    reference_topics = model.get_document_topics(bow_refer, minimum_probability=0.000000000001)

    bow_report = report_id2word.doc2bow(report_corpus.split()) 
    report_topics = model.get_document_topics(bow_report, minimum_probability=0.000000000001)
    KL = kullback_leibler(reference_topics, report_topics)
    print(reference_topics)
    print(report_topics)
    print("KL:", KL)
    return KL
    

# 人保2015
# singleDocLda(cut_renbao_2015, model_save_folder_2015, "2015")

# 人保2016
# singleDocLda(cut_renbao_2016, model_save_folder_2016, "2016")


refer_corpus_path = os.environ.get('cut_refer')    # corpus about InsurTech
# calcKL(os.path.join(model_save_folder_2015, "2015"), cut_renbao_2015, refer_corpus_path)

# calcKL(os.path.join(model_save_folder_2016, "2016"), cut_renbao_2016, refer_corpus_path)