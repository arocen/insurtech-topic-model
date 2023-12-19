# test and run LDA.py
import os
import LDA
import newsPreprocess as npre
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

load_dotenv()

sample_corpus_path = os.environ.get("sample_corpus_path")
sample_model_save_path = os.environ.get("sample_model_save_path")
modelByYear_folder = os.environ.get("modelByYear_folder")
cut_news_folder = os.environ.get("cut_result_by_year")
doc_div_chars = os.environ.get("doc_div_chars")

cut_refer_doc_path = os.environ.get("cut_refer")
refer_model_save_path = os.environ.get("refer_model_save_path")
cut_refer_doc_path2 = os.environ.get('cut_refer2')
refer_model_save_path2 = os.environ.get('refer_model_save_path2')

bootstrap_folder = os.environ.get("bootstrap_folder")

def testLDA():
    with open(sample_corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
        docs = text.split(doc_div_chars)
    LDA.runModel(docs, sample_model_save_path, num_topics=15)
    return

def fitReference(cut_refer_doc_path, refer_model_save_path):
    '''Fit reference document about Insurtech to LDA, save model.'''
    with open(cut_refer_doc_path, "r", encoding="utf-8") as f:
        text = f.read()

    LDA.runModel(text, refer_model_save_path, num_topics=15)
    return

def runLDAByYear(num_topics=15, cut_news_folder=cut_news_folder, modelByYear_save_folder=modelByYear_folder):
    
    corpusByYear = npre.load_preprocessed_multi_corpus(cut_news_folder)
    years = npre.getYearFromFilename(cut_news_folder)
    for corpus, year in tqdm(zip(corpusByYear, years)):
        save_path = os.path.join(modelByYear_save_folder, year)
        LDA.runModel(corpus, save_path, num_topics)
    return

def runLDAByYearWithRefer(num_topics=15, cut_refer_doc_path=cut_refer_doc_path, cut_news_folder=cut_news_folder, modelByYear_save_folder=modelByYear_folder):
    '''
    Fit reference document and news reports to same model.
    To train LDA to infer reference documents, use runLDAByYear instead.
    '''

    with open(cut_refer_doc_path, "r", encoding="utf-8") as f:
        refer = f.read()
    
    corpusByYear = npre.load_preprocessed_multi_corpus(cut_news_folder)
    years = npre.getYearFromFilename(cut_news_folder)
    for corpus, year in tqdm(zip(corpusByYear, years)):
        save_path = os.path.join(modelByYear_save_folder, year)

        # add reference doc to corpus list
        corpus.append(refer) # Do not assign return value (which is None) to a new variable
        LDA.runModel(corpus, save_path, num_topics)
    return



def bootstrapByYear(num_topics=15, cut_news_folder=cut_news_folder, bootstrap_folder=bootstrap_folder, num_iterations=100):
    corpusByYear = npre.load_preprocessed_multi_corpus(cut_news_folder)
    years = npre.getYearFromFilename(cut_news_folder)
    indices = pd.DataFrame()
    for corpus, year in zip(corpusByYear, years):
        save_path = os.path.join(bootstrap_folder, year)
        LDA.bootstrapSample(corpus, save_path, num_topics, year, indices, num_iterations=num_iterations)
    indices.to_excel(os.path.join(bootstrap_folder, "indices.xlsx"))
    return

def loadAllCorpus()->list[list[str]]:
    '''Load sorted all cut corpus.'''
    all_corpus = []
    names = ["cut_analyse_report_folder_pingan", "cut_analyse_report_folder_renbao", "cut_analyse_report_folder_xinhua",
                 "cut_analyse_report_folder_taibao", "cut_analyse_report_folder_guoshou"]
    for name in names:
        path = os.environ.get(name)
        corpus_by_year = npre.load_preprocessed_multi_corpus(path)
        corpus = [doc for doc_in_same_year in corpus_by_year for doc in doc_in_same_year]
        all_corpus.extend(corpus)
    all_corpus = sorted(all_corpus)
    return all_corpus
    

def bootstrapAllCompanies(num_topics=15, sample_percent=0.9, num_iterations=100, save_folder=os.environ.get("bootstrapModelAllAnalyseReports")):
    '''Bootstrap sample in all analyse reports.'''
    all_corpus = loadAllCorpus()
    indices = pd.DataFrame()
    for i in tqdm(range(num_iterations)):
        
        # set replace=False to ensure indices are unique
        sample_indices = np.random.choice(len(all_corpus), size=int(sample_percent * len(all_corpus)), replace=False)
        sampled_corpus = [all_corpus[i] for i in sample_indices]

        # Create dictionary and bag of words
        tokenized_data = [simple_preprocess(doc) for doc in sampled_corpus]
        dictionary = corpora.Dictionary(tokenized_data)
        bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_data]

        lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary)

        save_path = os.path.join(save_folder, "iter" + str(i))
        lda_model.save(save_path)  # save sample
        indices.at[i, "all_corpus"] = str(sample_indices.tolist())
    # save indices
    indices.to_excel(os.path.join(save_folder, "indices.xlsx"))
    return
    


# Bootstrap sample. 100 samples, 90% news reports.
# Derive topic distribution of reference document based on bootstrap samples. (reference documents are not included in the corpus of LDA)

# Bootstrap samples on news reports
# bootstrapByYear(num_iterations=100)


# Bootstrap samples on analyse reports about 平安
# 2015 to 2023 years
# cut_analyse_report_folder_pingan = os.environ.get("cut_analyse_report_folder_pingan")
# bootstrapModels_pingan = os.environ.get("bootstrapModels_pingan")
# bootstrapByYear(cut_news_folder=cut_analyse_report_folder_pingan, bootstrap_folder=bootstrapModels_pingan)

# Bootstrap samples on analyse reports about 人保
# 2018 to 2023 years
# cut_analyse_report_folder_renbao = os.environ.get("cut_analyse_report_folder_renbao")
# bootstrapModels_renbao = os.environ.get("bootstrapModels_renbao")
# bootstrapByYear(cut_news_folder=cut_analyse_report_folder_renbao, bootstrap_folder=bootstrapModels_renbao)

# 新华
# cut_analyse_report_folder_xinhua = os.environ.get("cut_analyse_report_folder_xinhua")
# bootstrapModels_xinhua = os.environ.get("bootstrapModels_xinhua")
# bootstrapByYear(cut_news_folder=cut_analyse_report_folder_xinhua, bootstrap_folder=bootstrapModels_xinhua)

# 太保
# cut_analyse_report_folder_taibao = os.environ.get("cut_analyse_report_folder_taibao")
# bootstrapModels_taibao = os.environ.get("bootstrapModels_taibao")
# bootstrapByYear(cut_news_folder=cut_analyse_report_folder_taibao, bootstrap_folder=bootstrapModels_taibao)

# 国寿
# cut_analyse_report_folder_guoshou = os.environ.get("cut_analyse_report_folder_guoshou")
# bootstrapModels_guoshou = os.environ.get("bootstrapModels_guoshou")
# bootstrapByYear(cut_news_folder=cut_analyse_report_folder_guoshou, bootstrap_folder=bootstrapModels_guoshou)


# Bootstrap sample in all reports
# bootstrapAllCompanies()