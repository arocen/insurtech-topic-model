# test and run LDA.py
import os
import LDA
import newsPreprocess as npre
from tqdm import tqdm
from dotenv import load_dotenv
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
    '''Fit reference document and news reports to same model'''

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

# To-do: 
# Bootstrap sample. 100 samples, 90% news reports.
# Derive topic distribution of reference document based on bootstrap samples. (reference documents are not included in the corpus of LDA)


# fit sample news to LDA
# testLDA()

# fit reference document to LDA


# referModel = fitReference(cut_refer_doc_path, refer_model_save_path)
# referModel2 = fitReference(cut_refer_doc_path2, refer_model_save_path2)

runLDAByYear()