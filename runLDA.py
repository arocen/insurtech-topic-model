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

def testLDA():
    with open(sample_corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
        docs = text.split()
    LDA.runModel(docs, sample_model_save_path, num_topics=15)
    return

def fitReference(cut_refer_doc_path, refer_model_save_path):
    '''Fit reference document about Insurtech to LDA, save model.'''
    with open(cut_refer_doc_path, "r", encoding="utf-8") as f:
        text = f.read()
        docs = text.split()
    LDA.runModel(docs, refer_model_save_path, num_topics=15)
    return

def runLDAByYear(num_topics=15, cut_news_folder=cut_news_folder, modelByYear_save_folder=modelByYear_folder):
    # to-do:
    corpusByYear = npre.load_preprocessed_multi_corpus(cut_news_folder)
    years = npre.getYearFromFilename(cut_news_folder)
    for corpus, year in tqdm(zip(corpusByYear, years)):
        save_path = os.path.join(modelByYear_save_folder, year)
        LDA.runModel(corpus, save_path, num_topics)
    return


# fit sample news to LDA
# testLDA()

# fit reference document to LDA
cut_refer_doc_path = os.environ.get("cut_refer")
refer_model_save_path = os.environ.get("refer_model_save_path")
cut_refer_doc_path2 = os.environ.get('cut_refer2')
refer_model_save_path2 = os.environ.get('refer_model_save_path2')

# referModel = fitReference(cut_refer_doc_path, refer_model_save_path)
# referModel2 = fitReference(cut_refer_doc_path2, refer_model_save_path2)

runLDAByYear()