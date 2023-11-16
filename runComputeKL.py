from gensim.models import LdaModel
import computeKL
import newsPreprocess as npre
from gensim import corpora
import os
from dotenv import load_dotenv
import pandas as pd
from pprint import pprint

load_dotenv()

# # Example usage
# num_topics = 10
# num_words = 1000

# sample_model_path = os.environ.get("sample_model_save_path")
# sample_corpus_path = os.environ.get("sample_corpus_path")
refer_corpus_path = os.environ.get('cut_refer')    # corpus about InsurTech
refer_corpus_path2 = os.environ.get('cut_refer2')  # corpus about insurance instead of Insurtech

doc_div_chars = doc_div_chars = os.environ.get("doc_div_chars")

def run(model_path, refer_corpus_path, news_path):
    model = LdaModel.load(model_path)  # Your model trained on news reports

    with open(news_path, "r", encoding="utf-8") as f:
        with open(news_path, "r", encoding="utf-8") as f:
            text = f.read()
            news_corpus = text.split(doc_div_chars)

    with open(refer_corpus_path, "r", encoding="utf-8") as f:
        reference_corpus = f.read()


    dictionary_news = corpora.Dictionary.load(model_path + ".dictionary")

    kl_div = computeKL.kl_divergence(model, reference_corpus, news_corpus, dictionary_news)
    print("KL Divergence:", kl_div)
    return kl_div

# run(refer_model_path, sample_model_path, refer_corpus_path, sample_corpus_path)
# run(refer_model_path2, sample_model_path, refer_corpus_path2, sample_corpus_path)


newsModelsFolder = os.environ.get("modelByYear_folder")
news_corpus_folder = os.environ.get("cut_result_by_year")
KL_save_folder = os.environ.get("KL_save_folder")

def computeByYear(newsModelsFolder, refer_corpus_path, news_corpus_folder, KL_save_folder):
    '''Compute K-L divergence of reference document and corpus of each year's news, save results as Excel file.'''

    newsModelPathList = sorted([os.path.join(newsModelsFolder, f) for f in os.listdir(newsModelsFolder) if f.isdigit()])
    newsCorpusPathList = sorted([os.path.join(news_corpus_folder, f) for f in os.listdir(news_corpus_folder) if f.endswith(".txt")])
    # print("newsModelPathList:", newsModelPathList)
    # print("newsCorpusPathList:", newsCorpusPathList)
    
    years = npre.getYearFromFilename()
    # print(years)
    annual_KL = pd.DataFrame(columns=years)
    for newsModelPath, newsCorpusPath, year in zip(newsModelPathList, newsCorpusPathList, years):
        # pprint([newsModelPath, newsCorpusPath, year])
        KL = run(newsModelPath, refer_corpus_path, newsCorpusPath)
        annual_KL.at[0, year] = KL
    
    print(annual_KL)
    KL_save_path = os.path.join(KL_save_folder, os.path.splitext(os.path.basename(refer_corpus_path))[0] + "_newsByYear" + ".xlsx")
    annual_KL.to_excel(KL_save_path)
    return

computeByYear(newsModelsFolder, refer_corpus_path, news_corpus_folder, KL_save_folder)