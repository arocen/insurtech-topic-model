from gensim.models import LdaModel
import computeKL
import newsPreprocess as npre
from gensim import corpora
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import re
import ast

load_dotenv()

# # Example usage
# num_topics = 10
# num_words = 1000

# sample_model_path = os.environ.get("sample_model_save_path")
# sample_corpus_path = os.environ.get("sample_corpus_path")
refer_corpus_path = os.environ.get('cut_refer')    # corpus about InsurTech
refer_corpus_path2 = os.environ.get('cut_refer2')  # corpus about insurance instead of Insurtech

doc_div_chars = doc_div_chars = os.environ.get("doc_div_chars")

def run(df, model_path, refer_corpus_path, news_path, year, sample_index):
    model = LdaModel.load(model_path)  # Your model trained on news reports

    with open(news_path, "r", encoding="utf-8") as f:
        with open(news_path, "r", encoding="utf-8") as f:
            text = f.read()
            news_corpus = text.split(doc_div_chars)

    with open(refer_corpus_path, "r", encoding="utf-8") as f:
        reference_corpus = f.read()


    dictionary_news = corpora.Dictionary.load(model_path + ".id2word")

    computeKL.kl_divergence(df, model, reference_corpus, news_corpus, dictionary_news, year, sample_index)
    return

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
    KL_year_company = pd.DataFrame(columns=years)
    for newsModelPath, newsCorpusPath, year in tqdm(zip(newsModelPathList, newsCorpusPathList, years)):
        run(KL_year_company, newsModelPath, refer_corpus_path, newsCorpusPath, year)
        
    
    print(KL_year_company)
    KL_save_path = os.path.join(KL_save_folder, os.path.splitext(os.path.basename(refer_corpus_path))[0] + "_newsByYear" + ".xlsx")
    KL_year_company.to_excel(KL_save_path)
    return

def computeBootstrapByYear(newsModelsFolder, refer_corpus_path, news_corpus_folder, KL_save_folder, indices_path):

    years = npre.getYearFromFilename(news_corpus_folder) # The default parameter here must change according to corpus selected.
    
    modelnames = [f for f in os.listdir(newsModelsFolder) if isSampleName(f)]
    # group modelnames by year
    ModelPathGroup = []


    for year in years:
        curr_year_paths = []
        for filename in modelnames:
            if filename[:4] == year:
                curr_year_paths.append(os.path.join(newsModelsFolder, filename))
        ModelPathGroup.append(curr_year_paths)



    newsCorpusPathList = sorted([os.path.join(news_corpus_folder, f) for f in os.listdir(news_corpus_folder) if f.endswith(".txt")])
    # print("newsModelPathList:", newsModelPathList)
    # print("newsCorpusPathList:", newsCorpusPathList)

    indices = pd.read_excel(indices_path)
    
    
    # print(years)
    KL_year_company = pd.DataFrame(columns=years, index=range(2062)) # set index with the range of most reports in each year
    for curr_year_paths, newsCorpusPath, year in zip(ModelPathGroup, newsCorpusPathList, tqdm(years)):
        # compute K-L with each sample model
        for ModelPath in tqdm(curr_year_paths):
            iternum = getIterFromName(os.path.basename(ModelPath))
            sample_index = ast.literal_eval(indices.at[iternum, year]) # convert str back to list of int
            run(KL_year_company, ModelPath, refer_corpus_path, newsCorpusPath, year, sample_index)
    
    count = getSampleTimesPerDoc(indices, years)
    # divide all values by number of bootstrap samples
    KL_year_company = KL_year_company.div(count)
    print(KL_year_company)

    refer_name = os.path.splitext(os.path.basename(refer_corpus_path))[0]
    company_name = os.path.splitext(os.path.basename(news_corpus_folder))[0].split('_')[1]
    KL_save_path = os.path.join(KL_save_folder, refer_name + "_" + company_name + "_bootstrap_analyseReportsByYear" + ".xlsx")
    KL_year_company.to_excel(KL_save_path)
    return


def isSampleName(filename):
    '''
    
    >>> import re
    >>> isSampleName("2016_iter0")
    True
    >>> isSampleName("2016_iter99.id2word")
    False

    '''
    pattern = r"^\d{4}_iter\d+$"
    match = re.match(pattern, filename)
    if match:
        return True
    else:
        return False

def getIterFromName(filename):
    pattern = r"^\d{4}_iter(\d+)$"
    iternum = re.findall(pattern, filename)[0]
    return int(iternum)


def getSampleTimesPerDoc(indices, years):
    count = pd.DataFrame(columns=years, index=range(2062))
    rows = indices.index
    for year in years:
        for row in rows:
            sample_index = ast.literal_eval(indices.at[row, year])
            for index in sample_index:
                if pd.isna(count.at[index, year]):
                    count.at[index, year] = 1
                else:
                    count.at[index, year] += 1

    return count

# computeByYear(newsModelsFolder, refer_corpus_path, news_corpus_folder, KL_save_folder)

# use a paper less related to InsurTech as reference document to validate that the index declines if topics are unrelated
# computeByYear(newsModelsFolder, refer_corpus_path2, news_corpus_folder, KL_save_folder)

# bootstrap_folder = os.environ.get("bootstrap_folder")
# bootstrap_folder_pingan = os.environ.get("bootstrapModels_pingan")    # pingan analyse reports models


# 人保
# bootstrap_folder_renbao = os.environ.get("bootstrapModels_renbao") # renbao analyse reports models
# cut_analyse_report_folder_renbao = os.environ.get("cut_analyse_report_folder_renbao")
# indices_path = os.path.join(bootstrap_folder, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_renbao, refer_corpus_path, cut_analyse_report_folder_renbao, KL_save_folder, indices_path)

# 新华
# bootstrap_folder_xinhua = os.environ.get("bootstrapModels_xinhua")
# cut_analyse_report_folder_xinhua = os.environ.get("cut_analyse_report_folder_xinhua")
# indices_path = os.path.join(bootstrap_folder_xinhua, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_xinhua, refer_corpus_path, cut_analyse_report_folder_xinhua, KL_save_folder, indices_path)

# 太保
# bootstrap_folder_taibao = os.environ.get("bootstrapModels_taibao")
# cut_analyse_report_folder_taibao = os.environ.get("cut_analyse_report_folder_taibao")
# indices_path = os.path.join(bootstrap_folder_taibao, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_taibao, refer_corpus_path, cut_analyse_report_folder_taibao, KL_save_folder, indices_path)

# 国寿
# bootstrap_folder_guoshou = os.environ.get("bootstrapModels_guoshou")
# cut_analyse_report_folder_guoshou = os.environ.get("cut_analyse_report_folder_guoshou")
# indices_path = os.path.join(bootstrap_folder_guoshou, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_guoshou, refer_corpus_path, cut_analyse_report_folder_guoshou, KL_save_folder, indices_path)

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()