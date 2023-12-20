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
import runLDA
from itertools import combinations

load_dotenv()

# # Example usage
# num_topics = 10
# num_words = 1000



def run(df, model_path, refer_corpus_path, news_path, year, sample_index):
    model = LdaModel.load(model_path)  # Your model trained on news reports

    with open(news_path, "r", encoding="utf-8") as f:
        text = f.read()
        news_corpus = text.split(doc_div_chars)

    with open(refer_corpus_path, "r", encoding="utf-8") as f:
        reference_corpus = f.read()


    dictionary_news = corpora.Dictionary.load(model_path + ".id2word")

    computeKL.kl_divergence(df, model, reference_corpus, news_corpus, dictionary_news, year, sample_index)
    return






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

def computeBootstrapByYear(newsModelsFolder, refer_corpus_path, news_corpus_folder, KL_save_folder, indices_path, doc_num=100):

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
    KL_year_company = pd.DataFrame(columns=years, index=range(doc_num)) # set index with the range of most reports in each year
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

def computeAllCorpus(refer_corpus_path, modelsFolder=os.environ.get("bootstrapModelAllAnalyseReports"), KL_save_folder=os.environ.get("KL_save_folder")):
    '''Compute KL divergence based on LDA models bootstraped from all analyse reports.'''
    # All analyse reports
    bootstrap_folder_all = os.environ.get("bootstrapModelAllAnalyseReports")
    all_corpus = runLDA.loadAllCorpus()
    indices_path = os.path.join(bootstrap_folder_all, "indices.xlsx")
    indices = pd.read_excel(indices_path)


    modelnames = [f for f in os.listdir(modelsFolder) if isModelNameWithoutYear(f)]
    with open(refer_corpus_path, "r", encoding="utf-8") as f:
        reference_corpus = f.read()
    
    # Loop models
    KL_df = pd.DataFrame(columns=["all_corpus"], index=range(1200))
    for modelname in tqdm(modelnames):
        model_path = os.path.join(bootstrap_folder_all, modelname)
        iternum = getIterFromNameWithoutYear(modelname)
        sample_index = ast.literal_eval(indices.at[iternum, "all_corpus"])
        model = LdaModel.load(model_path)
        dictionary_reports = corpora.Dictionary.load(model_path + ".id2word")

        # Compute KL
        computeKL.kl_divergence_without_year(KL_df, model, reference_corpus, all_corpus, dictionary_reports, sample_index, column_name="all_corpus")
    
    # divide all values by number of bootstrap samples
    count = getSampleTimesPerDocWithoutYear(indices)
    KL_df = KL_df.div(count)
    print(KL_df)

    refer_name = os.path.splitext(os.path.basename(refer_corpus_path))[0]
    KL_save_path = os.path.join(KL_save_folder, refer_name + "_" + "all_analyse_reports_bootstrap_sample.xlsx")
    KL_df.to_excel(KL_save_path)
    return


def computeAllCorpusSimpleAverage(refer_corpus_path, modelsFolder=os.environ.get("bootstrapModelAllAnalyseReports"), KL_save_folder=os.environ.get("KL_save_folder")):
    '''Compute average KL with another method.'''
        # All analyse reports
    bootstrap_folder_all = os.environ.get("bootstrapModelAllAnalyseReports")
    all_corpus = runLDA.loadAllCorpus()
    indices_path = os.path.join(bootstrap_folder_all, "indices.xlsx")
    indices = pd.read_excel(indices_path)


    modelnames = [f for f in os.listdir(modelsFolder) if isModelNameWithoutYear(f)]
    with open(refer_corpus_path, "r", encoding="utf-8") as f:
        reference_corpus = f.read()
    
    # Loop models
    KL_df = pd.DataFrame(columns=["all_corpus"], index=range(100))
    for i, modelname in tqdm(enumerate(modelnames)):
        model_path = os.path.join(bootstrap_folder_all, modelname)
        iternum = getIterFromNameWithoutYear(modelname)
        sample_index = ast.literal_eval(indices.at[iternum, "all_corpus"])
        model = LdaModel.load(model_path)
        dictionary_reports = corpora.Dictionary.load(model_path + ".id2word")

        # Compute KL
        computeKL.kl_divergence_without_year_simple_average(iternum, KL_df, model, reference_corpus, all_corpus, dictionary_reports, sample_index, column_name="all_corpus")
    
    print(KL_df)

    refer_name = os.path.splitext(os.path.basename(refer_corpus_path))[0]
    KL_save_path = os.path.join(KL_save_folder, refer_name + "_" + "all_analyse_reports_bootstrap_sample_simple_average.xlsx")
    KL_df.to_excel(KL_save_path)
    return


def isModelNameWithoutYear(filename):
    '''
    Check if a filename is the format of LDA models based on all analyse reports.
    Helper function of computeAllCorpus().

    >>> import re
    >>> isModelNameWithoutYear("2016_iter0")
    False
    >>> isModelNameWithoutYear("iter99")
    True
    >>> isModelNameWithoutYear("iter23.id2word")
    False
    '''
    pattern = r"^iter\d+$"
    match = re.match(pattern, filename)
    if match:
        return True
    else:
        return False
    
def getIterFromNameWithoutYear(filename):
    '''
    Get iterate number from file name of LDA models based on all analyse reports.
    Helper function of computeAllCorpus().

    >>> import re
    >>> getIterFromNameWithoutYear("iter99")
    99
    >>> getIterFromNameWithoutYear("iter20")
    22
    '''

    pattern = r"^iter(\d+)$"
    iternum = re.findall(pattern, filename)[0]
    return int(iternum)

def getSampleTimesPerDocWithoutYear(indices, column_label="all_corpus"):
    count = pd.DataFrame(columns=[column_label], index=range(1200))
    rows = indices.index
    for row in rows:
        sample_index = ast.literal_eval(indices.at[row, column_label])
        for index in sample_index:
            if pd.isna(count.at[index, column_label]):
                count.at[index, column_label] = 1
            else:
                count.at[index, column_label] += 1

    return count


def run_kl_divergence_mutal_refer(refer_corpus_path_list:list[str], modelsFolder=os.environ.get("bootstrapModelAllAnalyseReports"), KL_save_folder=os.environ.get("KL_save_folder")):
    
    bootstrap_folder_all = os.environ.get("bootstrapModelAllAnalyseReports")
    modelnames = [f for f in os.listdir(modelsFolder) if isModelNameWithoutYear(f)]

    
    # Save refer name and refer corpus in a dictionary
    reference_corpus = dict()
    for refer_path in refer_corpus_path_list:
        with open(refer_path, "r", encoding="utf-8") as f:
            refer_name = os.path.basename(refer_path).split(".")[0]
            reference_corpus[refer_name] = f.read()
    
    refer_combinations = get_refer_combinations(reference_corpus.keys())
    print(refer_combinations)
    column_names = [refer_pair[0][4:] + " vs " + refer_pair[1][4:] for refer_pair in refer_combinations]
    KL_df = pd.DataFrame(columns=column_names, index=range(100))
    for i, modelname in tqdm(enumerate(modelnames)):
        model_path = os.path.join(bootstrap_folder_all, modelname)
        model = LdaModel.load(model_path)
        dictionary_reports = corpora.Dictionary.load(model_path + ".id2word")
        iternum = getIterFromNameWithoutYear(modelname)

        # Compute KL
        for combination in refer_combinations:
            column_name = combination[0][4:] + " vs " + combination[1][4:]
            computeKL.kl_divergence_mutal_refer(iternum, KL_df, model, reference_corpus[combination[0]], reference_corpus[combination[1]], dictionary_reports, column_name)
    
    print(KL_df)

    KL_save_path = os.path.join(KL_save_folder, "all_analyse_reports_bootstrap_sample_mutal_refer.xlsx")
    KL_df.to_excel(KL_save_path)
    return


def get_refer_combinations(reference_corpus:list[str])->list[tuple]:
    '''Get non-repeating combinations of 2 reference documents from a list.'''
    return list(combinations(reference_corpus, 2))




refer_corpus_path = os.environ.get('cut_refer')    # corpus about InsurTech
refer_corpus_path2 = os.environ.get('cut_refer2')  # corpus about insurance instead of Insurtech
refer_corpus_path3 = os.environ.get("cut_refer3")  # another corpus about InsurTech
refer_corpus_path4 = os.environ.get("cut_refer4")  # another corpus about InsurTech
refer_corpus_path5 = os.environ.get('cut_refer5')  # corpus about insurance
doc_div_chars = os.environ.get("doc_div_chars")

newsModelsFolder = os.environ.get("modelByYear_folder")
news_corpus_folder = os.environ.get("cut_result_by_year")
KL_save_folder = os.environ.get("KL_save_folder")



# computeByYear(newsModelsFolder, refer_corpus_path, news_corpus_folder, KL_save_folder)

# use a paper less related to InsurTech as reference document to validate that the index declines if topics are unrelated
# computeByYear(newsModelsFolder, refer_corpus_path2, news_corpus_folder, KL_save_folder)

# bootstrap_folder = os.environ.get("bootstrap_folder")
# bootstrap_folder_pingan = os.environ.get("bootstrapModels_pingan")    # pingan analyse reports models


KL_save_folder_refer3_by_company = os.environ.get("KL_save_folder_refer3_by_company")


# Pay attention to the refer corpus path used when running.
# # 人保
# bootstrap_folder_renbao = os.environ.get("bootstrapModels_renbao") # renbao analyse reports models
# cut_analyse_report_folder_renbao = os.environ.get("cut_analyse_report_folder_renbao")
# indices_path = os.path.join(bootstrap_folder_renbao, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_renbao, refer_corpus_path3, cut_analyse_report_folder_renbao, KL_save_folder_refer3_by_company, indices_path)

# # 新华
# bootstrap_folder_xinhua = os.environ.get("bootstrapModels_xinhua")
# cut_analyse_report_folder_xinhua = os.environ.get("cut_analyse_report_folder_xinhua")
# indices_path = os.path.join(bootstrap_folder_xinhua, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_xinhua, refer_corpus_path3, cut_analyse_report_folder_xinhua, KL_save_folder_refer3_by_company, indices_path)

# # 太保
# bootstrap_folder_taibao = os.environ.get("bootstrapModels_taibao")
# cut_analyse_report_folder_taibao = os.environ.get("cut_analyse_report_folder_taibao")
# indices_path = os.path.join(bootstrap_folder_taibao, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_taibao, refer_corpus_path3, cut_analyse_report_folder_taibao, KL_save_folder_refer3_by_company, indices_path)

# # 国寿
# bootstrap_folder_guoshou = os.environ.get("bootstrapModels_guoshou")
# cut_analyse_report_folder_guoshou = os.environ.get("cut_analyse_report_folder_guoshou")
# indices_path = os.path.join(bootstrap_folder_guoshou, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_guoshou, refer_corpus_path3, cut_analyse_report_folder_guoshou, KL_save_folder_refer3_by_company, indices_path)

# # 平安
# bootstrap_folder_pingan = os.environ.get("bootstrapModels_pingan")
# cut_analyse_report_folder_pingan = os.environ.get("cut_analyse_report_folder_pingan")
# indices_path = os.path.join(bootstrap_folder_pingan, "indices.xlsx")
# computeBootstrapByYear(bootstrap_folder_pingan, refer_corpus_path3, cut_analyse_report_folder_pingan, KL_save_folder_refer3_by_company, indices_path)


# All analyse reports
# computeAllCorpus(refer_corpus_path)

# All analyse reports simple average
# computeAllCorpusSimpleAverage(refer_corpus_path)
# computeAllCorpusSimpleAverage(refer_corpus_path3)

# Use models of 45 topics
# computeAllCorpusSimpleAverage(refer_corpus_path, modelsFolder=os.environ.get("bootstrapModelAllAnalyseReports45topics"), KL_save_folder=os.environ.get("KL_save_folder_45topics"))
# computeAllCorpusSimpleAverage(refer_corpus_path2, modelsFolder=os.environ.get("bootstrapModelAllAnalyseReports45topics"), KL_save_folder=os.environ.get("KL_save_folder_45topics"))
# computeAllCorpusSimpleAverage(refer_corpus_path3, modelsFolder=os.environ.get("bootstrapModelAllAnalyseReports45topics"), KL_save_folder=os.environ.get("KL_save_folder_45topics"))
# computeAllCorpusSimpleAverage(refer_corpus_path4, modelsFolder=os.environ.get("bootstrapModelAllAnalyseReports45topics"), KL_save_folder=os.environ.get("KL_save_folder_45topics"))
# computeAllCorpusSimpleAverage(refer_corpus_path5, modelsFolder=os.environ.get("bootstrapModelAllAnalyseReports45topics"), KL_save_folder=os.environ.get("KL_save_folder_45topics"))

# Use average per doc instead
# computeAllCorpus(refer_corpus_path, KL_save_folder=os.environ.get("KL_save_folder_average_per_doc"))
# computeAllCorpus(refer_corpus_path2, KL_save_folder=os.environ.get("KL_save_folder_average_per_doc"))
# computeAllCorpus(refer_corpus_path3, KL_save_folder=os.environ.get("KL_save_folder_average_per_doc"))
# computeAllCorpus(refer_corpus_path4, KL_save_folder=os.environ.get("KL_save_folder_average_per_doc"))
# computeAllCorpus(refer_corpus_path5, KL_save_folder=os.environ.get("KL_save_folder_average_per_doc"))

# Refer combinations
refer_path_list = [refer_corpus_path, refer_corpus_path2, refer_corpus_path3, refer_corpus_path4, refer_corpus_path5]
run_kl_divergence_mutal_refer(refer_path_list)

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()