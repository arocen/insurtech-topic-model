# Load txt files by company and by year, cut into words.

import newsPreprocess as npre
import analyseReports2Txt as ar2t
import os
from dotenv import load_dotenv

load_dotenv()

doc_div_chars = os.environ.get("doc_div_chars")

# 平安
analyse_report_folder_pingan = os.environ.get("analyse_report_folder_pingan")
cut_analyse_report_folder_pingan = os.environ.get("cut_analyse_report_folder_pingan")

# 人保
analyse_report_folder_renbao = os.environ.get("analyse_report_folder_renbao")
cut_analyse_report_folder_renbao = os.environ.get("cut_analyse_report_folder_renbao")

# 新华
analyse_report_folder_xinhua = os.environ.get("analyse_report_folder_xinhua")
cut_analyse_report_folder_xinhua = os.environ.get("cut_analyse_report_folder_xinhua")

# 国寿
analyse_report_folder_guoshou = os.environ.get("analyse_report_folder_guoshou")
cut_analyse_report_folder_guoshou = os.environ.get("cut_analyse_report_folder_guoshou")

# 太保
analyse_report_folder_taibao = os.environ.get("analyse_report_folder_taibao")
cut_analyse_report_folder_taibao = os.environ.get("cut_analyse_report_folder_taibao")


def cutRawByYear(parent_folder:str, save_folder:str)->list[list[str]]:
    '''
    Return list of lists. 

    Use newsPreprocess.load_preprocessed_multi_corpus(save_folder) to load cut results.
    Use newsPreprocess.getYearFromFilename(save_folder) to get years.
    '''
    child_directories = ar2t.get_child_directories(parent_folder)

    cutReportsByYear = []
    for directory in child_directories:
        txts = sorted([f for f in os.listdir(directory) if f.endswith(".txt")])

        # Get full paths of txt
        txts = [os.path.join(directory, f) for f in txts]
        
        reportsIn1year = []
        for txt in txts:
            # cut into words
            with open(txt, 'r', encoding="utf-8") as file:
                report = file.read()
                reportsIn1year.append(report)
        cut_reports = npre.cut(reportsIn1year)
        cutReportsByYear.append(cut_reports)

    # Save results
    years = sorted([os.path.basename(child_directory) for child_directory in child_directories])
    save_paths = [os.path.join(save_folder, year + "_cut_analyseReports.txt") for year in years]
    for cutReportsIn1year, save_path in zip(cutReportsByYear, save_paths):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write((doc_div_chars.join(cutReportsIn1year)))

    return cutReportsByYear

# cutRawByYear(analyse_report_folder_pingan, cut_analyse_report_folder_pingan)
# cutRawByYear(analyse_report_folder_renbao, cut_analyse_report_folder_renbao)
# cutRawByYear(analyse_report_folder_xinhua, cut_analyse_report_folder_xinhua)
# cutRawByYear(analyse_report_folder_guoshou, cut_analyse_report_folder_guoshou)
# cutRawByYear(analyse_report_folder_taibao, cut_analyse_report_folder_taibao)