# test and run newsPreprocess.py

import newsPreprocess as npre
import os
from dotenv import load_dotenv


load_dotenv()
doc_folder_path = os.environ.get('doc_folder_path')
save_folder = os.environ.get("cut_result_by_year")

def run():
    # load raw
    docs_by_year = npre.load_doc()

    filenames = sorted([f for f in os.listdir(doc_folder_path) if f.endswith(".txt")])
    years = [filename[:4] for filename in filenames]
    save_names = [year + "_cut_doc.txt" for year in years]
    save_paths = [os.path.join(save_folder, save_name) for save_name in save_names]
    
    cleaned_docs_by_year = npre.remove_nonsense(docs_by_year)
    new_cleaned_docs_by_year = npre.remove_href(cleaned_docs_by_year)

    # list of lists
    splittedDocsByYear = npre.split_docs_by_year(new_cleaned_docs_by_year)
    cutDocsByYear = []
    
    for docs, save_path in zip(splittedDocsByYear, save_paths):
        cut_docs = npre.cut(docs)
        cutDocsByYear.append(cut_docs)

        # save by year
        npre.save_results(cut_docs, save_path)
    
    print("length of cutDocsByYear:", len(cutDocsByYear))

run()