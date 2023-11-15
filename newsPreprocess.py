# Preprocess nes reports about InsurTech
import os
from dotenv import load_dotenv
import re
import jieba
from tqdm import tqdm

# load .env file
load_dotenv()

doc_div_chars = os.environ.get("doc_div_chars")
cut_results = os.environ.get("cut_result_by_year")
split_pattern = os.environ.get("split_pattern")
uselessText_path = os.environ.get("uselessText_path")

def load_doc(doc_folder_path:str=os.environ.get('doc_folder_path'))->list[str]:
    '''
    读取各年度文档
    返回一个列表，其中每一个元素表示一个年度内的所有文档

    folder_path: 所有txt文件所在文件夹
    '''
    # Get a list of text files in the folder and sort them by filename

    txt_files = sorted([f for f in os.listdir(doc_folder_path) if f.endswith(".txt")])
    
    docs_by_year = []
    for filename in txt_files:
        file_path = os.path.join(doc_folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            docs_by_year.append(content)

    return docs_by_year



def load_uselessText(uselessText_path:str=os.environ.get("uselessText_path"))->list[str]:
    with open(uselessText_path, "r", encoding="utf-8") as f:
        whole = f.read()
        uselessText = whole.split("\n\n")
    return uselessText

def remove_nonsense(docs_by_year:list[str])->list[str]:
    '''remove useless text'''

    uselessText = load_uselessText()
    for i in range(len(docs_by_year)): # use list index to change value of list items
        for text in uselessText:
            docs_by_year[i] = docs_by_year[i].replace(text, "") # strings are immutable
    
    return docs_by_year

def remove_href(docs_by_year:list[str]):
    '''remove href link'''
    pat = "(文字快照：.*?\n)"
    for i in range(len(docs_by_year)):
        docs_by_year[i] = re.sub(pat, "", docs_by_year[i])
    return docs_by_year


def split_docs_by_year(docs_by_year:list[str], split_pattern:str=split_pattern)->list[list[str]]:
    '''
    切分每个年度内的文档
    返回一个列表，其中每一个元素为一个列表，表示一个年度内的所有文档

    split_pattern: 用于文档切分的pattern
    '''
    splittedDocsByYear = []
    for docs in docs_by_year:
        doc_list = re.split(split_pattern, docs)
        doc_list = doc_list[:-1]    # remove the last part
        splittedDocsByYear.append(doc_list)  # use append instead of extend
    return splittedDocsByYear

def cut(docs:list[str], my_dict_path:str=os.environ.get('dict_from_excel'))->list[str]:
    '''
    将列表中的元素切分为词语，返回列表, 列表中每一个元素为一个分词后的句子

    my_dict_path: 自定义词典路径
    '''
    if my_dict_path:
        # 让jieba加载自定义词典
        jieba.load_userdict(my_dict_path)
    
    cut_docs = []
    print("Cutting docs with jieba:")
    for doc in tqdm(docs):
        tokens = [token for token in jieba.cut(doc)]    # To-do: consider removing stopwords
        result = ' '.join(tokens)
        cut_docs.append(result)
    print("Cutting completed!")
    return cut_docs

def save_results(cut_docs:list[str], save_path=cut_results):
    '''
    Save the cut docs, documents are divided by doc_div_chars.
    '''
    try:
        with open(save_path, "w", encoding='utf-8') as f:
            f.write(doc_div_chars.join(cut_docs)) # python's .writelines() does not add \n
    except:
        print(f"Error saving cut_docs to {save_path}")
    return

def load_preprocessed_multi_corpus(folder_path=cut_results)->list[list[str]]:
    '''read documents which are divided by doc_div_chars'''
    corpusByYear = []
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            corpus = f.read()
            docs = corpus.split(doc_div_chars)
            corpusByYear.append(docs)
    return corpusByYear

def getYearFromFilename(folder_path=cut_results)->list[str]:
    '''return years of corpus list according to names of txt files'''
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
    years = [filename[:4] for filename in filenames]
    return years
