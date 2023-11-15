# Preprocess reference documents
import jieba
import os
from dotenv import load_dotenv

load_dotenv()

# use dictionary of news reports as the dictionary of reference document temporarily
refer_cut_dict_path = os.environ.get('dict_from_excel')
save_path = os.environ.get('cut_refer')

def cutAndSave(refer_path, dict_path=refer_cut_dict_path, save_path=save_path):
    '''cut reference document with jieba and save results'''
    with open(refer_path, "r", encoding="utf-8") as f:
        text = f.read()

    if dict_path:
        # 让jieba加载自定义词典
        jieba.load_userdict(dict_path)
    tokens = [token for token in jieba.cut(text)]    # To-do: consider removing stopwords
    results = ' '.join(tokens)

    with open(save_path, 'w', encoding='utf-8') as f2:
        f2.write(results)
    
    return results



refer_path = os.environ.get('refer_doc_path')
cutAndSave(refer_path)