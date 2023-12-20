# Preprocess reference documents
import jieba
import os
from dotenv import load_dotenv

load_dotenv()

# use dictionary of news reports as the dictionary of reference document temporarily
refer_cut_dict_path = os.environ.get('dict_from_excel')
save_path = os.environ.get('cut_refer')
stopwords_path = os.environ.get('stopwords_path')

def load_stopwords(stopwords_path)->list[str]:
    with open(stopwords_path, "r", encoding="utf-8") as f:
        stopwords = f.read().splitlines()    # bug fix: added .splitlines()

    return stopwords


def cutAndSave(refer_path, dict_path=refer_cut_dict_path, save_path=save_path, stopwords_path=stopwords_path):
    '''cut reference document with jieba and save results'''

    stopwords = load_stopwords(stopwords_path)
    with open(refer_path, "r", encoding="utf-8") as f:
        text = f.read()
    if dict_path:
        # 让jieba加载自定义词典
        jieba.load_userdict(dict_path)
    tokens = [token for token in jieba.cut(text) if token not in stopwords]    # remove stopwords
    results = ' '.join(tokens)

    with open(save_path, 'w', encoding='utf-8') as f2:
        f2.write(results)
    
    return results


refer_path = os.environ.get('refer_doc_path')
refer_path2 = os.environ.get('refer_doc_path_2')
refer_path3 = os.environ.get('refer_doc_path_3')
refer_path4 = os.environ.get('refer_doc_path_4')
refer_path5 = os.environ.get('refer_doc_path_5')
cut_refer_save_path = os.environ.get('cut_refer')
cut_refer_save_path3 = os.environ.get('cut_refer3')
cut_refer_save_path4 = os.environ.get('cut_refer4')
cut_refer_save_path5 = os.environ.get('cut_refer5')
# cutAndSave(refer_path, save_path=cut_refer_save_path)
# cutAndSave(refer_path3, save_path=cut_refer_save_path3)
cutAndSave(refer_path5, save_path=cut_refer_save_path5)