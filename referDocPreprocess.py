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
refer_path6 = os.environ.get('refer_doc_path_6')
refer_path7 = os.environ.get('refer_doc_path_7')
refer_path8 = os.environ.get('refer_doc_path_8')
refer_path9 = os.environ.get('refer_doc_path_9')
refer_path10 = os.environ.get('refer_doc_path_10')
refer_path11 = os.environ.get('refer_doc_path_11')
cut_refer_save_path = os.environ.get('cut_refer')
cut_refer_save_path3 = os.environ.get('cut_refer3')
cut_refer_save_path4 = os.environ.get('cut_refer4')
cut_refer_save_path5 = os.environ.get('cut_refer5')
cut_refer_save_path6 = os.environ.get('cut_refer6')
cut_refer_save_path7 = os.environ.get('cut_refer7')
cut_refer_save_path8 = os.environ.get('cut_refer8')
cut_refer_save_path9 = os.environ.get('cut_refer9')
cut_refer_save_path10 = os.environ.get('cut_refer10')
# cutAndSave(refer_path, save_path=cut_refer_save_path)
# cutAndSave(refer_path3, save_path=cut_refer_save_path3)
# cutAndSave(refer_path6, save_path=cut_refer_save_path6)
# cutAndSave(refer_path7, save_path=cut_refer_save_path7)
# cutAndSave(refer_path8, save_path=cut_refer_save_path8)
# cutAndSave(refer_path9, save_path=cut_refer_save_path9)
# cutAndSave(refer_path10, save_path=cut_refer_save_path10)