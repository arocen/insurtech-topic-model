# Run LDA with single report in 1 year and calculate KL dicergence.

from dotenv import load_dotenv
import os
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

load_dotenv()


cut_renbao_2015 = os.environ.get("cut_renbao_2015")
cut_renbao_2016 = os.environ.get("cut_renbao_2016")
model_save_folder_2015 = os.environ.get("renbao_2015_model")
model_save_folder_2016 = os.environ.get("renbao_2016_model")

def singleDocLdaKL(doc_path, model_save_folder, model_save_name, num_topics=15):
    with open(doc_path, "r", encoding="utf-8") as f:
        doc = f.read()
    tokenized_data = [simple_preprocess(doc)]
    dictionary = corpora.Dictionary(tokenized_data)

    # save dictionary, add a suffix to filename
    dictionary.save(model_save_folder + ".dictionary")

    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_data]

    lda_model = LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary)
    lda_model.save(os.path.join(model_save_folder, model_save_name))
    return


# 人保2015
# singleDocLdaKL(cut_renbao_2015, model_save_folder_2015, "2015")

# 人保2016
singleDocLdaKL(cut_renbao_2016, model_save_folder_2016, "2016")