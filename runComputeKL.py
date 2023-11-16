from gensim.models import LdaModel
import computeKL
from gensim import corpora
import os
from dotenv import load_dotenv

load_dotenv()

# # Example usage
# num_topics = 10
# num_words = 1000

refer_model_path = os.environ.get("refer_model_save_path")
refer_model_path2 = os.environ.get("refer_model_save_path2")
sample_model_path = os.environ.get("sample_model_save_path")
sample_corpus_path = os.environ.get("sample_corpus_path")
refer_corpus_path = os.environ.get('cut_refer')
refer_corpus_path2 = os.environ.get('cut_refer2')

def run(referModel_path, sampleModel_path, refer_corpus_path, sample_corpus_path):
    referModel = LdaModel.load(referModel_path)  # Your model trained on news reports
    sampleModel = LdaModel.load(sampleModel_path)  # Your model trained on the reference document


    with open(sample_corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
        news_corpus = text.split()

    with open(refer_corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
        reference_corpus = text.split()



    dictionary_refer = corpora.Dictionary.load(refer_model_path + ".id2word")
    dictionary_news = corpora.Dictionary.load(sample_model_path + ".id2word")

    num_topics = 15
    kl_div = computeKL.kl_divergence(referModel, sampleModel, num_topics, reference_corpus, news_corpus, dictionary_refer, dictionary_news)
    print("KL Divergence:", kl_div)
    return kl_div

run(refer_model_path, sample_model_path, refer_corpus_path, sample_corpus_path)
run(refer_model_path2, sample_model_path, refer_corpus_path2, sample_corpus_path)