from gensim.models import LdaModel
import computeKL
import os
from dotenv import load_dotenv

load_dotenv()

# # Example usage
# num_topics = 10
# num_words = 1000

refer_model_path = os.environ.get("refer_model_save_path")
sample_model_path = os.environ.get("sample_model_save_path")

# Assuming model1 is trained on the reference document and model2 is trained on news reports
referModel = LdaModel.load(refer_model_path)  # Your model trained on news reports
sampleModel = LdaModel.load(sample_model_path)  # Your model trained on the reference document

sample_corpus_path = os.environ.get("sample_corpus_path")
with open(sample_corpus_path, "r", encoding="utf-8") as f:
    text = f.read()
    news_corpus = text.split()

refer_corpus_path = os.environ.get('cut_refer')
with open(refer_corpus_path, "r", encoding="utf-8") as f:
    text = f.read()
    reference_corpus = text.split()


num_topics = 15
kl_div = computeKL.kl_divergence(referModel, sampleModel, num_topics, reference_corpus, news_corpus)
print("KL Divergence:", kl_div)
