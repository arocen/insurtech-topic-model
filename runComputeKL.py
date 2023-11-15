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

num_topics = 15
kl_div = computeKL.kl_divergence(referModel, sampleModel, num_topics)
print("KL Divergence:", kl_div)
