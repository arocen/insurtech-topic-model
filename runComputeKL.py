from gensim.models import LdaModel

# # Example usage
# num_topics = 10
# num_words = 1000

# Assuming model1 is trained on the reference document and model2 is trained on news reports
model1 = LdaModel(...)  # Your model trained on the reference document
model2 = LdaModel(...)  # Your model trained on news reports

kl_div = kl_divergence(model1, model2, num_topics, num_words)
print("KL Divergence:", kl_div)
