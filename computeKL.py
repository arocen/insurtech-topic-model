# compute K-L divergence with gensim
# source: ChatGPT


from gensim import similarities
from gensim.corpora import Dictionary
import numpy as np

def kl_divergence(model1, model2, num_topics):
    """
    Calculate Kullback-Leibler divergence between two LDA models.

    Parameters:
    - model1, model2: LdaModel instances
    - num_topics: Number of topics in the models
    - num_words: Number of words to consider for each topic

    Returns:
    - kl_div: Kullback-Leibler divergence
    """
    # Create a common dictionary for both models
    common_dictionary = Dictionary([model1.id2word.doc2bow([]), model2.id2word.doc2bow([])])

    # Create a similarity index using model1's topics
    # similarity_index = similarities.MatrixSimilarity(model1.get_topics())

    # Get the topic distributions for the reference document
    reference_doc = "Your reference document here"
    reference_bow = common_dictionary.doc2bow(reference_doc.split())
    reference_topics = model1.get_document_topics(reference_bow)

    # Get the topic distributions for the news report
    news_report = "Your news report here"
    report_bow = common_dictionary.doc2bow(news_report.split())
    report_topics = model2.get_document_topics(report_bow)

    # Create arrays for the topic distributions
    reference_distribution = np.zeros(num_topics)
    report_distribution = np.zeros(num_topics)

    # Fill the arrays with probabilities
    for topic, prob in reference_topics:
        reference_distribution[topic] = prob

    for topic, prob in report_topics:
        report_distribution[topic] = prob

    # Calculate the KL divergence
    kl_div = sum(reference_distribution[i] * np.log(reference_distribution[i] / report_distribution[i])
                 for i in range(num_topics) if reference_distribution[i] > 0)

    return kl_div

