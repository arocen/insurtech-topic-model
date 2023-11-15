# test and run LDA.py
import os
import LDA
from dotenv import load_dotenv
load_dotenv()

sample_corpus_path = os.environ.get("sample_corpus_path")
sample_model_save_path = os.environ.get("sample_model_save_path")

def testLDA():
    with open(sample_corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
        docs = text.split()
    model = LDA.runModel(docs, sample_model_save_path, num_topics=15)
    return

testLDA()