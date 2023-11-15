# Construct InsurTech Index using topic model (LDA) and K-L divergence



corpus:
1. Annual reports
2. News reports
3. Analysis reports
   

To-do of News reports:
- [x] Preprocess news corpus.
- [ ] Run LDA on news reports by year with topic number K, get and save word distributions of each topic.
- [ ] Run LDA on a reference document about InsurTech(Q: with what topic number?), get and save word distributions.
- [ ] Calculate K-L divergence between distributions of reference document and those of news reports. (Q: How to match the requirements that probabilities sums to 1? How to handle the difference of vocabularies (BOW) between reference document and news reports)

test
- [ ] Load saved gensim LDA model, id2word, etc.

other thoughts:
- DTM for calculating multi-year word distributions?
- How to obtain other corpus?