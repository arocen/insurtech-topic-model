# Construct InsurTech Index using topic model (LDA) and K-L divergence



corpus:
1. Annual reports
2. News reports
3. Analysis reports
   

To-do of News reports:
- [x] Preprocess news corpus.
- [x] Run LDA on news reports by year with topic number K, get and save word distributions of each topic.
- [x] Run LDA on a reference document about InsurTech(Q: with what topic number?), get and save word distributions.
- [x] Calculate K-L divergence between distributions of reference document and those of news reports. (Q: How to match the requirements that probabilities sums to 1? How to handle the difference of vocabularies (BOW) between reference document and news reports)


## implement K-L divergence

Input: topic distributions of 2 documents

One is reference document, another is a news document.

Topics that is used to predict the topic distributions of reference document, are are derived from annual reports (or news reports).

> We provide the means of 100 bootstrap samples that we obtain by repeatedly sampling 90% of the annual reports at random and applying the LDA method with 45 topics. Based on the topics obtained in each bootstrap sample, we derive the distribution of topics in the various reference documents that have not entered into the estimation process of the LDA. The bands provide 95% confidence intervals for the mean computed based on the bootstrap samples.
> ———— Estimating the relation between digitalization and the market value of insurers

test
- [x] Load saved gensim LDA model, id2word, etc.

optimization
- [x] Load stopwords list when cutting with jieba
- [x] Compute K-L divergence by year
- [ ] Linearly scaled K-L divergence the interval [0, 1]
- [ ] bootstrap sample
- [ ] get company or province level measure
- [ ] use metric other than K-L divergence
- [ ] write code with gensim wraper of mallet



method 1: Take word distributions of topics as input of K-L divergence. Determine an Insurtech topic according to values of K-L divergence. Choose the smallest K-L divergence as the measure of InsurTech.

method 2: Take topic distributions of documents as input of K-L divergence. Then we can measure InsurTech according to values of K-L divergence.

other thoughts:
- DTM for calculating multi-year word distributions?
- How to obtain other corpus?
- Adjust topic numbers