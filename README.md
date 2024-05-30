# Project 2 - Potential Talents

## Introduction
This project caters to the Human Resources and Staffing Industry. A Talent sourcing and management company wants to automate their hiring process and build a recommendation engine for spotting potential candidates that could fit the role.

What business problem does the project aim to solve?

 The project aims to address and solve the following issues:
- Spot top talent using ML's latest NLP algorithms and rank them based on their fit for the required role.
- Enable the HR department to select candidates based on their preference from provided recommendations and refine the search further using the Re-ranking algorithm
- Automate talent spotting process
- Eliminate human bias from the hiring process

## Methodology
In this project,various NLP models were tested from the basic, Bag-of-words model, to advanced transformer-based algorithms. I used couple of simple models to establish baseline performance to compare them with word-to-vector representing algorithms and transformer-based algorithms. 

Below are the models that were used and tested to obtain similarity scores between user-defined positions and positions available in raw data:

1) Bag of words
2) TFF-IDF
3) Word-2Vec
4) FaastText
5) GloVe
6) BERT
7) SBERT

## Conclusion  
https://github.com/Pranay-Uc-DXB/Project-2-Q6O5w5YaaeJOdoUH/assets/62109186/55bc727b-5c12-4349-8b13-b62784f46c60


Similarity scores, obtained by using cosine similarity between user-defined vectors v/s closest vector available in raw data, were used to return similar text or job title results. From my analysis, I saw FastText and SBERT performed really well on the dataset. There was 52 ppt improvement when compared to basemodels such TF-IDF and word-2-Vec models. Because this project only catered to finding similar job titles, I would recommend using FastText. However, if the application needs to be extended in capturing semantic context on an entire resume then I would recommend using SBERT.

In my project, I used transformer-based, SBERT architecture, to not only obtain similarity scores, but to also use it as a part of search and reranking algorithm to fine-tune user-defined search for better recommendations. And as expected, by adding more context, our similarity scores improved quite considerably. Nevertheless, I used Streamlit to deploy my app for demo purposes. 


