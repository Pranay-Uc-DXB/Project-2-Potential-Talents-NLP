# %%
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# %%
data=pd.read_csv(r"../Data/potential-talents - Aspiring human resources - seeking human resources.csv")

Data_SBert2=data.copy()

# %%
with open('Sbert_model.pkl','rb') as file:
    Script=pickle.load(file)

# %%
from Train_model import string_to_sbert_embedding

with open('Sbert_Cosine_func.pkl','rb') as file:
    string_to_sbert_embedding= pickle.load(file)

# %%
Query= "Aspiring Human Resources"

#Converting Query and job_titles into embeddings 
Query_mean_pool=string_to_sbert_embedding(Query)
Data_SBert2['fit']=Data_SBert2['job_title'].apply(lambda x: string_to_sbert_embedding(str(x)))

# %%
def Sim_Score_Cal(df_with_embed,Query_embed):
    df_with_embed['sim_score']=df_with_embed['fit'].apply(lambda x: cosine_similarity(Query_embed,x))
    df_with_embed['sim_score']=df_with_embed['sim_score'].apply(lambda x: x[0][0])
    df_with_embed=df_with_embed.sort_values(by='sim_score',ascending=False)
    df_with_embed=df_with_embed.drop_duplicates(subset='job_title')
    df_with_embed=df_with_embed.reset_index().drop(['id','index'],axis=1)
    
    return df_with_embed

# %%
# Applying SBERT Model to retrieve results and display top 10 picks 

SBert2_results=Sim_Score_Cal(Data_SBert2,Query_mean_pool)
SBert2_results.iloc[:10]

# %%
from Train_model import Sum_New_keyword_embed

with open('Reranker_algo_sum_method.pkl','rb') as file:
    Sum_New_keyword_embed=pickle.load(file)

# %%
from Train_model import Reranker

with open('Reranker_func.pkl','rb') as file:
    Reranker=pickle.load(file)


# %%
# Selecting/Starring Candidate with index 1 from the above results and re-ranking in conjunction with the below Query    
Query= "Aspiring Human Resources"
Reranker(1,Query, SBert2_results,Sum_New_keyword_embed,10)        