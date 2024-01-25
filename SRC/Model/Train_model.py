# %%
import pickle
import torch
import transformers
from sklearn.metrics.pairwise import cosine_similarity

# %%
model_name= 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1' 

# %%
with open ('Sbert_model.pkl','wb') as file:
    pickle.dump(model_name,file)

# %%
SBert_model=transformers.AutoModel.from_pretrained(model_name)
Tokenizer=transformers.AutoTokenizer.from_pretrained(model_name)

# %%
def  string_to_sbert_embedding(string):
    tokens_to_encode=Tokenizer.encode_plus(string, add_special_tokens=True, return_tensors='pt')
    output=SBert_model(**tokens_to_encode)
    #print('last_hidden_state: ',output.last_hidden_state.shape)
    mean_pool=torch.mean(output.last_hidden_state,dim=1)
    mean_pool=mean_pool.detach().numpy()
    #print('mean_pool: ',mean_pool.shape)
    return mean_pool

# %%
with open('Sbert_Cosine_func.pkl','wb') as file:
    pickle.dump(string_to_sbert_embedding,file)

# %%
def Sum_New_keyword_embed(Selected_Query_Index,Query,model_results):
    
    New_keyword_embed=string_to_sbert_embedding(Query+' '+model_results.iloc[Selected_Query_Index,0])
    return New_keyword_embed

# %%
with open('Reranker_algo_sum_method.pkl','wb') as file:
    pickle.dump(Sum_New_keyword_embed,file)

# %%
def Reranker(Selected_Query_Index, Query, model_results, reranking_technique,Top_n):
    
    New_keyword_embed_= reranking_technique(Selected_Query_Index, Query, model_results)
    
    model_results['sim_score']=model_results['fit'].apply(lambda x: cosine_similarity(New_keyword_embed_,x))
    model_results['sim_score']=model_results['sim_score'].apply(lambda x: x[0][0])
    model_results=model_results.sort_values(by='sim_score',ascending=False)
    model_results=model_results.drop_duplicates(subset='job_title')
    model_results=model_results.reset_index().drop(['index','fit'],axis=1)
    model_results=model_results.iloc[:Top_n]
    return model_results


# %%
with open('Reranker_func.pkl','wb') as file:
    pickle.dump(Reranker,file)