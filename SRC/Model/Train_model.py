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


