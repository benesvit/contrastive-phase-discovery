import torch 
import torch.nn.functional as F


def continuous_contrastive_loss(similarity,
                                embeddings,
                                mu = 1.0,
                                scale='sigmoid',
                                da = 10,
                                db = 0.5,
                                embedding_distance='euclidean',
                                eps=0.01,):
    
    if embedding_distance == 'euclidean':
        emb_dist = torch.cdist(embeddings, embeddings)
    elif embedding_distance == 'cosine':
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        emb_dist = 1 - torch.matmul(embeddings_norm,embeddings_norm.T)


    if scale == 'sigmoid':
        similarity_p = torch.divide(1,1+torch.exp(-da*(similarity - db)))
        similarity_n = torch.divide(1,1+torch.exp(-da*(db-similarity)))
    elif scale == 'lin':
        similarity_p = similarity
        similarity_n = 1 - similarity
    elif scale =='tanh':
        similarity_p = torch.tanh(da*(similarity - db))/2.0 + 0.5
        similarity_n = torch.tanh(-(da*(similarity - db)))/2.0 + 0.5
    
    positive_loss = similarity_p * torch.pow(emb_dist, 2)
    negative_loss = similarity_n * torch.pow(torch.clamp(mu - emb_dist, min=0.0), 2)



    total_loss = positive_loss + negative_loss
    # Vyhneme se dvojímu počítání a diagonále:    
    mask = torch.triu(torch.ones_like(total_loss), diagonal=1)
    loss = total_loss * mask
    
    return loss.sum() / mask.sum()



def Q_distance(Q,  c_max = 1.0, c_min = -1.0):
    Q = F.normalize(Q, p=2, dim=1)
    cos_sim = torch.matmul(Q, Q.T)
    
    cos_dist = (cos_sim - c_min) / (c_max - c_min)
    
    return cos_dist
