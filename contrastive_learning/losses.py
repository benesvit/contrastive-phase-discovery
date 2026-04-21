import torch 
import torch.nn.functional as F

def wms_loss(similarity, embeddings, d_alpha=25.0, d_beta= 0.2, alpha=10.0, beta=1.0, lamb=1.0, eps=0.1, sumfunction='ms', w_fun = 'exp', ms_mining=False):
    '''
    Loss function adapted from literature

    Parameters:
        similarity (torch.tensor): external similiarity (distances between order params) - for more similar examples expects larger values
        embeddings (torch.tensor): embeddings obtained from NNs
        sumfunction (string): 'ms' or 'plain'
        w_fun (string): chooses how to scale the distances tensor to cast the distances into [0, 1]

        other parameters control the shape of the functions
        
    
    '''
    embeddings = F.normalize(embeddings, p=2, dim=1)

    batch_size = embeddings.shape[0]
    # Apply mask to the distances, e.g. g(d(x_i, x_j))
    if w_fun == 'exp':
        mask_pos = torch.divide(1.0, (1.0 + torch.exp(d_alpha *(similarity - d_beta))))
        mask_neg = torch.divide(1.0, (1.0 + torch.exp(d_alpha *(d_beta - similarity))))
    elif w_fun == 'lin':
        mask_pos = similarity
        mask_neg = 1.0 - similarity # assuming we are between 0 and 1
    elif w_fun == 'tanh':
        mask_pos = torch.tanh(d_alpha*(similarity - d_beta))/2.0
        mask_neg = torch.tanh(-(d_alpha*(similarity - d_beta)))/2.0

    

    mask_pos = mask_pos.to(torch.float32) - torch.eye(batch_size, dtype=torch.float32, device=embeddings.device)
    mask_neg = mask_neg.to(torch.float32)

    
    # Cosine similarity, between 0 and 1 for angles that are acute, otherwise 0 (no similarity).
    # sim_mat = 1 - torch.matmul(embeddings, embeddings.T)
    sim_mat = torch.cdist(embeddings, embeddings)
    # sim_mat = torch.clamp(sim_mat, min=0.0)
    # sim_mat = torch.cdist(embeddings, embeddings, p=2)

    # Obtain positivity/negativity scores s+/s-
    pos_mat = torch.mul(sim_mat, mask_pos)
    neg_mat = torch.mul(sim_mat, mask_neg)

    if ms_mining:
        max_val = torch.max(neg_mat, dim=1, keepdim=True).values
        tmp_max_val = torch.max(pos_mat, axis=1, keepdim=True).values
        for_min_val = torch.multiply(sim_mat - tmp_max_val, mask_pos)
        min_val = torch.min(for_min_val,keepdim=True, dim=1).values + tmp_max_val
        mask_pos = torch.where(pos_mat < max_val + eps, mask_pos,torch.zeros_like(mask_pos))
        mask_neg = torch.where(neg_mat > min_val - eps, mask_neg, torch.zeros_like(mask_neg))

    if sumfunction == 'plain':
        # pos_exp = torch.where(mask_pos > 0.0, pos_mat, torch.zeros_like(pos_mat))
        # neg_exp = torch.where(mask_neg > 0.0, neg_mat, torch.zeros_like(neg_mat))
        pos_term = torch.sum(pos_exp, dim=1)
        neg_term = torch.sum(neg_exp, axis=1)

        loss = torch.abs(torch.mean(neg_term - pos_term))
    
    elif sumfunction == 'ms':
        pos_exp = torch.exp(-alpha*(pos_mat - lamb))
        pos_exp = torch.where(mask_pos > 0.0, pos_exp, torch.zeros_like(pos_exp))

        neg_exp = torch.exp(beta * (neg_mat - lamb))
        neg_exp = torch.where(mask_neg > 0.0, neg_exp, torch.zeros_like(neg_exp))

        pos_term = torch.log(1.0 + torch.sum(pos_exp, axis=1)) / alpha
        neg_term = torch.log(1.0 + torch.sum(neg_exp, axis=1)) / beta

        loss = torch.mean(pos_term + neg_term)

    return loss


def continuous_contrastive_loss(similarity,
                                embeddings,
                                mu = 1.0,
                                scale='sigmoid',
                                da = 10,
                                db = 0.5,
                                embedding_distance='euclidean',
                                ms_mining=False,
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

    # This is adapted from the wms loss. The idea is to let the model learn on hard negatives and weak positives.
    # In the case of soft contrastive learning they find these negatives/positives by examining the loss scores
    # and bounding them.
    if ms_mining:
        neg_max = torch.max(similarity_n, dim=1, keepdim=True).values
        pos_min = torch.min(similarity_p, axis=1, keepdim=True).values
        mask_pos = positive_loss < neg_max + eps
        mask_neg = negative_loss > pos_min - eps
        positive_loss = torch.where(mask_pos, positive_loss,torch.zeros_like(positive_loss))
        negative_loss = torch.where(mask_neg, negative_loss, torch.zeros_like(negative_loss))



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
