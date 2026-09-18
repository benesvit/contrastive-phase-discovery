import torch 
import torch.nn.functional as F


def continuous_contrastive_loss(similarity,
                                embeddings,
                                mu = 1.0,
                                scale='sigmoid',
                                da = 10,
                                db = 0.5,
                                embedding_distance='euclidean',
                                ):
    """
    Contrastive loss with soft positive/negative weights.

    Instead of a binary same/different label, every pair carries a continuous
    similarity that is passed through a gate to give its positive and negative
    weight, so a pair contributes to both terms.

    Parameters
    ----------
    similarity : tensor, shape (B, B)
        Pairwise target similarity in [0,1], as returned by Q_distance.
    embeddings : tensor, shape (B, D)
        Embeddings of the same batch.
    mu : float
        Margin of the negative term.
    scale : {'sigmoid', 'lin', 'tanh'}
        Gate turning the similarity into the positive/negative weights.
    da, db : float
        Steepness and midpoint of the sigmoid and tanh gates.
    embedding_distance : {'euclidean', 'cosine'}
        Distance used between embeddings.

    Returns
    -------
    tensor
        Scalar loss, averaged over the strict upper triangle.
    """
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
    """
    Pairwise cosine similarity of descriptor vectors, rescaled to [0,1].

    Parameters
    ----------
    Q : tensor, shape (B, F)
        Descriptor vectors; rows are L2-normalized before the product.
    c_max, c_min : float
        Cosine range mapped onto [0,1].

    Returns
    -------
    tensor, shape (B, B)
        Similarity target of continuous_contrastive_loss.
    """
    Q = F.normalize(Q, p=2, dim=1)
    cos_sim = torch.matmul(Q, Q.T)
    
    cos_dist = (cos_sim - c_min) / (c_max - c_min)
    
    return cos_dist
