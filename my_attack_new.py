import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Function
import permutation_ops
from scipy.optimize import linear_sum_assignment
 
 

def calculate_threshold(r_s,u,m,k,max_t):
    w=torch.sort(r_s,1)[1]
    w_m=torch.sum(w,0)

    #get s_all
    w_all=w_m*(int(u/m))
    sorted_w=torch.sort(w_all)[0]
    sorted_edges=torch.sort(w_all)[1]
    # get the w
    n=w.size(1)
    w_k=sorted_w[int((1-k)*n)]

    w_s=w_k-m*(n-1)
    w_s2=w_k+m*(n-1)

    # find threshold v 
    
    # t1=torch.nonzero(torch.lt(sorted_w,w_s))[-1]
    # t2=torch.nonzero(torch.gt(sorted_w,w_s2))[0]
    # handle out of range isseu
    smaller_indices = torch.nonzero(torch.lt(sorted_w, w_s))
    greater_indices=torch.nonzero(torch.gt(sorted_w,w_s2))

    # handle the case where no elements are smaller than the threshold
    t1 = smaller_indices[-1] if smaller_indices.numel() > 0 else torch.tensor(0)
    t2 = greater_indices[0] if greater_indices.numel()>0 else torch.tensor(n)


    # solve out of memory issue

    if t2 - t1 > max_t:
        t1 = torch.tensor(int(n * (1 - k)) - max_t//2, dtype=torch.int32) if torch.tensor(int(n * (1 - k)) - max_t//2, dtype=torch.int32)>0 else torch.tensor(0)
        t2 = torch.tensor(int(n * (1 - k)) + max_t//2, dtype=torch.int32) if torch.tensor(int(n * (1 - k)) + max_t//2, dtype=torch.int32)<n else torch.tensor(n)

    # w_t1=sorted_w[t1]
    # w_t2=sorted_w[t2]
    
    return t1,t2,w,w_m,sorted_edges
 

def ensure_exact_one_hot(matrix):
    """
    Convert an approximate permutation matrix to an exact permutation matrix using the Hungarian algorithm.
    
    Args:
        matrix: Tensor of shape [batch_size, n, n]
    
    Returns:
        exact_matrix: Tensor of shape [batch_size, n, n] with one-hot permutation matrices.
    """
    batch_size, n, _ = matrix.size()
    exact_matrix = torch.zeros_like(matrix)
    
    for b in range(batch_size):
        cost_matrix = -matrix[b].detach().cpu().numpy()  # Convert to numpy and negate for maximization problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        exact_matrix[b, row_ind, col_ind] = 1
    
    return exact_matrix

def optimize(u,r_s,k,m,device,lr,nep,max_t,temp,iteration,noise):
    t1,t2,w,w_b,sorted_edges=calculate_threshold(r_s,u,m,k,max_t) 
    sorted_w=torch.sort(w)[0]
    
    selected_edges=sorted_edges[t1:t2]
    selected_edges_m= selected_edges.repeat(m, 1).unsqueeze(1)

    # p_m=self.p_ms.float()
    w_sub=sorted_w[:,t1:t2].float().unsqueeze(1)
    w_sub.requires_grad=True

    # aggreated beign w

    w_b_sub=w_b[t1:t2].float()
    

    # create permutation matrix with size (t2-t1)*(t2-t1)
    E=torch.eye((t2-t1).item(),requires_grad=True).to(device)
    E_1=[E for _ in range (m)]
    E_sub=torch.stack(E_1)

    # gumble softmax optimise logits
    logits=torch.rand_like(E_sub)
    logits.requires_grad=True

    optim=torch.optim.Adam([logits],lr)
    for i in range(nep):
        # apply gumble softmax 
        # E_sub_train=F.gumbel_softmax(logits,tau=1,hard=True)
        
        # apply permutation_ops
        E_sub_train,b=permutation_ops.my_gumbel_sinkhorn(logits,temp,noise_factor=noise,n_iters=iteration)

        w_mal=torch.matmul(w_sub,E_sub_train)
        w_mal_agg=torch.sum(w_mal,0).squeeze(0)

        # f=10*(A@s_sub)@(A@s_m_mal)-torch.sum((A@s_m_mal)**2)
        f=torch.norm((w_b_sub-w_mal_agg),p=2)
        # g=torch.norm(E_sub_train.sum(dim=1)-1,p=2)
        # g_sum=torch.sum(g)
        Loss=-f
        
        optim.zero_grad()
        Loss.backward()
        optim.step()
    # print(E_sub_train)
    E_final=ensure_exact_one_hot(E_sub_train)
    mal_selected=(selected_edges_m.float())@E_final
    mal_rank=torch.cat((r_s[:,:t1],mal_selected.squeeze(1),r_s[:,t2:]),dim=1)
    
    return mal_rank


    

