import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Function
import permutation_ops
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
 

def calculate_threshold_old(r_s,u,m,k,max_t):
    w=torch.sort(r_s,1)[1]  # repution for m beign updates
    w_m=torch.sum(w,0)   # aggreated reputation e_1, e_2, ... e_n
    
    #get s_all
    w_all=w_m*(int(u/m)-1)
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
def calculate_threshold(r_s,u,m,k,max_t):
    w=torch.sort(r_s,1)[1]  # repution for m beign updates
    w_m=torch.sum(w,0)   # aggreated reputation e_1, e_2, ... e_n
    
    #get s_all
    w_all=w_m*(int(u/m)-1)
    sorted_w=torch.sort(w_all)[0]
    sorted_edges=torch.sort(w_m)[1]
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
 
def calculate_threshold_every_edge(r_s,u,m,k,max_t):
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
    t1= torch.tensor(0)
    t2 = torch.tensor(n)


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

def optimize_old(u,r_s,k,m,device,lr,nep,max_t,temp,iteration,noise):
    t1,t2,w,w_b,sorted_edges=calculate_threshold(r_s,u,m,k,max_t) 
    sorted_w=torch.sort(w)[0]
    
    selected_edges=sorted_edges[t1:t2]
    selected_edges_m= selected_edges.repeat(m, 1).unsqueeze(1)

    # p_m=self.p_ms.float()
    w_sub=sorted_w[:,t1:t2].float().unsqueeze(1)    # sorted based on repution 
    w_sub.requires_grad=True


    w_b_sub=w_b[t1:t2].float()          # w_b is aggreated repution 
    

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
        w_mal_agg=torch.sum(w_mal,0).squeeze(0)   # get the aggreated reputation for vunerable edges

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
def optimize(u,r_s,k,m,device,lr,nep,max_t,temp,iteration,noise):
    t1,t2,w,w_b,sorted_edges=calculate_threshold(r_s,u,m,k,max_t) 
    sorted_reputation=torch.sort(w_b)
    
    # selected_edges=sorted_edges[t1:t2]            #get vunberale edges
    # selected_edges_m= selected_edges.repeat(m, 1).unsqueeze(1)

    # p_m=self.p_ms.float()

    sorted_reputation_sub=sorted_reputation[0][t1:t2].float() # get the aggreated repution for vunerable edges
    vunerable_edges=sorted_reputation[1][t1:t2]    # vunerable edges
    vunerable_rankings=w[:,vunerable_edges].float().unsqueeze(1)


    sorted_reputation_sub.requires_grad=True

    # w_sub=w_b[t1:t2]
    
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

        rank_mal=torch.matmul(vunerable_rankings,E_sub_train)
        w_mal_agg=torch.sum(rank_mal,0).squeeze(0)   # get the aggreated reputation for vunerable edges

        # f=10*(A@s_sub)@(A@s_m_mal)-torch.sum((A@s_m_mal)**2)
        f=torch.norm((sorted_reputation_sub-w_mal_agg),p=2)
        # g=torch.norm(E_sub_train.sum(dim=1)-1,p=2)
        # g_sum=torch.sum(g)
        Loss=-f
        
        optim.zero_grad()
        Loss.backward()
        optim.step()
    # print(E_sub_train)
    E_final=ensure_exact_one_hot(E_sub_train)
    mal_rank_vunberable_edges=((vunerable_rankings)@E_final).squeeze(1)
    # mal_rank=torch.cat((r_s[:,:t1],mal_selected.squeeze(1),r_s[:,t2:]),dim=1)
    w[:,vunerable_edges]=mal_rank_vunberable_edges.long()      #update w 
    
    return w

def modification_old(r_s,u,m,k):
    w=torch.sort(r_s,1)[1]
    w_m=torch.sum(w,0)
    
    #get s_all
    w_all=w_m*(int(u/m)-1)
    sorted_w=torch.sort(w_all)[0]
    sorted_edges=torch.sort(w_all)[1]
    # get the w
    n=w.size(1)
    w_k=sorted_w[int((1-k)*n)]
    d1=torch.abs(w_k-sorted_w)

    # Convert the tensor to a NumPy array
    array = d1.cpu().numpy()

    w_k_index = int((1 - k) * n)
    # Calculate the index differences
    index_differences = torch.arange(len(sorted_w)) - w_k_index

    # Convert tensors to NumPy arrays
    array = d1.cpu().numpy().flatten()
    index_diff_array = index_differences.cpu().numpy()

    y_value =  1728 * 5


    # Plot the array
    plt.figure(figsize=(8, 6))
    plt.plot(index_diff_array, array,color='#92A5D1', linewidth=2,label='Reputation change needed')
    plt.axhline(y=y_value, color='#C25759', linestyle='--', linewidth=2,label=f'Reputation change attacker can cause')  # Horizontal line
    # plt.fill_between(index_diff_array[shade_start:shade_end], array[shade_start:shade_end], y_value, color='gray', alpha=0.3)  # Shaded area
    plt.fill_between(index_diff_array, array, y_value, where=(array < y_value), color='gray', alpha=0.3)

    plt.xlabel('Index Difference',fontsize=20)
    plt.ylabel('Value',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('difference_cifar.pdf')

    # plt.title('Plot of 1D Tensor with Index Differences')
    plt.show()


    # w_s=w_k-m*(n-1)
    # w_s2=w_k+m*(n-1)

    # # find threshold v 
    
    # # t1=torch.nonzero(torch.lt(sorted_w,w_s))[-1]
    # # t2=torch.nonzero(torch.gt(sorted_w,w_s2))[0]
    # # handle out of range isseu
    # smaller_indices = torch.nonzero(torch.lt(sorted_w, w_s))
    # greater_indices=torch.nonzero(torch.gt(sorted_w,w_s2))

    # # handle the case where no elements are smaller than the threshold
    # t1 = smaller_indices[-1] if smaller_indices.numel() > 0 else torch.tensor(0)
    # t2 = greater_indices[0] if greater_indices.numel()>0 else torch.tensor(n)

def modification(r_s,u,m,k):
    w=torch.sort(r_s,1)[1]
    w_m=torch.sum(w,0)
    
    #get s_all
    w_all=w_m*(int(u/m)-1)
    sorted_w=torch.sort(w_all)[0]      # sorted reputation
    sorted_edges=torch.sort(w_all)[1]      # get ranking again
    # get the w
    n=w.size(1)
    w_k=sorted_w[int((1-k)*n)]
    d1=torch.abs(w_k-w_all)

    # Convert the tensor to a NumPy array
    array = d1.cpu().numpy()

    w_k_index = int((1 - k) * n)
    # Calculate the index differences
    # index_differences = torch.arange(len(sorted_w)) - w_k_index
    index = torch.arange(1, len(w_all) + 1).cpu().numpy()


    # Convert tensors to NumPy arrays
    array = d1.cpu().numpy().flatten()
    # index_diff_array = index.cpu().numpy()

    # y_value =  1728 * 5
    # y_value =  288*2*5


    # Plot the array
    # plt.figure(figsize=(8, 6))
    # # plt.plot(index, array,color='#92A5D1', linewidth=2,label='Reputation change needed for each edge')
    # plt.scatter(index, array, color='#92A5D1', s=10,label='Reputation change needed')
    # plt.axhline(y=y_value, color='#C25759', linestyle='--', linewidth=2,label=f'Reputation change attacker can cause')  # Horizontal line
    # # plt.fill_between(index_diff_array[shade_start:shade_end], array[shade_start:shade_end], y_value, color='gray', alpha=0.3)  # Shaded area
    # # plt.fill_between(index, array, y_value, where=(array < y_value), color='gray', alpha=0.3)

    # plt.xlabel('Edge index',fontsize=20)
    # plt.ylabel('Value',fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.legend(fontsize=20)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('difference_cifar.pdf')

    # # plt.title('Plot of 1D Tensor with Index Differences')
    # plt.show()

    # Histogram parameters
    # num_bins = 10  # You can adjust the number of bins

    # # Plot the histogram
    # plt.figure(figsize=(8, 6))
    # plt.hist(array, bins=num_bins, color='#92A5D1', edgecolor='black', alpha=0.7)
    # # plt.axhline(y=y_value, color='#C25759', linestyle='--', linewidth=2, label=f'Reputation change attacker can cause')  # Horizontal line
    # plt.axvline(x=y_value, color='#C25759', linestyle='--', linewidth=2, label=f'Reputation change attacker can cause')  # Vertical line

    # plt.xlabel('Reputation change needed', fontsize=25)
    # plt.ylabel('Number of edges', fontsize=25)
    # # x_ticks = np.linspace(array.min(), array.max(), num=4)
    # plt.xticks(fontsize=20)
    # plt.xticks(fontsize=20,ticks=[0,5000,10000,15000], labels=[0,50000,10000,15000])

    # plt.yticks(fontsize=20)
    # # plt.legend(fontsize=20)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('histogram_cifar.pdf')

    # plt.show()

    # plot number of edges smaller than y

# Define the range of i values
    i_values = np.arange(1, 21)

    # Calculate malicious_rate for each i value
    malicious_rate =  i_values/25

    # Define the y_value range with more points
    # y_values = np.array([1728 * i for i in i_values])
    y_values = np.array([288*2 * i for i in i_values])


    # Calculate the number of values in array smaller than each y_value
    num_values_smaller_than_y_value = [np.sum(array < y) for y in y_values]

    # Find the minimum number of values smaller than y_value where malicious_rate > 0
    min_value = min(num_values_smaller_than_y_value)
    min_percentage = min_value/len(array)

    # Calculate percentage
    percentage_values_smaller_than_y_value = np.array(num_values_smaller_than_y_value) / len(array) 


    # Plot the number of values smaller than y_value against malicious_rate
    plt.figure(figsize=(8, 6))
    plt.plot(malicious_rate, percentage_values_smaller_than_y_value, marker='o', linestyle='-', color='#92A5D1', linewidth=2)

    plt.xlabel('Malicious Rate', fontsize=25)
    # plt.ylabel('Percentage of vulnerable edges', fontsize=25)
    # plt.title('Number of Values Smaller than y_value vs Malicious Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    # Highlight the minimum value on the plot
    plt.axhline(y=min_percentage, color='#C25759', linestyle='--', linewidth=2, label=f'Lower bound: {min_percentage:.2f}')
    plt.legend(fontsize=20)
    plt.savefig('mnist_m_r.pdf')

    plt.show() 
   











    # w_s=w_k-m*(n-1)
    # w_s2=w_k+m*(n-1)

    # # find threshold v 
    
    # # t1=torch.nonzero(torch.lt(sorted_w,w_s))[-1]
    # # t2=torch.nonzero(torch.gt(sorted_w,w_s2))[0]
    # # handle out of range isseu
    # smaller_indices = torch.nonzero(torch.lt(sorted_w, w_s))
    # greater_indices=torch.nonzero(torch.gt(sorted_w,w_s2))

    # # handle the case where no elements are smaller than the threshold
    # t1 = smaller_indices[-1] if smaller_indices.numel() > 0 else torch.tensor(0)
    # t2 = greater_indices[0] if greater_indices.numel()>0 else torch.tensor(n)

def reverse(rank,u,r_s,k,m,max_t):
    t1,t2,w,w_b,sorted_edges=calculate_threshold(r_s,u,m,k,max_t) 
    mid_tensor = rank[t1:t2]
    reverse_tensor=torch.flip(mid_tensor, [0])
    rest_first=rank[:t1]
    rest_last=rank[t2:]
    mal_rank=torch.cat((rest_first, reverse_tensor, rest_last), dim=0)
    return mal_rank
    

