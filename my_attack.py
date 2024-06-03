import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.autograd import Function
import permutation_ops
 
 

# class My_ATTACK_OPTIMISE:
#     def __init__(self,u, p_ms,agr_matrix_m,k,m,mylist,device):
#         self.device = device
#         self.s_m=agr_matrix_m.to(self.device)
#         self.p_ms = p_ms.to(self.device)
#         self.k=k
#         self.m=m
#         self.u=u


def calculate_threshold(s_m,u,m,k,device):
    #get s_all
    s_all=s_m*(int(u/m))
    # get the w
    n=s_m.size(0)
    A=torch.arange(0,n,dtype=torch.float32).to(device)
    w=torch.matmul(A,s_all.float())
    # w_k
    sorted_w=torch.sort(w)[0]
    w_k=sorted_w[int((1-k)*n)]
    # print('new is:')
    # print(idxx)
    w_s=w_k-m*(n-1)
    w_s2=w_k+m*(n-1)

    # find threshold v 
    
    t1=torch.nonzero(torch.lt(sorted_w,w_s)).squeeze()[-1]
    t2=torch.nonzero(torch.gt(sorted_w,w_s2)).squeeze()[0]
    # w_t1=sorted_w[t1]
    # w_t2=sorted_w[t2]
    
    return t1,t2
   
# not good for 0.1 still achieves 0.98
# def get_mal_rank(self):
    
#     p_m=self.p_ms
#     t1,t2=self.calculate_threshold()
#     n=self.s_m.size(0)
#     A=torch.arange(0,n,dtype=torch.int32)
#     p_sub=p_m[:,:,t1:t2]

#     mid=p_sub.size(2)//2
#     part1=p_sub[:,:,:mid]
#     part2=p_sub[:,:,mid:]

#     p_sub_mal=torch.cat((part2,part1),dim=2)
#     p_mal=torch.cat((p_m[:,:,:t1],p_sub_mal[:,:,:],p_m[:,:,t2:]),dim=2)
#     s_mal=torch.sum(p_mal,dim=0).to(torch.int32)

#     w=torch.matmul(A,s_mal)
#     mal_rank=torch.sort(w)[1] 
#     return mal_rank

# # not good for 0.1 still achieves 0.98
# def get_mal_rank_reverse(self,):
    
#     p_m=self.p_ms
#     t1,t2=self.calculate_threshold()
#     n=self.s_m.size(0)
#     A=torch.arange(0,n,dtype=torch.int32)
#     p_sub=p_m[:,:,t1:t2]
#     # p_mal=p_sub[:,:,::-1]
#     p_mal=torch.flip(p_sub,[2])
#     s_mal=torch.sum(p_mal,dim=0).to(torch.int32)

#     w=torch.matmul(A,s_mal)
#     mal_rank=torch.sort(w)[1] 
#     return mal_rank



def optimize(u,p_ms,s_m,k,m,device,num_epochs=100):
    t1,t2=calculate_threshold(s_m,u,m,k,device)
    
    n=s_m.size(0)
    A=torch.arange(0,n,dtype=torch.float32,requires_grad=True).to(device)

    # p_m=self.p_ms.float()
    p_sub=p_ms[:,:,t1:t2]
    p_sub.requires_grad=True
    
    s_sub=s_m[:,t1:t2].float()
    s_sub.requires_grad=True

    # create permutation matrix with size (t2-t1)*(t2-t1)
    E=torch.eye(t2-t1,requires_grad=True).to(device)
    E_1=[E for _ in range (m)]
    E_sub=torch.stack(E_1)

    # E_sub_non_leaf=E_s
    # # E_sub_non_leaf=torch.stack(E_sub)
    # E_sub=E_sub_non_leaf.detach().clone()
    # E_sub.requires_grad_()

    # gumble softmax optimise logits
    logits=torch.rand_like(E_sub)
    logits.requires_grad=True

    optim=torch.optim.Adam([logits],lr=1e-1)
    for i in range(num_epochs):
        # apply gumble softmax 
        # E_sub_train=F.gumbel_softmax(logits,tau=1,hard=True)
        
        # apply permutation_ops
        E_sub_train,b=permutation_ops.my_gumbel_sinkhorn(logits,1)


        p_m_mal=torch.matmul(p_sub,E_sub_train)
        # s_m_mal=torch.sum(p_m_mal,dim=0).detach().clone().requires_grad_()
        s_m_mal=torch.sum(p_m_mal,dim=0)
        # f=10*(A@s_sub)@(A@s_m_mal)-torch.sum((A@s_m_mal)**2)
        f=torch.norm(A@(s_sub-s_m_mal),p=2)
        # g=torch.norm(E_sub_train.sum(dim=1)-1,p=2)
        # g_sum=torch.sum(g)
        Loss=-f
        
        optim.zero_grad()
        Loss.backward()
        optim.step()
    # print(E_sub_train)
    s_mal_concat=torch.cat((s_m[:,:t1],s_m_mal,s_m[:,t2:]),dim=1)
    w_mal=(A@s_mal_concat).int()
    mal_rank=torch.sort(w_mal)[1]
    
    return mal_rank


    

