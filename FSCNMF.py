import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from scipy.sparse import csc_matrix as csc
from numpy import linalg as LA
import string
import sys
import os
import gc
import sys


gc.enable()

tot_Iterations = 10 # upto which order FSCNMF should run


if len(sys.argv)!=5:
	print 'Insufficient arguments \n exiting program..........\n'
	sys.exit()

adjFile = sys.argv[1]   #'adjacencyList.csv'   # assuming comma as delimiter. Each node's info  on new line
contFile = sys.argv[2]     #'content.csv'   # assuming comma as delimiter. Each node's info  on new line
pidFile = sys.argv[3]         #'paperId.csv'   # each elemet on new line
num_communities = int(sys.argv[4])         #'labels.csv'   # each elemet on new line

#~ true_labels = pd.read_csv(labelsFile, header=None)
#~ true_labels = true_labels[0].tolist()

#~ num_communities = len(set(true_labels))    #unique labels give number of community
k=10*num_communities #dimension of the final embedding space



df_paperId = pd.read_csv(pidFile, header=None)
print df_paperId.shape

#paperId to index dictionary
paperIdInd={}
ind=0
for item in df_paperId.ix[:,0]:
    paperIdInd[str(item)]=ind
    ind+=1

       
#Processing the adjaceny list to create a sparse adjacency matrix
print 'Processing adjaceny list started'


rowList=[]
columnList=[]
valueList=[]
with open(adjFile) as infile:
    data = infile.read().split('\n')
    if len(data[-1])==0:
        data.pop()

row=0   
for line in data:
    reflists=line.split(',')
    if len(reflists[-1])==0:
        reflists.pop()
        
    curr_item=reflists[0]
    reflists=reflists[1:]

    for item in reflists:
        if item in paperIdInd:
            rowList.append(paperIdInd[curr_item])
            columnList.append(paperIdInd[item])
            valueList.append(1)   # for weighted graph it can be the weights of edge

    row+=1
        
adjMatSPC = csc((np.array(valueList), (np.array(rowList), np.array(columnList))), shape=(len(df_paperId), len(df_paperId)))
            
print '*********************** Processing adjaceny list '

#Processing the content
    
print '**********************  Processing content started'
    
C = pd.read_csv(contFile, header=None)   #prefered to be in tf-idf form
C = C.as_matrix(columns=None)


gc.collect()


small=pow(10,-3)   #least value possible in the embedding

n=len(df_paperId)  #number of nodes
d=C.shape[1]       # number of features in content matrix

outerIter=10
opti1Iter=3
opti2Iter=3

for num_iter in range(tot_Iterations): #To rtun nth order FSCNMF, give n in the input

    print 'FSCNMF++ of order '+str(num_iter+1)+' has started'
    gc.collect()
    
    tmpMat = adjMatSPC
        
    temp = adjMatSPC
    for i in range(2, num_iter+2):
        temp = temp * adjMatSPC
        tmpMat = tmpMat + temp
        
    tmpMat = (tmpMat).astype(float) / (num_iter+1)
    A = (tmpMat.toarray()).astype(float)
    A = A.astype(float) #Get the matrix for FSCNMF of order (num_iter+1)
    
    A = np.nan_to_num(A)  #remove the nan value
    
    
    #Initializing matrices based on regular NMF
    model1 = NMF(n_components=k, init='nndsvd', random_state=0)
    B1 = model1.fit_transform(A)
    B2 = model1.components_
    
    model2 = NMF(n_components=k, init='nndsvd', random_state=0)
    U = model2.fit_transform(C)
    V = model2.components_


    
    B1=np.nan_to_num(B1) #Removing Nan or infinity, if any by numbers 0 or a large number respectively
    B2=np.nan_to_num(B2)
    U=np.nan_to_num(U)
    V=np.nan_to_num(V)
    
    
    B1_new=B1
    B2_new=B2
    U_new=U
    V_new=V
    
    
    const1 = np.ones((n,k))/k
    const2 = np.ones((k,d))/d
    const3 = np.ones((k,n))/k
    
    
    alpha1=1000 #10000.0 #1000 #match constraint
    alpha2=1.0 #0.001
    alpha3=1.0
    beta1=1000.0 #1000.0 #match constraint
    beta2=1.0
    beta3=1.0
    
    
    
    opti1_values=[]
    opti2_values=[]
    
    count_outer=1
    
    while True:
        
        print '\nOuter Loop '+str(count_outer)+' started \n'
        
        gamma = 0.001
        count1 = 1
        
        while True: #Optimization 1
            funcVal = 1.0/2.0 * pow( LA.norm(A-np.matmul(B1,B2),'fro'), 2) + alpha1/2 * pow(LA.norm(B1-U,'fro'),2) + alpha2 * np.abs(B1).sum() + alpha3 * np.abs(B2).sum()
            opti1_values.append(funcVal)
            

            B1_new = np.multiply(B1, np.divide( np.matmul(A,np.transpose(B2))+alpha1*U , (np.matmul(np.matmul(B1,B2), np.transpose(B2)) + alpha1*B1 + alpha2*B1).clip(min=small) ) ).clip(min=small) #Multiplicative update rule - aswin
            B2_new = np.multiply(B2, np.divide( np.matmul(np.transpose(B1),A), (np.matmul(np.transpose(B1),np.matmul(B1,B2))+beta3*B2).clip(min=small) )).clip(min=small)

                
            B1 = B1_new
            B2 = B2_new
            
            gamma = gamma/count1
            
            count1+=1
        
            if count1>opti1Iter:
                opti1_values.append(1.0/2.0 * pow( LA.norm(A-np.matmul(B1,B2),'fro'), 2) + alpha1/2 * pow(LA.norm(B1-U,'fro'),2) + alpha2 * np.abs(B1).sum() + alpha3 * np.abs(B2).sum())
                opti1_values.append(None)
                break
                
        
        count2=1
        gamma = 0.001
        while True:  #Optimization 1
            
            funcVal = 1.0/2.0 * pow(LA.norm(C - np.matmul(U,V), 'fro'), 2) + beta1/2 * pow(LA.norm(U-B1, 'fro'), 2) + beta2 * np.abs(U).sum() + beta3 * np.abs(V).sum()
            opti2_values.append(funcVal)
            
            
            U_new = np.multiply(U, np.divide( np.matmul(C,np.transpose(V))+beta1*B1 , (np.matmul(np.matmul(U,V), np.transpose(V)) + beta1*U + beta2*U).clip(min=small) ) ).clip(min=small) #Multiplicative update rule - aswin
            V_new = np.multiply(V, np.divide( np.matmul(np.transpose(U),C), (np.matmul(np.transpose(U),np.matmul(U,V))+beta3*V).clip(min=small) )).clip(min=small)
            
            
            U = U_new
            V = V_new
            
            gamma = gamma/count2
            
            count2+=1
        
            if(count2>opti2Iter):
                opti2_values.append(1.0/2.0 * pow(LA.norm(C - np.matmul(U,V), 'fro'), 2) + beta1/2 * pow(LA.norm(U-B1, 'fro'), 2) + beta2 * np.abs(U).sum() + beta3 * np.abs(V).sum())
                opti2_values.append(None)
                break
            
            
        
        count_outer+=1
        if(count_outer>outerIter):
            break
        
    
    optiFile = open('FSCNMF++_'+str(num_iter+1)+'_OptiValues.csv', 'w')

    for item in opti1_values:
        optiFile.write(str(item)+'\t')
    
    optiFile.write('\n')
    
    for item in opti2_values:
        optiFile.write(str(item)+'\t')
    
    optiFile.close()
    
    
    #Normalising the rows
    B1 = (B1_new/((np.abs(B1_new).sum(axis=1)).reshape(n,1))).clip(min=small) 
    B2 = (B2_new/((np.abs(B2_new).sum(axis=1)).reshape(k,1))).clip(min=small)
    
    U = (U_new/((np.abs(U_new).sum(axis=1)).reshape(n,1))).clip(min=small)
    V = (V_new/((np.abs(V_new).sum(axis=1)).reshape(k,1))).clip(min=small)
    
    
    
    np.savetxt('FSCNMF++_order'+str(num_iter+1)+'_rep.txt', B1)
    
    gc.collect()
    
    print 'Embedding done for FSCNMF++ of order ',num_iter+1,'\n'
    
    

