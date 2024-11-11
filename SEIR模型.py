import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
row=50
col=100
a=np.zeros([row,col])
num=5
n=0
while(1):
    irow=np.random.randint(0,row)
    icol=np.random.randint(0,col)
    if(a[irow,icol]==0):
        a[irow,icol]=0.75
        n=n+1
    if(n==5):
        break
time_list=[]
total_list=[]
#########-------evolve
for time in range(500): #多次实验得出350天过后病例数不变
    total=0
    #更新状态矩阵
    for i in range(row):
        for j in range(col):
            # 具有感染能力的人， 50%的概率感染别人
            if(a[i,j]>0.5 and a[i,j]<1):
                irow=np.array([i,i-1,i,i+1])
                jcol=np.array([j-1,j,j+1,j])
                flag=(irow>=0) * (irow<row) * (jcol>=0) * (jcol<row)
                rn=np.random.rand(4)
                flagij=(flag==1) * (rn<0.5)
                indx=np.where(flagij==1)
                a[irow[indx],jcol[indx]]=a[irow[indx],jcol[indx]]+0.40
                #a[irow[flag],jcol[flag]]=a[irow[flag],jcol[flag]]+0.25
            # 被感染的加0.25
            if(a[i,j]>0 and a[i,j]<1):
                a[i,j]=a[i,j]+0.40
            #统计确诊人数
            if(a[i,j]>=1):
                a[i,j]=1
                total=total+1
    time_list.append(time+1)
    total_list.append(total)
plt.plot(time_list,total_list)
plt.xlabel('time')
plt.ylabel('Number of people infected')
plt.title('Infection trend of simple grid model simulation')
plt.show()
print(total)