from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
'''
意义 感染者传染强度 潜伏者传染强度 潜伏转换率 因病死亡率 完全恢复率 暂时恢复率 复感染率
符号 a1 a2 b m n1 n2 p
值 0.5 0.2 0.1 0.05 0.3 0.7 0.3
'''
a1,a2,b,m,n1,n2,p=0.5,0.2,0.1,0.05,0.3,0.7,0.3
def models(z,t):
    S,E,I,R1,R2,ID=z
    N1=S+E+I+R2
    exp1=-(a1*I+a2*E)*S/N1+p*R2
    exp2=(a1*I+a2*E)*S/N1-b*E
    exp3=b*E-I*(m+n1+n2)
    exp4=n1*I
    exp5=n2*I-p*R2
    exp6=m*I
    return [exp1,exp2,exp3,exp4,exp5,exp6]
t=np.linspace(0,400,100)
p0=[9999,1,0,0,0,0]
z=integrate.odeint(models,p0,t).T
print(z)
name=['S','E','I','R1','R2','ID']
for i in range(len(name)):
    plt.plot(t,z[i],label=name[i])
plt.legend()
plt.title('SEIR MODEL')
plt.show()