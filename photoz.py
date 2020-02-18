from ROOT import gROOT,TCanvas,TH1F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df=pd.read_csv('/home/elad/Advance data analysis/ada_project/Redshift_table.csv')
features=['up','gp','rp', 'ip', 'zp','Y', 'J', 'H', 'K', 'IRAC_1', 'IRAC_2']
x=df.loc[:,features].values
y=df.loc[:,['z']].values
x=StandardScaler().fit_transform(x)
pca=PCA(n_components=2)
principalCompnents=pca.fit_transform(x)
principalDf=pd.DataFrame(data=principalCompnents,columns=['principal component 1', 'principal component 2'])
finalDF=pd.concat([principalDf,df[['z']]], axis=1)
fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111, projection='3d')
# ax.set_xlabel('Principal compnent 1', fontsize=15)
# ax.set_ylabel('Principal component 2', fontsize=15)
# ax.set_title('2 component PCA',fontsize=20)
ax.scatter(finalDF['principal component 1'],finalDF['principal component 2'],finalDF['z'])
ax.set_title('The redshift as a function of two principle compontents',fontweight='bold', color = 'black', fontsize='15', horizontalalignment='center')
ax.set_xlabel('Principal component 1',fontweight='bold', color = 'black', fontsize='15', horizontalalignment='center')
ax.set_ylabel('Principle Component 2',fontweight='bold', color = 'black', fontsize='15', horizontalalignment='center')
ax.set_zlabel('z',fontweight='bold', color = 'black', fontsize='15', horizontalalignment='center')
plt.show()
fig.savefig('/home/elad/Advance data analysis/ada_project/pca_3d.png',dpi=100)
ratio2=pca.explained_variance_ratio_
features_size=np.size(features)
pca=PCA(n_components=features_size)
principalCompnents=pca.fit_transform(x)
pca_ratios=pca.explained_variance_ratio_

compnent_num=np.linspace(1,features_size,num=features_size)
#plt.step(component,ratio3)
#plt.plot(component,ratio3,'C0o')
fig=plt.gcf()
plt.bar(compnent_num,pca_ratios)
plt.xticks(compnent_num)
plt.title('The variance vs component number',fontweight='bold', color = 'black', fontsize='16', horizontalalignment='center')
plt.xlabel('Component number', fontweight='bold', color = 'black', fontsize='13', horizontalalignment='center')
plt.ylabel('Variance', fontweight='bold', color = 'black', fontsize='13', horizontalalignment='center')
plt.show()
fig.savefig('/home/elad/Advance data analysis/ada_project/pca_bar_var.png',dpi=100)

