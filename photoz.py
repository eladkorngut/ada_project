import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import linear_model

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D

def standertize_data(df,features):
    #standartization (normailzing mean=0 and variance)
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    return x

def df_pca(df,features,component_num):
    col=[]
    # Function that will retun a dataframe with a given number of components (component_num) and z
    sd_data=standertize_data(df,features)
    pca = PCA(n_components=component_num)
    pc=pca.fit_transform(sd_data)
    # for i in range(component_num):
    #     col.append('Principal component'+str(i))
    temp=range(component_num)
    temp=map(str,temp)
    col=np.core.defchararray.add(['PCA '],temp)
    principalDf = pd.DataFrame(data=pc, columns=col)
    finalDF = pd.concat([principalDf, df[['z']]], axis=1)
    return finalDF

def print_pca_2_component(finalDF):
    #Printing and saving the graph for 2 components vs z
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(finalDF['principal component 1'], finalDF['principal component 2'], finalDF['z'])
    ax.set_title('The redshift as a function of two principle compontents', fontweight='bold', color='black',
                 fontsize='15', horizontalalignment='center')
    ax.set_xlabel('Principal component 1', fontweight='bold', color='black', fontsize='15',
                  horizontalalignment='center')
    ax.set_ylabel('Principle Component 2', fontweight='bold', color='black', fontsize='15',
                  horizontalalignment='center')
    ax.set_zlabel('z', fontweight='bold', color='black', fontsize='15', horizontalalignment='center')
    plt.show()
    fig.savefig('/home/elad/Advance data analysis/ada_project/pca_3d.png', dpi=100)

def print_pca_component_var(pca_ratios,features_size):
    #Function to print and save as fig the variance at each component
    compnent_num = np.linspace(1, features_size, num=features_size)
    fig = plt.gcf()
    plt.bar(compnent_num, pca_ratios)
    plt.xticks(compnent_num)
    plt.title('The variance vs component number', fontweight='bold', color='black', fontsize='16',
              horizontalalignment='center')
    plt.xlabel('Component number', fontweight='bold', color='black', fontsize='13', horizontalalignment='center')
    plt.ylabel('Variance', fontweight='bold', color='black', fontsize='13', horizontalalignment='center')
    plt.show()
    fig.savefig('/home/elad/Advance data analysis/ada_project/pca_bar_var.png', dpi=100)


def pca_results(df,features):
    #The function print and save the variance at each component and the 3d model, answer question 1b
    pca_df=df_pca(df,features,2) #The data frame for two components
    print_pca_2_component(pca_df)
    #Code bellow is for getting how much of the variance is contained in each component
    sd_data=standertize_data(df,features)
    features_size=np.size(features)
    pca = PCA(n_components=features_size)
    pca.fit_transform(sd_data)
    pca_ratios = pca.explained_variance_ratio_
    print_pca_component_var(pca_ratios, features_size)


#Main body of the program start here
#importing data and assigning what are the features (the DF without the errors)

df=pd.read_csv('/home/elad/Advance data analysis/ada_project/Redshift_table.csv')
features=['up','gp','rp', 'ip', 'zp','Y', 'J', 'H', 'K', 'IRAC_1', 'IRAC_2']
#pca_results(df,features)

one_var_df=df_pca(df,features,3)
fig = plt.gcf()
plt.scatter(one_var_df['PCA 0'],one_var_df['z'])
plt.show()
plt.scatter(one_var_df['PCA 1'],one_var_df['z'])
plt.show()
fig = plt.gcf()
plt.scatter(one_var_df['PCA 0'],one_var_df['PCA 1'],c=one_var_df['z'],cmap='coolwarm')
fig.savefig('/home/elad/Advance data analysis/ada_project/scatter_heat.png',dpi=100)
plt.show()

sd_data = standertize_data(df, features)
pca = PCA(n_components=2)
pc = pca.fit_transform(sd_data)
# for i in range(component_num):
#     col.append('Principal component'+str(i))
temp = range(2)
temp = map(str, temp)
col = np.core.defchararray.add(['PCA '], temp)
principalDf = pd.DataFrame(data=pc, columns=col)
k=KMeans(n_clusters=5).fit(one_var_df)
centers=k.cluster_centers_
plt.scatter(principalDf['PCA 0'],principalDf['PCA 1'],c=one_var_df['z'],cmap='coolwarm')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
lin_model_num=k.labels_
plt.scatter(principalDf['PCA 0'],principalDf['PCA 1'],c=lin_model_num,cmap='brg')
plt.show()
lin_model_num_df=pd.DataFrame(lin_model_num,columns=['Model'])
data_model_num = pd.concat([one_var_df, lin_model_num_df], axis=1)
x=data_model_num[data_model_num['Model']==3]
x_trunck=x[['PCA 0', 'PCA 1']]
y=x[['z']]
regr=linear_model.LinearRegression()
regr.fit(x_trunck,y)
new_point_pca0=12.322401
new_point_pca1=-5.58670
temp=regr.predict([[new_point_pca0,new_point_pca1]])


# plt.scatter(one_var_df['PCA 0'],one_var_df['PCA 1'],one_var_df['PCA 2'],c=one_var_df['z'],cmap='plasma')
