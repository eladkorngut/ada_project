from ROOT import gROOT,TCanvas,TH1F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def standertize_data(df,features):
    #standartization (normailzing mean=0 and variance)
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    return x

def df_pca(df,features,component_num):
    # Function that will retun a dataframe with a given number of components (component_num) and z
    sd_data=standertize_data(df,features)
    pca = PCA(n_components=component_num)
    pc=pca.fit_transform(sd_data)
    principalDf = pd.DataFrame(data=pc, columns=['principal component 1', 'principal component 2'])
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
pca_results(df,features)

