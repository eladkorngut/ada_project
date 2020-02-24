import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

def standertize_data(df,features):
    #standartization (normailzing mean=0 and variance)
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    return x

def df_pca_only(df,features,component_num):
    # Function that will retun a dataframe with a given number of components (component_num)
    sd_data=standertize_data(df,features)
    pca = PCA(n_components=component_num)
    pc=pca.fit_transform(sd_data)
    return sd_data, pd.DataFrame(data=pc, columns=np.core.defchararray.add(['PCA '],map(str,range(component_num))))

def df_pca_with_z(df_pca_only,df_cvs_file):
    # Function that will return a dataframe with a given number of components (component_num) and z
    return pd.concat([df_pca_only, df_cvs_file[['z']]], axis=1)

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

def print_pca_heatmap(df_pca3_z,fig):
    plt.scatter(df_pca3_z['PCA 0'], df_pca3_z['PCA 1'], c=df_pca3_z['z'], cmap='coolwarm')
    plt.title('The two most dominant PCA vs redshift (z) ', fontweight='bold', color='black', fontsize='16',
              horizontalalignment='center')
    plt.xlabel('PCA 1', fontweight='bold', color='black', fontsize='13', horizontalalignment='center')
    plt.ylabel('PCA 2', fontweight='bold', color='black', fontsize='13', horizontalalignment='center')


def pca_results(sd_data,features):
    #The function retuns the PCA model used to transfrom the data
    #Code bellow is for getting how much of the variance is contained in each component
    features_size=np.size(features)
    pca = PCA(n_components=features_size) #The model that is used for the pca
    pca_vec=pca.fit_transform(sd_data)
    return pca, pca_vec

def print_pca_data(pca,df,features_size):
    #The function print and save the variance at each component and the heatmap for 2 PCA, answer question 1b
    fig = plt.gcf()
    print_pca_heatmap(df,fig)
    fig.savefig('/home/elad/Advance data analysis/ada_project/scatter_heat.png', dpi=100)
    plt.show()
    pca_ratios = pca.explained_variance_ratio_
    print_pca_component_var(pca_ratios, features_size)

def print_kmeans_heatmap(df,kmeans_model):
    fig = plt.gcf()
    print_pca_heatmap(df,fig)
    centers=kmeans_model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    fig.savefig('/home/elad/Advance data analysis/ada_project/heat_map_kmeans.png', dpi=100)
    plt.show()

def print_kmeans_model_selction(df,kmeans_model):
    fig = plt.gcf()
    print_pca_heatmap(df,fig)
    lin_model_num=kmeans_model.labels_
    plt.scatter(df['PCA 0'], df['PCA 1'], c=lin_model_num, cmap='brg')
    fig.savefig('/home/elad/Advance data analysis/ada_project/model_selction_kmeans.png', dpi=100)
    plt.show()

def create_df_pca_z_model(df,kmeans_model):
    return pd.concat([df, pd.DataFrame(kmeans_model.labels_,columns=['Model'])], axis=1)

def select_model_data(df,m):
    return df[df['Model'] == m]

def separate_data_target(df):
    return df[['PCA 0', 'PCA 1', 'PCA 2']],  df[['z']]

def linear_fit_1kmean(df,m,regr):
    x, y =separate_data_target(select_model_data(df, m))
    # pca_data_labels =["PCA 0", "PCA 1", "PCA 2"]
    # pca_data = x.loc[:, pca_data_labels].values
    # pca_z=y.loc[:,['z']].values
    return regr.fit(x,y)

def linear_fit_all(df,size_models):
    linear_model_fits=[]
    for i in range(size_models):
        regr = linear_model.LinearRegression()
        linear_model_fits.append(linear_fit_1kmean(df, i, regr))
    return linear_model_fits

def find_closest_cluster_pca_only(test_set_point,kmeans_model,size_centers):
    centers=kmeans_model.cluster_centers_
    no_z_centers = np.delete(centers,3,1)
    #What written below should be joined into one command
    test_set_point=([test_set_point,]*size_centers)
    sub=np.subtract(no_z_centers,test_set_point)
    sq=np.square(sub)
    sum_sq=np.sum(sq,axis=1)
    min_value=np.min(sum_sq)
    min_index=np.where(sum_sq == min_value)
    return min_index[0]

def predictions_test_set(linear_aprox_array,kmeans_model,test_set,size_centers):
    prediction_for_t=[]
    features=['PCA 0', 'PCA 1', 'PCA 2']
    test_set_array = test_set.loc[:, features].values   #assuming passing a dataframe then needs to convert to array
    # closest_cluster=kmeans_model.predict(test_set_array)
    for t in test_set_array:
        closest_cluster_index=find_closest_cluster_pca_only(t, kmeans_model, size_centers)
        prediction_for_t.append(float(linear_aprox_array[int(closest_cluster_index)].predict([t])))
    pd_pred = pd.DataFrame(np.asarray(prediction_for_t),index=test_set.index.values ,columns=['Prediction'])
    pd_return=pd.concat([test_set, pd_pred],axis=1)
    rel_err = np.absolute(pd_return['z'] - pd_return['Prediction']) * 100 / pd_return['z']
    pd_return = pd_return.assign(Error=rel_err)
    return pd_return

#Main body of the program start here
#importing data and assigning what are the features (the DF without the errors)
component_num=3
size_cluster=5
df=pd.read_csv('/home/elad/Advance data analysis/ada_project/Redshift_table.csv')
features=['up','gp','rp', 'ip', 'zp','Y', 'J', 'H', 'K', 'IRAC_1', 'IRAC_2']
sd_data, df_pca=df_pca_only(df,features,component_num)
df_pca_z=df_pca_with_z(df_pca, df)
pca_model, pca_array=pca_results(sd_data, features)
print_pca_data(pca_model, df_pca_z, np.size(features))


kmeans_model = KMeans(n_clusters=size_cluster).fit(df_pca_z)
train_set, test_set = train_test_split(df_pca_z,test_size=0.2)
print_kmeans_heatmap(train_set, kmeans_model)
print_kmeans_model_selction(df_pca_z, kmeans_model)

df_pca_z_model = create_df_pca_z_model(df_pca_z, kmeans_model)
array_of_lin_models=linear_fit_all(df_pca_z_model,size_cluster)

#Bellow, final_analysis, is the most important piece of the work it answer the research question
final_analysis = predictions_test_set(array_of_lin_models, kmeans_model, test_set, size_cluster)



