#This code was created by elad korngut in order
#to find z, it requires the data file from the
#moodle will be in cvs format. It is written
#in python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import mean_squared_error
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def standertize_data(df,features):
    #standartization (normailzing mean=0 and variance)
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    return x

def df_pca_only(df,features,component_num):
    # Function that will retun the PCA
    #  as a dataframe with a given number
    # of components (component_num)
    sd_data = standertize_data(df,features)
    pca = PCA(n_components=component_num)
    pc=pca.fit_transform(sd_data)
    return sd_data, pd.DataFrame(data=pc,
            columns=np.core.defchararray.add(['PCA '],
            map(str,range(component_num))))

def df_pca_with_z(df_pca_only,df_cvs_file):
    # Attaching z to our PCA DF (component_num) and z
    df_cvs_file.reset_index(inplace=True)
    return pd.concat([df_pca_only, df_cvs_file['z']],axis=1)

def print_pca_component_var(pca_ratios,features_size):
    #Function to print and save as
    # fig the variance at each component
    compnent_num = np.linspace(1,
                    features_size, num=features_size)
    fig = plt.gcf()
    plt.bar(compnent_num, pca_ratios)
    plt.xticks(compnent_num)
    plt.title('The information held '
              'vs component number'
        , fontweight='bold',
    color='black', fontsize='16'
    ,horizontalalignment='center')
    plt.xlabel('Component number',
    fontweight='bold',
        color='black', fontsize='11',
        horizontalalignment='center')
    plt.ylabel('Information', fontweight='bold', color='black',
    fontsize='11', horizontalalignment='center')
    plt.show()
    fig.savefig('/home/elad/Advance data analysis'
    '/ada_project/pca_bar_var.png', dpi=100)

def print_pca_heatmap(df_pca3_z,colorbar=True):
    plt.scatter(df_pca3_z['PCA 0'], df_pca3_z['PCA 1'],
                c=df_pca3_z['z'], cmap='plasma')
    if(colorbar):
        plt.colorbar().set_label('z',
                        fontweight='bold',fontsize=13)
    plt.title('The two most dominant PCA vs redshift (z) ',
    fontweight='bold', color='black', fontsize='16',
        horizontalalignment='center')
    plt.xlabel('PCA 1', fontweight='bold',
        color='black', fontsize='13',
        horizontalalignment='center')
    plt.ylabel('PCA 2', fontweight='bold', color='black',
        fontsize='13', horizontalalignment='center')


def pca_results(sd_data,features,num_component):
#The function retuns the PCA model used to transfrom the data
#Code bellow is for getting how much of the
#variance is contained in each component
    pca = PCA(n_components=num_component)
    pca_vec=pca.fit_transform(sd_data)
    return pca, pca_vec

def print_pca_data(pca,df,features_size):
    #The function print and save the variance
    # at each component and the heatmap
    # for 2 PCA, answers question 1b
    fig = plt.gcf()
    print_pca_heatmap(df,True)
    fig.savefig('/home/elad/Advance data analysis'
                '/ada_project/scatter_heat.png', dpi=100)
    plt.show()
    pca_ratios = pca.explained_variance_ratio_
    print('The information precentage held '
          'by each PCA is ',100*pca_ratios)
    print_pca_component_var(pca_ratios, features_size)

def print_kmeans_heatmap(df,kmeans_model):
    #Plots and saves kmeans clusters on the data
    fig = plt.gcf()
    print_pca_heatmap(df,True)
    centers=kmeans_model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1],
                c='black', s=200, alpha=0.5)
    fig.savefig('/home/elad/Advance data analysis/'
                'ada_project/heat_map_kmeans.png', dpi=100)
    plt.show()

def print_kmeans_model_selction(df,kmeans_model):
#Plots the different model on the data (which
#cluster does each point belongs to
    fig = plt.gcf()
    print_pca_heatmap(df,False)
    lin_model_num=kmeans_model.labels_
    plt.scatter(df['PCA 0'], df['PCA 1'],
    c=lin_model_num, cmap='brg')
    fig.savefig('/home/elad/Advance data analysis/'
    'ada_project/model_selction_kmeans.png', dpi=100)
    plt.show()

def create_df_pca_z_model(df,kmeans_model):
#adding a model column to our DF
    return pd.concat([df, pd.DataFrame(kmeans_model.
            labels_,columns=['Model'])], axis=1)

def select_model_data(df,m):
    return df[df['Model'] == m]

def separate_data_target(df):
    return df[['PCA 0', 'PCA 1', 'PCA 2']],  df[['z']]

def linear_fit_1kmean(df,m,regr):
#creates a linear fit for one of the models
#around cluster m
    x, y =separate_data_target(select_model_data(df, m))
    return regr.fit(x,y)

def linear_fit_all(df,size_models):
#approximate a linear fit mode for data points around
#a certain cluster
    linear_model_fits=[]
    for i in range(size_models):
        regr = linear_model.LinearRegression()
        linear_model_fits.\
            append(linear_fit_1kmean(df, i, regr))
    return linear_model_fits

def find_closest_cluster_pca_only\
    (test_set_point,kmeans_model,size_centers):
#find for each of the test set which
#cluster does he belongs
    centers=kmeans_model.cluster_centers_
    no_z_centers = np.delete(centers,3,1)
    test_set_point=([test_set_point,]*size_centers)
    sub=np.subtract(no_z_centers,test_set_point)
    sq=np.square(sub)
    sum_sq=np.sum(sq,axis=1)
    min_value=np.min(sum_sq)
    min_index=np.where(sum_sq == min_value)
    return min_index[0]

def predictions_test_set(linear_aprox_array,
    kmeans_model,test_set,size_centers):
    #This function uses the linear models
    # that were found to return prediction and
    #relative error between the prediction
    #and the real value from training set
    prediction_for_t=[]
    features=['PCA 0', 'PCA 1', 'PCA 2']
    test_set_array = test_set.loc[:, features].values
    for t in test_set_array:
        closest_cluster_index=\
            find_closest_cluster_pca_only\
            (t, kmeans_model, size_centers)
        prediction_for_t.append\
        (float(linear_aprox_array
        [int(closest_cluster_index)]
        .predict([t])))
    pd_pred = pd.DataFrame(np.asarray(prediction_for_t),index=test_set.index.values ,columns=['Prediction'])
    pd_return=pd.concat([test_set, pd_pred],axis=1)
    rel_err = np.absolute(pd_return['z'] -
    pd_return['Prediction']) * 100 / pd_return['z']
    pd_return = pd_return.assign(Error=rel_err)
    return pd_return

def kde_pdf(df,z_axis):
#using a python KDE to find the pdf of both
#z and the prediciton
    kde = stats.gaussian_kde(df['z'])
    pdf = kde.evaluate(z_axis)
    kde_pred = stats.\
    gaussian_kde(df['Prediction'])
    pdf_pred = kde_pred.evaluate(z_axis)
    return pdf, pdf_pred

def print_pdf_kde(pdf,pdf_pred,z_axis):
#This is only to print the results
    fig = plt.gcf()
    plt.plot(z_axis, pdf)
    plt.plot(z_axis, pdf_pred, color='red')
    plt.title('Pdf test set (blue) and prediciton (red)',
        fontweight='bold', color='black', fontsize='16',
        horizontalalignment='center')
    plt.xlabel('z', fontweight='bold',
        color='black', fontsize='13',
        horizontalalignment='center')
    plt.ylabel('PDF', fontweight='bold',
        color='black', fontsize='13',
        horizontalalignment='center')
    fig.savefig('/home/elad/Advance data analysis/'
        'ada_project/pdf_pred_org.png', dpi=100)
    plt.show()

def print_RMSE_z_pred(df,RMSE):
#print of the graph with the error bars
    fig = plt.gcf()
    df = df.sort_values(by=['z'])
    sub=df['z']-df['Prediction']
    plt.errorbar(df['z'], sub,
    yerr=RMSE,fmt='o')
    plt.plot([0, 2],[0, 0],
    linewidth=4, color='red')
    plt.title('z (redshift) vs error ',
    fontweight='bold', color='black', fontsize='16',
    horizontalalignment='center')
    plt.xlabel('z', fontweight='bold',
    color='black', fontsize='13',
    horizontalalignment='center')
    plt.ylabel('Error', fontweight='bold',
    color='black', fontsize='13',
    horizontalalignment='center')
    plt.show()
    fig.savefig('/home/elad/Advance data analysis/'
    'ada_project/error_bar_RMSE.png', dpi=100)

def print_elbow_method(df,):
# This part will find the number of clusters
# needed using the elbow method
    fig = plt.gcf()
    Nc = range(1, 25)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    kmeans
    score = [kmeans[i].fit(df).score(df)
             for i in range(len(kmeans))]
    score
    plt.plot(Nc, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()
    fig.savefig('/home/elad/Advance data analysis/'
    'ada_project/elbow_method.png', dpi=100)


def prediction_high_z(analysis,df_high_z,z_cutoff):
    kde = stats.gaussian_kde(df_high_z['z'])
    for t in analysis[analysis['Prediction']>z_cutoff]['Prediction']:
        high_z_axis=np.linspace(t,2,100)
        pdf=kde.evaluate(high_z_axis)
        pdf = pdf / np.sum(pdf)
        prediction = np.random.choice(high_z_axis, 1, True, pdf)
        prediction=prediction[0]
        mask=analysis.Prediction == t
        column_name='Prediction'
        analysis.loc[mask, column_name] = prediction
    return analysis

def print_heatmap_cov_correlation(df,title,file_name):
    fig = plt.gcf()
    fig.suptitle(title,verticalalignment='top',size='small')
    sns.heatmap(df, cmap='plasma',)
    fig.savefig('/home/elad/Advance data analysis/ada_project/'+file_name)
    plt.show()


def find_errors(df_pca_z_model,array_of_lin_models,covarince_error,correlation_pca_feature,component_num,size_cluster):
    meas_error=np.zeros(len(array_of_lin_models))
    correlation_pca_feature_shape = np.shape(correlation_pca_feature)
    for l in range(len(array_of_lin_models)):
        for i in range(component_num):
            for j in range(correlation_pca_feature_shape[1]):
                for m in range(array_of_lin_models[l].coef_.size):
                    meas_error[l]+= array_of_lin_models[l].coef_[0][m] * correlation_pca_feature.iloc[i, j] * covarince_error.iloc[i, j]
    meas_error=np.sqrt(meas_error)
    RMSE = np.zeros(size_cluster)
    score_models=np.zeros(size_cluster)
    for m in range(size_cluster):
        temp=df_pca_z_model[df_pca_z_model['Model'] == m]
        data = df_pca_z_model[df_pca_z_model['Model'] == m].loc[:, ['PCA 0', 'PCA 1', 'PCA 2']].values
        real_val = df_pca_z_model[df_pca_z_model['Model'] == m].loc[:, ['z']].values
        prediction = array_of_lin_models[m].predict(data)
        RMSE[m] = np.sqrt(mean_squared_error(prediction,real_val))
        score_models[m]=array_of_lin_models[m].score(data,real_val)
    df=pd.DataFrame({"Measurement error":meas_error,'RMSE':RMSE, 'Score':score_models })
    return df


#Main body of the program start here
#importing data and assigning what
# are the features (the DF without the errors)
component_num=3
size_cluster=8
df=pd.read_csv('/home/elad/Advance data analysis'
'/ada_project/Redshift_table.csv')
df=df[df['IRAC_1_err']!=-1]
df=df[df['IRAC_2_err']!=-1]
features=['up','gp','rp', 'ip', 'zp','Y',
'J', 'H', 'K', 'IRAC_1', 'IRAC_2']
sd_data, df_pca=df_pca_only(df,
features,component_num)
df_pca_z=df_pca_with_z(df_pca, df)
pca_model, pca_array=pca_results(sd_data,
features,np.size(features))
print_pca_data(pca_model, df_pca_z,
np.size(features))


kmeans_model = KMeans\
(n_clusters=size_cluster).fit(df_pca_z)
train_set, test_set =\
train_test_split(df_pca_z,test_size=0.2)
print_kmeans_heatmap(train_set, kmeans_model)
print_kmeans_model_selction\
(df_pca_z, kmeans_model)

df_pca_z_model = \
create_df_pca_z_model(df_pca_z, kmeans_model)
array_of_lin_models=\
linear_fit_all(df_pca_z_model,size_cluster)

#Bellow, analysis, is the most
# important piece of the work. It answer the research question

analysis = predictions_test_set\
(array_of_lin_models, kmeans_model, test_set, size_cluster)


kde = stats.gaussian_kde(analysis['z'])
z_axis = np.linspace(0, 2, 110)
pdf_test_set, pdf_prediction = \
kde_pdf(analysis,z_axis)
print_pdf_kde(pdf_test_set,
pdf_prediction, z_axis)
RMSE=np.sqrt(mean_squared_error
(analysis['z'],analysis['Prediction']))
print("The RMSE is: ", RMSE)
print_RMSE_z_pred(analysis,RMSE)
df_z_mean, df_z_std = np.mean(df['z']),np.std(df['z'])

print('The mean and the std of z are: ',
df_z_mean,' ',df_z_std)
print('The mean and the STD of the test set are: '
, np.mean(test_set['z']), np.std(test_set['z']))
print("The mean and the STD of the prediction are: ",
np.mean(analysis['Prediction']), np.std(analysis['Prediction']))
print("The delta between mean and STD are: ",
np.absolute(np.mean(analysis['Prediction'])-np.mean(test_set['z'])),
np.absolute(np.std(analysis['Prediction'])-np.std(test_set['z'])))
print("And in precentage: ",
np.absolute(np.mean(analysis['Prediction'])-
np.mean(test_set['z']))/np.mean(test_set['z'])*100,
np.absolute(np.std(analysis['Prediction'])-
np.std(test_set['z']))/np.std(test_set['z'])*100)

print('The p value is: ',stats.ttest_ind(analysis['z'],analysis['Prediction']))
print_elbow_method(df_pca_z)




z_cutoff=1.3
df_high_z=df_pca_z[df_pca_z['z']>z_cutoff]
analysis=prediction_high_z(analysis,df_high_z,z_cutoff)
print_RMSE_z_pred(analysis,RMSE)
print('The high redshift (z) std is ', np.std(df_high_z['z']))


correlation_matrix_data=pd.DataFrame(np.cov(np.transpose(sd_data)),columns=features,index=features)

covariance_data=pd.DataFrame(np.cov(np.transpose(df[features])),columns=features,index=features)
errors= ['up_err','gp_err','rp_err', 'ip_err', 'zp_err','Y_err','J_err',
         'H_err', 'K_err', 'IRAC_1_err', 'IRAC_2_err']
covariance_error = pd.DataFrame(np.cov(np.transpose(df[errors])),columns=errors,index=errors)
sd_errors = StandardScaler().fit_transform(df[errors])
correlation_errors= pd.DataFrame(np.cov(np.transpose(sd_errors)),columns=errors,index=errors)
print_heatmap_cov_correlation(correlation_matrix_data,"Correlation data",'cov_matrix.png')
print_heatmap_cov_correlation(covariance_data,"Covariance data",'cov_matrix.png')
print_heatmap_cov_correlation(covariance_error,"Covariance error",'cov_matrix.png')
print_heatmap_cov_correlation(correlation_errors,"Correlation errors",'cov_matrix.png')
df_comp = pd.DataFrame(pca_model.components_,columns=features,index=np.core.defchararray.add(['PC '],
            map(str,range(11))))
print_heatmap_cov_correlation(df_comp,"PC vs feauters",'pc_vs_fetuares.png')



temp=array_of_lin_models[0].coef_
#this is for a particular model
dim_df_comp=np.shape(df_comp)
temp_model=array_of_lin_models[0]
delta_z=0.0
temp=df_comp.iloc[1,2]
for i in range(3):
    for j in range(dim_df_comp[1]):
        for m in range(temp_model.coef_.size):
            delta_z+=temp_model.coef_[0][m]*df_comp.iloc[i,j]*covariance_error.iloc[i,j]
# temp2 = find_errors(df_pca_z_model,array_of_lin_models,covariance_error,df_comp,component_num)
# for m in range(size_cluster):
#     temp_df=df_pca_z_model['Model'==m]
#     RMSE = np.sqrt(mean_squared_error
#                    (analysis['z'], analysis['Prediction']))

# temp_model.score()
df_errors= find_errors(df_pca_z_model,array_of_lin_models,covariance_error,
                   df_comp,component_num,size_cluster)


