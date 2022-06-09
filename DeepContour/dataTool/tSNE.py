# this file is more suitable for anlysising all pic in a structured folder 
#the following has two examples of structured data folder 
#all_dir_list  =  glob.glob("D:/Deep learning/dataset/original/*/*/pic_all/*.jpg")
#all_dir_list  =  glob.glob("D:/Deep learning/dataset/label data/img/*/*.jpg")
#all_dir_list  =  glob.glob("D:/Deep learning/dataset/label data/img_genetate_devi/*/*.jpg")
#TODO: test this tsne code for new data
from __future__ import print_function
import time

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import cv2
import os
import glob
# Fetch the minist dataset 
#mnist = fetch_mldata("MNIST original")
class Data_VIS (object):
    def __init__(self):
        dir = "D:/Deep learning/dataset/original/animal_tissue/1/pic_all/"
        dir = "D:/Deep learning/dataset/For_contour_sheath_train/train_OLG/pic/1/"
        dir = "D:/Deep learning/dataset/original/phantom/2/pic_all/"
         
        # to go through this big folder in the fixed format:
        # the "*" indicate that it can be any word for it  
        #all_dir_list  =  glob.glob("D:/Deep learning/dataset/original/*/*/pic_all/*.jpg")
        #all_dir_list  =  glob.glob("D:/Deep learning/dataset/label data/img/*/*.jpg")
        all_dir_list  =  glob.glob("D:/Deep learning/dataset/label data/img_genetate_devi/*/*.jpg")


        
        len_all = len(all_dir_list)
        this_image = []
        S = 100
        self.S  = S
        sample_space  = 1 
        buffer_L = int(len_all/sample_space)
        # set the buffer len as the 
        self. data_stack =  np.zeros ((buffer_L, S*S))
        self. label_stack = np.zeros (buffer_L)

        # thsi is dir_list of a specific folder 
        #read_sequence = all_dir_list  # all item stringin this folder

        ##read_sequence = os.listdir(dir) # all item stringin this folder
        #seqence_Len = len(read_sequence)
        #sample_space = int ((seqence_Len  - 1 )  / buffer_L)

        # read image by image 

        #initialized the first type
        self.class_num = 1  # folder type number
        first_dir  = all_dir_list[0]
        pre_sub_f = first_dir. split ('\\') [1]
        for i in range(buffer_L):
            # read and tranform
            this_dir =  all_dir_list[i*sample_space] # sample with a  space inver
            this_image = cv2. imread(this_dir)
            this_image  =   cv2.cvtColor(this_image, cv2.COLOR_BGR2GRAY)
            this_image = cv2.resize(this_image, (S,S), interpolation=cv2.INTER_AREA)
            cv2.imshow('combin video',this_image) 
            cv2.waitKey(1)

            # save the stack 
            flat_img = this_image.flatten()  # flat the image 
            self. data_stack[i,:] = flat_img
            # generate a fake lable 
            # get the sub folder name to determine the label 

            sub_f_l = this_dir. split ('\\')
            sub_f = sub_f_l  [1] 
            # check the class is switched or not 
            if (sub_f != pre_sub_f):
                self.class_num +=1
                pre_sub_f=  sub_f

            self.label_stack [i] =  str(self.class_num)
            

        #self. data_stack =  np.zeros ((buffer_L, S*S))
        #self. label_stack = np.zeros (buffer_L)

        ## thsi is dir_list of a specific folder 
        #read_sequence = os.listdir(dir) # all item stringin this folder
        #seqence_Len = len(read_sequence)
        #sample_space = int ((seqence_Len  - 1 )  / buffer_L)

        ## read image by image 
        #for i in range(buffer_L):
        #    # read and tranform
        #    this_dir = dir + read_sequence[i*sample_space] # sample with a  space inver
        #    this_image = cv2. imread(this_dir)
        #    this_image  =   cv2.cvtColor(this_image, cv2.COLOR_BGR2GRAY)
        #    this_image = cv2.resize(this_image, (S,S), interpolation=cv2.INTER_AREA)
        #    cv2.imshow('combin video',this_image) 
        #    cv2.waitKey(1)

        #    # save the stack 
        #    flat_img = this_image.flatten()  # flat the image 
        #    self. data_stack[i,:] = flat_img
        #    # generate a fake lable 
        #    self.label_stack [i] = '0'
        
        # Equal space resample from a folder 
 

    def visualization(self):
        number_lable = self.class_num  # how may types of color labels 
        
        X  = self.data_stack/255.0
        y = self . label_stack 

        print(X.shape, y.shape)

        # convert it to the panda format 
        feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]  # 784 
        df = pd.DataFrame(X,columns=feat_cols)
        df['y'] = y
        df['label'] = df['y'].apply(lambda i: str(i))
        X, y = None, None
        print('Size of the dataframe: {}'.format(df.shape))

        # For reproducability of the results
        np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])
        #We now have our dataframe and our randomisation vector. 
        #Lets first check what these numbers actually look like.
        #To do this we’ll generate 30 plots of randomly selected images.
        plt.gray()
        fig = plt.figure( figsize=(16,7) )
        for i in range(0,15):
            ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
            ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((self.S,self.S)).astype(float))
        plt.show()

        #Dimensionality reduction using PCA

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df[feat_cols].values)
        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1] 
        df['pca-three'] = pca_result[:,2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", number_lable),
            data=df.loc[rndperm,:],
            legend="full",
            alpha=0.3
        )
        #Explained variation per principal component: [0.09746116 0.07155445 0.06149531]
        #For a 3d-version of the same plot
        #ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        #ax.scatter(
        #    xs=df.loc[rndperm,:]["pca-one"], 
        #    ys=df.loc[rndperm,:]["pca-two"], 
        #    zs=df.loc[rndperm,:]["pca-three"], 
        #    c=df.loc[rndperm,:]["y"], 
        #    cmap='tab10'
        #)
        #ax.set_xlabel('pca-one')
        #ax.set_ylabel('pca-two')
        #ax.set_zlabel('pca-three')
        #plt.show()

        #2 T-Distributed Stochastic Neighbouring Entities (t-SNE)

        # first run the PCA again on the subset
        N = 10000
        df_subset = df.loc[rndperm[:N],:].copy()
        data_subset = df_subset[feat_cols].values
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data_subset)
        df_subset['pca-one'] = pca_result[:,0]
        df_subset['pca-two'] = pca_result[:,1] 
        df_subset['pca-three'] = pca_result[:,2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        #x 
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(data_subset)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        #Now that we have the two resulting dimensions we
        # can again visualise them by creating a scatter plot
        #  of the two dimensions and coloring each sample by its respective label.
        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        #plt.figure(figsize=(16,10))
        #sns.scatterplot(
        #    x="tsne-2d-one", y="tsne-2d-two",
        #    hue="y",
        #    palette=sns.color_palette("hls", 10),
        #    data=df_subset,
        #    legend="full",
        #    alpha=0.3
        #)
        #This is already a significant improvement over the PCA visualisation we used earlier.
        # We can see that the digits are very clearly clustered in their own sub groups.
        #  If we would now use a clustering algorithm to pick out the seperate clusters 
        #  we could probably quite accurately assign new points to a label. Just to compare PCA & T-SNE:
        plt.figure(figsize=(16,7))
        ax1 = plt.subplot(1, 2, 1)
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("hls", number_lable),
            data=df_subset,
            legend="full",
            alpha=0.3,
            ax=ax1
        )
        ax2 = plt.subplot(1, 2, 2)
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", number_lable),
            data=df_subset,
            legend="full",
            alpha=0.3,
            ax=ax2
        )
        time.sleep(0.1)
        time.sleep(0.1)

        pass


if __name__ == '__main__':
    data_shower = Data_VIS()
    data_shower.visualization()
    time.sleep(0.1)

    # get the data from my folder 


   
    #pause(0.01) 

    #We’ll now take the recommendations to heart and actually reduce the number of dimensions before feeding the data into the t-SNE algorithm.
    # For this we’ll use PCA again. We will first create a new dataset containing the fifty dimensions generated by the PCA reduction algorithm.
    #  We can then use this dataset to perform the t-SNE on
    #pca_50 = PCA(n_components=50)
    #pca_result_50 = pca_50.fit_transform(data_subset)
    #print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))