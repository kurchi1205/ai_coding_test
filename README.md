# ai_coding_test



## Problem Statement : 
To analyze whether a test image is similar to the given train images , if it is similar then to return Apple

## Given : 
8 Images of Apple 

## Libraries Required:
- Pandas
- OS
- OpenCV
- Matplotlib
- Numpy
- Keras 
- SKLearn
- Pickle

## Solution Theory:
- To build an Autoencoder that will encode and decode the train images . 
- Once the train images are encoded , the image features can be extracted from the latent space . '
- If these feature vectors are similar to the feature vector of the test image , that implies the test image is similar to the given images 
- Then I can return the value Apple

## Steps of the solution 
- Processing the train images which includes **resizing the images** and **normalizing the pixels.**
- Developing the **Autoencoder model** with maximum of **256 features** in the latent space 
- Training the model for 50 epochs 
- Extracting the features from the latent space 
- Performing **Mean** over the features and deriving the resultant vector.
- Forming a cluster the feature vectors using Kmeans Clustering and getting the **centroid feature vector** .
- Calculating **Cosine- Similarity** between each image vector and the centroid feature vector.
- Finding the **Threshold Similarity** by calculating **mean of the similarity values of all the train images** . 
- For each of the test image features , I have calculated the cosine similarity with the centroid vector
- I have subtracted the test similarity of each image from the Threshold Similarity .
- If the **result <=0.01** , then the test image is an apple else not

#### For getting results, run Testing_file.py . The results will also be stored in Results.csv
