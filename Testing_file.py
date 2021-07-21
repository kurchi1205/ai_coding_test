import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D , BatchNormalization , MaxPooling2D,UpSampling2D
from keras import Model,Input
from keras.optimizers import adam_v2
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def generate_feat(encoder_layer,input_layer,imgs):
    encoder = Model(input_layer,encoder_layer)
    encoded_imgs = encoder.predict(imgs)
    encoded_img=[]
    for i in encoded_imgs:
        encoded_img.append(i.mean(axis=-1))
    
    encoded_img=np.array(encoded_img)
    encoded_img = encoded_img.reshape(encoded_img.shape[0],encoded_img.shape[1]*encoded_img.shape[2])
    return encoded_img

def generate_results(autoencoder):
    test_img_paths =os.listdir('ai_coding_test-main/test/')
    test_imgs=[]
    #Reading and processing the test images
    
    for i in test_img_paths:
        test_img = cv2.imread('ai_coding_test-main/test/'+i)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(test_img , (224,224))
        img_array = np.array(test_img,dtype=np.float32)
        img_array /= 255.0
        test_imgs.append(img_array)
        
    test_imgs=np.array(test_imgs)
    #Generating the features
    test_img_feat=generate_feat([autoencoder.layers[5].output],[autoencoder.layers[0].input],test_imgs)
    # Predicting the results
    results = pd.DataFrame(columns=['Image_src','Label'])
    f = open('thresh_sim.pckl', 'rb')
    thresh_sim = pickle.load(f)
    f.close()
    test_sim=[]
    centroid = np.load('centroid_vector.npy')
    print('Predicting the results: ')
    for i in range(len(test_img_feat)):
        test_sim.append(cosine_similarity(centroid,[test_img_feat[i]]))
    for i in range(len(test_sim)):
        if thresh_sim-test_sim[i] <=0.01:
            label='Apple'
            plt.imshow(test_imgs[i])
            plt.show()
            print(label)
        else:
            label='Not Apple'
            plt.imshow(test_imgs[i])
            plt.show()
            print(label)
        results.loc[i,'Image_src']= 'ai_coding_test-main/test/'+test_img_paths[i]
        results.loc[i,'Label']=label
        print("****************************************************")
    results.to_csv('Results.csv',index=False)
    
    
def main():
    autoencoder = load_model('best_model.hdf5')
    opt = adam_v2.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='mse')
    generate_results(autoencoder)
    
if __name__ == '__main__':
    main()

    