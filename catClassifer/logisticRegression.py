import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import skimage
import imageio.v2 as imageio



from PIL import Image
from scipy import ndimage
from lr_utils import *
from skimage.transform import resize


train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()


m_train = train_y.shape[1]
m_test = test_y.shape[1]
num_px = train_x_orig.shape[1]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T


test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T


train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

"""
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
"""
layers_dims = [12288, 20, 7, 5, 1] 

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    
    np.random.seed(1)
    costs = []
     
    parameters = initialize_parameters_deep(layers_dims)  
     
    for i in range(0, num_iterations):
        print(f"X.shape:{X.shape}")
        AL, caches = L_model_forward(X, parameters)
     
        cost = compute_cost(AL, Y)
     
        grads = L_model_backward(AL, Y, caches)
        
        parameters = update_parameters(parameters, grads, learning_rate) 

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost)) 
            costs.append(cost)
             
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def receiveImageInfo(image):
    my_image = resize(image, output_shape=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T 
    predictions_train = predict(my_image, 1,parameters)
    print("y = " + str(np.squeeze(predictions_train)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(predictions_train)),].decode("utf-8") +  "\" picture.")
    reuslt = "y = " + str(np.squeeze(predictions_train)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(predictions_train)),].decode("utf-8") +  "\" picture."
    return reuslt

print(f"train_x.shape:{train_x.shape}, train_y.shape:{train_y.shape}")    
'''
train_x.shape:(12288, 209), train_y.shape:(1, 209)
'''
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


