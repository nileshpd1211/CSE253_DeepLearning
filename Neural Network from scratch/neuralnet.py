################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np
from utility import *
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    img = ((img)/255).astype(np.float32)

    return img


def one_hot_encoding(lbl, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    d = np.zeros((lbl.max() + 1, lbl.size))
    d[lbl, np.arange(lbl.size)] = 1
    return d.T



def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    div = np.exp(x - x.max(axis=0))
    return div / div.sum(axis=0)

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.
    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid",eps=None):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type        # Placeholder for input. This will be used for computing gradients.
        self.x = None
        self.eps=None


    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        self.x = a
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        self.x = x
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.x =x
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        self.x = x
        return  x*(x>0)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return self.sigmoid(self.x)*(1-self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - (self.tanh(self.x)**2)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return 1*(self.x>0)



class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        #self.w = np.random.randn(in_units, out_units) / np.sqrt((in_units + 1)/2
        self.w = np.random.randn(in_units, out_units) / np.sqrt((in_units + 1)/2)
        self.b =  np.zeros((1,out_units)) 

        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.v_prev = 0
        self.vb_prev = 0
        self.gamma = 0
        self.reg= 0

    def __call__(self, x):
        """
        Make layer callable.
        """
        self.x = x
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        #self.a = np.dot(self.w,x)+self.b
        self.x = x
        self.a = np.matmul(x,self.w)+self.b
        
        return self.a

    def backward(self, delta, lr):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """

        self.d_w = np.matmul(self.x.T,delta)
        self.d_b = delta.reshape(-1,)
        self.d_b = np.mean(self.d_b , axis = 0)

        v = self.gamma * self.v_prev + (1 - self.gamma) * self.d_w
        self.w += (v * lr) - self.reg*(self.w)**2
        self.v_prev = v

        vb = self.gamma * self.vb_prev + (1 - self.gamma) * self.d_b
        self.b += (vb * lr) - self.reg*(self.b)**2
        self.vb_prev = vb

        #self.w += (self.d_w * lr)
        #self.b += (self.d_b * lr)

        new_delta = np.matmul(delta,self.w.T)

        return new_delta

class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.
    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None       # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable
        self.lr = config['learning_rate']

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))


    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        self.x = x
        self.targets = targets
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        self.targets = targets
        temp_out = self.x
        loss = None
        for layer in self.layers:
            out = layer.forward(temp_out)
            temp_out = out

        self.y = softmax(temp_out.T).T    #Softmax
        
        if(targets is not None):
            loss = self.loss(self.y, targets)
        
        return self.y, loss


    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''

        return -np.sum(np.multiply(targets,np.log(logits + 10e-20)))

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''

        delta = self.targets - self.y

        #Iterate over all layers in reverse order
        grad =1
        for l in range(len(self.layers)-1,-1,-1):
            if isinstance(self.layers[l],Activation):
                grad = self.layers[l].backward(delta)
            else:
                delta = self.layers[l].backward(grad*delta, self.lr)

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    epochs = config['epochs']
    lr = config['learning_rate']
    early_stop = config['early_stop']
    es_epoch = config['early_stop_epoch']

    train_loss = []
    val_loss = []
    epoch_arr = []
    acc_train=[]
    acc_val=[]
    count = 0
    stop_epoch=0
    batched_data = DataLoader(x_train, y_train, config['batch_size'])

    for epoch in range(epochs):
        epoch_arr.append(epoch)
        running_loss = 0
        run_acc = 0
        for idx, (xdata, ydata) in enumerate(batched_data):
            preds, loss = model.forward(xdata, ydata)
            model.backward()
            running_loss += loss
            run_acc += Accuracy(preds,ydata)

        running_loss /= idx
        run_acc /= idx
        train_loss.append(running_loss)
        acc_train.append(run_acc)
        print("---------- epoch {} ----------".format(epoch))
        print("Training loss: {}".format(running_loss))


        val_preds, val_loss_val = model.forward(x_valid, y_valid)
        val_loss.append(val_loss_val)
        if(epoch>0 and val_loss[-2]<val_loss[-1] and early_stop==True):
            count +=1
            if(count==es_epoch):
                stop_epoch = epoch
                print("Training stopped after {}".format(stop_epoch))
                break

        else:
            count = 0


        print("Validation loss: {}".format(val_loss_val))
        acc_val.append(Accuracy(val_preds,y_valid))

        plt.cla()
        plt.plot(epoch_arr, acc_train, label='train')
        plt.plot(epoch_arr,acc_val, label='validation')
        plt.legend(loc='upper left')
        plt.pause(0.005)


    plt.tight_layout()
    plt.show()

    plt.plot(epoch_arr, train_loss, label='train')
    plt.plot(epoch_arr, val_loss, label='validation')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def gradient_check(model1,model2,model3, x_train, y_train, eps,config):
    layer_no=0
    weight_coord = 0,8
    eps = 1e-4
    print(model1.layers[layer_no].b.shape)
    print('eps',eps)

    model1.layers[layer_no].b[0,4] += eps

    pred,loss1=model1.forward(x_train,y_train)

    model2.layers[layer_no].b[0,4] -= eps

    pred,loss2=model2.forward(x_train,y_train)

    num_grad = (loss1 - loss2)/(2*eps)
    print('num_grad',num_grad)

    #print(model3.layers[0].w[0,1])

    pred,loss3=model3.forward(x_train,y_train)
    model3.backward()

    print('back grad',model3.layers[layer_no].d_b)

    print('difference:', (abs(num_grad) -abs(model3.layers[layer_no].d_b)))


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    preds, loss = model.forward(X_test, y_test)
    accuracy = Accuracy(preds,y_test)
    print("Test Accuracy : {}".format(accuracy))

    return accuracy


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    # model  = Neuralnetwork(config)
    # # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # # Create splits for validation data here.
    x_train, y_train, x_valid, y_valid = train_val_split(x_train.T, y_train.T, 0.3)

    # # train the model
    # train(model, x_train, y_train, x_valid, y_valid, config)

    model1  = Neuralnetwork(config)
    model2  = Neuralnetwork(config)
    model3  = Neuralnetwork(config)

    sample=10
    gradient_check(model1,model2,model3,x_train[0:sample,:].reshape(sample,-1),y_train[0:sample,:].reshape(sample,-1),1e-2,config)


    # test_acc = test(model, x_test, y_test)
