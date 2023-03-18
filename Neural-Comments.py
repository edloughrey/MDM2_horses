
import numpy as np
print("meow1")

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    ''' Multi-layer perceptron class. '''
    print("meow2")

    def __init__(self, *args): #Self - all the values defined here can be passed on and accessed using self, *args allows to pass multiple arguments to the function
        ''' Initialization of the perceptron with given sizes.  '''
        self.shape = args #dimensions stored as a turple in args
        n = len(args) #turple size 
        print("meow5")
        
        # Build layers
        self.layers = [] 
        
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1)) #add the array with the size of the first dimensions + 1 written as ones, e.g. for dim2 = array([1.,1.,1.])

        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i])) #add more arrays to the main array with the rest of the dimensions. [array([]), array([])...]
            
        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size, #make matrices of 0s in the dimensions of the layers
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()
        
    def reset(self):
        ''' Reset weights '''
        print("meow6")
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25 #actually assignes the values to the matrix
            
    def propagate_forward(self, data): 
        ''' Propagate data from input layer to output layer. '''
        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1])) #dot product of the two matricies using sigmoid (function specifically for the neural networks)
        
        # Return output
        return self.layers[-1] 


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    print("meow3")
    
    #Program starts
    def learn(network,samples, epochs=2500, lrate=.1, momentum=0.1):
        # Train 
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate, momentum )
        # Test
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            print (i, samples['input'][i], '%.2f' % o[0]),
            print ('(expected %.2f)' % samples['output'][i])
        print

    print("meow4")
    network = MLP(2,2,1) #passed as self to __init__
    samples = np.zeros(4, dtype=[('input',  float, 2), ('output', float, 1)])


#------------------------------------------------------------------------------
    print ("Learning the sin function")
    network = MLP(1,10,1) #pass as self again
    samples = np.zeros(500, dtype=[('x',  float, 1), ('y', float, 1)])
    samples['x'] = np.linspace(0,1,500)
    samples['y'] = np.sin(samples['x']*np.pi) #create the axis 

    print("meow7")
    for i in range(10000):
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['x'][n]) #not sure what this does
        network.propagate_backward(samples['y'][n])

    print("meow8")
    plt.figure(figsize=(10,5))
    # Draw real function
    x,y = samples['x'],samples['y']
    plt.plot(x,y,color='b',lw=1)
    # Draw network approximated function
    for i in range(samples.shape[0]):
        y[i] = network.propagate_forward(x[i]) #does the calculation for y 
    plt.plot(x,y,color='r',lw=3)
    plt.axis([0,1,0,1])
    plt.show()