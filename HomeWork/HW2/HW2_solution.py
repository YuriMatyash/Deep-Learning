import numpy as np

# Hyper parameters
MEAN = 0.0  # Mean for the normal distribution
STD = 1/3  # Standard deviation for the normal distribution
ITERATIONS = 1000   # Number of iterations for training
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])  # Input dataset
Y = np.array([[0],
              [0],
              [1],
              [1]]) # Output dataset 	


# Compute the sigmoid function of x
def sigmoid(x:np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
 
# computes the derivative of sigmoid function
def sigmoid_derivative(x:np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

# weights, array with random values, mean 0, std 1/3, 3x1 shape
np.random.seed(1)
weights = np.random.normal(loc= MEAN, scale= STD, size= (3, 1))  

# Training loop
for epoc in range(ITERATIONS):
    # Forward pass
    l0 = X
    l1 = sigmoid(np.dot(l0, weights))

    # error calculation
    # calculate the difference between the network prediction and the real value of y
    l1_error = Y - l1

    # slope of sigmoid function at the values in l1
    # Multiply that difference with the sigmoid derivative
    l1_delta = l1_error * sigmoid_derivative(l1)

    # backpropagation, update weights
    # use the dot product of this number with the input layer to update your weights for the next iteration.
    weights += np.dot(l0.T, l1_delta)

    if epoc % 100 == 0:
        print(f"Error after {epoc} iterations: {np.mean(np.abs(l1_error))}")
    
    if epoc == 999:
        print()
        print("Final weights after training:")
        print(weights)
        print("final error:")
        print(np.mean(np.abs(l1_error)))


'''
Explaination for my own sanity/understanding:

the network takes the input(l0) and multiplies it by the current weights.
we run the result through the sigmoid function to squash the output between 0 and 1.
so, prediction = sigmoid(input x weights)

we then check how far off our prediction is from the actual labels, this is our error.
we compare the prediction(l1) against the actual answer(Y)
error = actual - prediction

the backpropagation step involves calculating how much we need to adjust our weights to reduce the error.

repeat this process for a set number of iterations, each time adjusting the weights slightly to minimize the error.
'''