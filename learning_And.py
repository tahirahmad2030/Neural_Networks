import numpy as np
import pandas as pd


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


np.random.seed(42)
weights = np.random.normal(scale=1 / 2**.5, size=2)
print weights
epochs = 1000
learnrate = 0.01

data= data = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1], 
                 ])
                
targets=[0,0,0,1]


for e in range(epochs):
    del_w = np.zeros(2)
    #print del_w
    i=0
    for x in (data):
        #print x
        y= targets[i]
        output = (np.dot(x, weights))
        #print output
        # The error, the target minus the network output
        error = y - output
        #print error
        #print output
        # The error term
        #   Notice we calulate f'(h) here instead of defining a separate
        #   sigmoid_prime function. This just makes it faster because we
        #   can re-use the result of the sigmoid function stored in
        #   the output variable
        error_term = error * output * (1 - output)
        
        #print error_term
        #print        
# The gradient descent step, the error times the gradient times the inputs
        #for i, j in zip(del_w, x):
         #   i = i + error_term*j
            
        del_w += error_term * x
        #print del_w
        i+=1
    weights += learnrate * del_w / 4
        
print weights

#Testing the weights, We suppose that the bias is negation twice the weights(-2*weights) 
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []
bias=0
for w in weights:
    bias+=w
bias=-bias

# Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weights[0] * test_input[0] + weights[1] * test_input[1] + bias
    output = int(linear_combination >= 0.0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))


