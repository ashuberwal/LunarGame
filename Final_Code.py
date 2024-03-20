import math
from random import random,shuffle
from scaler import scaling,descaling,convert_v


# Read the files from Directories
data_training = open("C:/Essex/NN & DL assignment/Assignment Code/TRAIN_DATA.csv",'r')
training_set = []
for r in data_training:
    r = r.split(',')
    r = [float(v.replace('\n','')) for v in r]
    training_set.append(list(r))

data_validation = open("C:/Essex/NN & DL assignment/Assignment Code/TEST_DATA.csv",'r') 
vvalidating_set = []
for r in data_validation:
    r = r.split(',')
    r = [float(v.replace('\n','')) for v in r]
    vvalidating_set.append(list(r))
    
# scaling

x = list(zip(*training_set))
x = x[:2]
x = [list(x) for x in zip(*x)]

validation_x = list(zip(*vvalidating_set))
validation_x = validation_x[:2]
validation_x = [list(validation_x) for validation_x in zip(*validation_x)]

y = list(zip(*training_set))
y = y[2:]
y = [list(y) for y in zip(*y)]

validation_y = list(zip(*vvalidating_set))
validation_y = validation_y[2:]
validation_y = [list(validation_y) for validation_y in zip(*validation_y)]

x_set_obj = convert_v(x)
x_scaled = scaling(x,x_set_obj,0,1)
validation_x_scaled = scaling(validation_x,x_set_obj,0,1)

y_set_obj = convert_v(y)
y_scaled = scaling(y,y_set_obj,0,1)
validation_y_scaled = scaling(validation_y,y_set_obj,0,1)

with open('C:/Essex/NN & DL assignment/Assignment Code/x_obj.txt','w') as fp:
    fp.write(str(x_set_obj))
    
with open('C:/Essex/NN & DL assignment/Assignment Code/y_obj.txt','w') as fp:
    fp.write(str(y_set_obj))
    
scaled_data = [x+y for x,y in zip(x_scaled,y_scaled)]
scaled_valid_data = [x+y for x,y in zip(validation_x_scaled,validation_y_scaled)]

# Initialize a network
def initialization(n_inputs,n_hidden,n_no_outputs):
	neural_network = [[{'neural_networks_weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)],
               [{'neural_networks_weights':[random() for i in range(n_hidden + 1)]} for i in range(n_no_outputs)]]
	return neural_network

# Calculate neurons activation for an input
def to_activate(neural_networks_weights, inputs):
    act = neural_networks_weights[-1]
    for i in range(len(neural_networks_weights)-1):
        act += neural_networks_weights[i] * inputs[i]
    return act

# Transfer neurons act
def to_activate_function(act):
	return math.tanh(act)

# Forward propagate input to a network output
def forwd_prop(neural_network,r):
    inputs = r
    for net_lay in neural_network:
        new_inputs = []
        for neurons in net_lay:
            act = to_activate(neurons['neural_networks_weights'],inputs)
            neurons['output'] = to_activate_function(act)
            new_inputs.append(neurons['output'])
        inputs = new_inputs
    return inputs

# Transfer Derivative
def der_act_func(out):
	return (1.0 + out) * (1.0 - out)

# Error Back Propagation
def bpe(neural_network,expected):
    for i in reversed(range(len(neural_network))):
        net_lay = neural_network[i]
        errors = []
        if i != len(neural_network)-1: # delta for hidden net_lays
            for j in range(len(net_lay)):
                error = 0.0
                for neurons in neural_network[i + 1]:
                    error += (neurons['neural_networks_weights'][j] * neurons['delta'])
                errors.append(error)
        else:                   # delta for output net_lay
            for j in range(len(net_lay)):
                neurons = net_lay[j]
                errors.append(expected[j]-neurons['output'])
        for j in range(len(net_lay)):
            neurons = net_lay[j]
            neurons['delta'] = errors[j] * der_act_func(neurons['output'])
    return None

# Update the Neural network neural_networks_weights with the calculated errors
def uw(neural_network,r,lr):
    for i in range(len(neural_network)):
        inputs = r
        if i != 0:
            inputs = [neurons['output'] for neurons in neural_network[i - 1]]
        for neurons in neural_network[i]:
            for j in range(len(inputs)):
                neurons['neural_networks_weights'][j] += lr * neurons['delta'] * inputs[j]
            neurons['neural_networks_weights'][-1] += lr * neurons['delta']
    return None


# Training Neural Netowk we made for epchs
lr = 0.1
n_epochs = 1000
neural_network = initialization(2,6,2)
stop_early = 500

for epochs in range(0,n_epochs):
    sum_error = 0
    if epochs > 0:
        shuffle(scaled_data)
    for r in scaled_data:
        expected = r[-2:]
        r = r[:-2]
        outputs = forwd_prop(neural_network,r)
        sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        bpe(neural_network,expected)
        uw(neural_network,r,lr)
    if epochs == 0:
        init_eval_error = 0
        initial_epochs = epochs
        for r in scaled_valid_data:
            expected = r[-2:]
            r = r[:-2]
            outputs = forwd_prop(neural_network,r)
            init_eval_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        init_eval_error = init_eval_error/len(scaled_valid_data)
        print(f'Loss of Validation improve from infinity to {init_eval_error:.3f}...Saving Model!')
        print('>epochs=%d, lrate=%.3f, train_loss=%.3f, val_loss=%.3f' % (epochs, lr, sum_error/len(scaled_data), init_eval_error))
    else:
        curr_eval_error = 0
        for r in scaled_valid_data:
            expected = r[-2:]
            r = r[:-2]
            outputs = forwd_prop(neural_network,r)
            curr_eval_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
        curr_eval_error = curr_eval_error/len(scaled_valid_data)
        if curr_eval_error < init_eval_error:
            with open('C:/Essex/NN & DL assignment/Assignment Code/Trained_Model.txt','w') as fp:
                fp.write(str(neural_network))
                print(f'Loss of Validation improve from {init_eval_error:.3f} to {curr_eval_error:.3f}...Saving Model!')
                print('>epochs=%d, lrate=%.3f, train_loss=%.3f, val_loss=%.3f' % (epochs, lr, sum_error/len(scaled_data), curr_eval_error))
            init_eval_error = curr_eval_error
            initial_epochs = epochs
        elif epochs - initial_epochs > stop_early:
            break
        else:
            print('>epochs=%d, lrate=%.3f, train_loss=%.3f, val_loss=%.3f' % (epochs, lr, sum_error/len(scaled_data), curr_eval_error))



