import math
from random import random,shuffle
from scaler import scale,descale,convert


# Read the files from Directories
data_training = open("C:/Essex/NN & DL assignment/Assignment Code/Train_Game_Data.csv",'r')
training_set = []
for r in data_training:
    r = r.split(',')
    r = [float(v.replace('\n','')) for v in r]
    training_set.append(list(r))

data_validation = open("C:/Essex/NN & DL assignment/Assignment Code/Validation_Game_Data.csv",'r') 
validating_set = []
for r in data_validation:
    r = r.split(',')
    r = [float(v.replace('\n','')) for v in r]
    validating_set.append(list(r))
    
# scaling

x = list(zip(*training_set))
x = x[:2]
x = [list(x) for x in zip(*x)]

validation_x = list(zip(*validating_set))
validation_x = validation_x[:2]
validation_x = [list(validation_x) for validation_x in zip(*validation_x)]

y = list(zip(*training_set))
y = y[2:]
y = [list(y) for y in zip(*y)]

validation_y = list(zip(*validating_set))
validation_y = validation_y[2:]
validation_y = [list(validation_y) for validation_y in zip(*validation_y)]

x_set_obj = convert(x)
x_scaled = scale(x,x_set_obj,-1,1)
validation_x_scaled = scale(validation_x,x_set_obj,-1,1)

y_set_obj = convert(y)
y_scaled = scale(y,y_set_obj,-1,1)
validation_y_scaled = scale(validation_y,y_set_obj,-1,1)

with open('C:/Essex/NN & DL assignment/Assignment Code/x_obj.txt','w') as fp:
    fp.write(str(x_set_obj))
    
with open('C:/Essex/NN & DL assignment/Assignment Code/y_obj.txt','w') as fp:
    fp.write(str(y_set_obj))
    
scaled_data = [x+y for x,y in zip(x_scaled,y_scaled)]
scaled_valid_data = [x+y for x,y in zip(validation_x_scaled,validation_y_scaled)]
print(x)
print(scaled_valid_data)
print(scaled_data)


# Initialize a network
def initialization(n_no_inputs,n_no_hidden,n_no_outputs):
	neural_network = [[{"neural_networks_weights":[random() for i in range(n_no_inputs + 1)]} for i in range(n_no_hidden)],
               [{"neural_networks_weights":[random() for i in range(n_no_hidden + 1)]} for i in range(n_no_outputs)]]
	print(neural_network)
	return neural_network
    
# Calculate nuron activation for an input
def to_activate(nn_weights, inputs):
    activation = nn_weights[-1]
    for i in range(len(nn_weights)-1):
        activation += nn_weights[i] * inputs[i]
    print(activation)
    return activation

initialization(1,2,1)
        


to_activate(2,1)