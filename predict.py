import math

# Calculate neurons activation for an input
def to_activate(neural_networks_weights, inputs):
    act = neural_networks_weights[-1]
    for i in range(len(neural_networks_weights)-1):
        act += neural_networks_weights[i] * inputs[i]
    return act

# Transfer neurons act
def to_activate_function(act):
	return math.tanh(act)


def predict(i_row,trained_model):
    inputs = i_row
    for net_lay in trained_model:
        new_inputs = []
        for neurons in net_lay:
            act = to_activate(neurons['neural_networks_weights'],inputs)
            neurons['output'] = to_activate_function(act)
            new_inputs.append(neurons['output'])
        inputs = new_inputs
    return inputs