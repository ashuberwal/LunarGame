
# We are scaling the data between -1 and 1
def scaling(d_set,initial_obj,starting_r,ending_r):
    d_set_t = list(zip(*d_set))
    scaling = []
    for r in d_set_t:
        scaled_o_r = []
        for v in list(r):
            n_std = (v - initial_obj[d_set_t.index(r)][0]) / (initial_obj[d_set_t.index(r)][1] - initial_obj[d_set_t.index(r)][0])
            scaled_o_r.append(n_std * (ending_r-(starting_r)) + (starting_r))
        scaling.append(scaled_o_r)
    return [list(x) for x in zip(*scaling)]
    
def descaling(d_set,initial_obj,starting_r,ending_r):
    d_set_t = list(zip(*d_set))
    n_orig = []
    for r in d_set_t:
        scaled_o_r = []
        for v in list(r):
            n_std = (v - starting_r)/(ending_r - starting_r)
            scaled_o_r.append(n_std*(initial_obj[d_set_t.index(r)][1]-initial_obj[d_set_t.index(r)][0]) + initial_obj[d_set_t.index(r)][0])
        n_orig.append(scaled_o_r)
    return [list(x) for x in zip(*n_orig)]

def convert_v(d_set):
    initial_obj = {}
    d_set_t = list(zip(*d_set))
    for r in d_set_t:
        initial_obj[d_set_t.index(r)] = [min(r),max(r)]
    return initial_obj