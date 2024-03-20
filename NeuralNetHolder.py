import ast
import pickle
from scaler import scaling,descaling
from predict import predict

class NeuralNetHolder:

    def __init__(self):
        super().__init__()

    
    def predict(self, input_r):
        
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        trained_model = open("D:/Internship/New folder (2)/Essex/NN & DL assignment/Assignment Code/Trained_Model.txt",'r')
        for r in trained_model:
            trained_model = ast.literal_eval(r)
            
        x_obj = open("D:/Internship/New folder (2)/Essex/NN & DL assignment/Assignment Code/x_obj.txt",'r')
        for r in x_obj:
            x_obj = ast.literal_eval(r)
            
        y_obj = open("D:/Internship/New folder (2)/Essex/NN & DL assignment/Assignment Code/y_obj.txt",'r')
        for r in y_obj:
            y_obj = ast.literal_eval(r)
        

        input_r_scaled = scaling([[float(x) for x in input_r.split(",")]],x_obj,0,1)
        
        # Scaling the Input Row
        #input_row_scaled = x_scaler.transform([[float(x) for x in input_row.split(",")]])
        
        #simple_nn = NeuralNetwork()
        output_row = predict(input_r_scaled[0],trained_model)
        output_row_scaled = descaling([output_row],y_obj,0,1)
        
        # Changing X with Y and Y with X
        temp = output_row_scaled[0][:]
        output_row_scaled[0][0] = temp[1]
        output_row_scaled[0][1] = temp[0]
        
        return output_row_scaled[0]        
