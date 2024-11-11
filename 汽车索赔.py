## Machine-Learning - Car-Insurance-Claim-Prediction
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

raw=pd.read_csv("car_insurance_claim.csv")
df = raw.drop(['ID','BIRTH','OCCUPATION','CAR_TYPE','CLAIM_FLAG'], axis=1)


def structure(x):
    
    print("Mean                   :", x.mean())
    print("Median                 :", x.median())
    print("Minimum                :", x.min())
    print("Maximum                :", x.max())
    print("25th percentile of arr :", 
       np.percentile(x, 25)) 
    print("50th percentile of arr :",  
       np.percentile(x, 50)) 
    print("75th percentile of arr :", 
       np.percentile(x, 75))

#Structure of Claim Amount Data
clmamt = df.loc[:,('CLM_AMT')]
structure(clmamt)
plt.boxplot(clmamt)
plt.show()

def decisiontree(df,col):
    X = df.loc[:, ('KIDSDRIV','AGE','HOMEKIDS','YOJ','INCOME','PARENT1','HOME_VAL','MSTATUS','GENDER','EDUCATION',
                   'TRAVTIME','CAR_USE','BLUEBOOK','RED_CAR','OLDCLAIM','CLM_FREQ','REVOKED','MVR_PTS','CAR_AGE',
                   'URBANICITY')]  #independent columns
    y = df.loc[:,(col)]    #target column
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.savefig('DT.png')
    plt.show()
decisiontree(df1w.dropna(),'CLM_AMT')

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

def MSE(iterations,NNeurons):
    if __name__ == "__main__":
        #Seed the random number generator
        random.seed(1)
        # Create layer 1 (4 neurons, each with 3 inputs)
        layer1 = NeuronLayer(NNeurons, 5)
        # Create layer 2 (a single neuron with 4 inputs)
        layer2 = NeuronLayer(1, NNeurons)
        # Combine the layers to create a neural network
        neural_network = NeuralNetwork(layer1, layer2)
        #print("Stage 1) Random starting synaptic weights: ")
        neural_network.print_weights()
        # The training set. We have 7 examples, each consisting of 3 input values
        # and 1 output value.
        training_set_inputs = x_train.values
        training_set_outputs = y_train.values
        neural_network.train(training_set_inputs, training_set_outputs, iterations)
        neural_network.print_weights()
    hidden_state1,predictions = neural_network.think(x_test.values)
    mse = metrics.mean_squared_error(y_test, predictions)
    return mse
