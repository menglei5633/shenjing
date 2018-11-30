import numpy
#import scipy

def expit(x):
    return 1.0 / (1.0 + numpy.exp(-x))

class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
            
        self.activation_function = lambda x: expit(x)

        self.lr = learningRate

        #self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        #self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
	self.wih = numpy.zeros((self.hnodes, self.inodes))
	self.who = numpy.zeros((self.onodes, self.hnodes))
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        targets = numpy.array(targets_list, ndmin = 2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        #compute errors
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #update weights
        self.who += self.lr * numpy.dot((output_errors * final_outputs *
            (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *
            (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass


    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin = 2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
	
    def save_txt(self):
	numpy.savetxt("wih.txt", self.wih);
	numpy.savetxt("who.txt", self.who);
   
    def load_txt(self):
	self.wih = numpy.loadtxt("wih.txt")
	self.who = numpy.loadtxt("who.txt")

input_nodes = 4
hidden_nodes = 100
output_nodes = 2

learn_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)

#train
#train_data_file = open("train_100.csv", "r")
train_data_file = open("train_data.txt", "r")
train_data_list = train_data_file.readlines()
train_data_file.close()

print("train start:")
for record in train_data_list:
    all_values = record.split(' ')
    inputs = (numpy.asfarray(all_values))/ 100 * 0.99 + 0.1
    targets = numpy.array([0.99, 0.01]);
    #targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    inputs1 = inputs + [0.1, 0.1, 0.1, 0.1]
    targets1 = [0.01, 0.99]
    n.train(inputs1, targets1)
    inputs2 = inputs - [0.1, 0.1, 0.1, 0.1]
    targets2 = [0.01, 0.99]
    n.train(inputs2, targets)
    pass
'''
for record in train_data_list:
    all_values = record.split(' ')
    inputs = (numpy.asfarray(all_values))/ 128 * 0.99 + 0.1
    targets = numpy.array([0.99, 0.01]);
    #targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass
'''

print("train end")
n.save_txt()


