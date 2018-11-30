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

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

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
	numpy.savetxt("wih.txt", self.wih)
	numpy.savetxt("who.txt", self.who)

    def load_txt(self):
	self.wih = numpy.loadtxt("wih.txt")
	self.who = numpy.loadtxt("who.txt")

input_nodes = 4
hidden_nodes = 100
output_nodes = 2

learn_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learn_rate)
n.load_txt()
#test
#test_data_file = open("test_10.csv", "r")
test_data_file = open("test_data.txt", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

print("test start:")
#scorecard = numpy.zeros(len(test_data_list))
i = 0
for record in test_data_list:
    all_values = record.split(" ")
    inputs = (numpy.asfarray(all_values[1:]))/ 100 * 0.99 + 0.1
    print("%d correct" % int(all_values[0]))
    result = n.query(inputs)
    print(result)
    pos = int(numpy.argmax(result))
    print("%d answer" % pos)
    pass
'''
    if pos == int(all_values[0]):
        scorecard[i] = 1
        pass
    else:
        scorecard[i] = 0
        pass
    i = i + 1
'''
print("test end")
#scorecard_array = numpy.asarray(scorecard)
#print(scorecard_array)
#print("performance = ", scorecard_array.sum() / scorecard_array.size)



