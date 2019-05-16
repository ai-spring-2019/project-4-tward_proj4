#"C:\Users\clipp\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.7\Python 3.7 (64-bit).lnk" C:\Users\clipp\Documents\AI\project-4-tward_proj4\project4.py
"""
PLEASE DOCUMENT HERE

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math, copy, operator
from collections import deque
from functools import reduce


ACTUAL_FILE = sys.argv[1]
NUM_EPOCHS = 500
if len(sys.argv) > 2:
    NUM_EPOCHS = sys.argv[2]

FILENAME = ('C:%sUsers%sclipp%sDocuments%sAI%sproject-4-tward_proj4%s%s' % (chr(92), chr(92), chr(92), chr(92), chr(92), chr(92), ACTUAL_FILE))



class NNNode:
    def __init__(self, layer):
        # 0 is input layer
        self.layer = layer
        self.connected_nodes = []
        self.dummy_weight = random.uniform(-2, 2)
        self.weights = []
        self.parents = []
        self.activation = None

    def init_children(self, lst):
        self.connected_nodes = lst
        self.weights = list(zip(self.connected_nodes, 
                                [random.random() for _ in range(len(lst))]))
        print(self)

    def __str__(self):
        return ("Im a node on layer %d with the following weights: %s" % (self.layer, self.weights))


class NeuralNetwork:
    def __init__(self, structure=[1, 1], epochs=100):

        # Here we have the information provided when the constructor is called
        # Store structure of network and the number of epochs to run
        self.structure = structure
        self.epochs = epochs

        # Here we store the various weights in an adjacency matrix
        tot_nodes = sum(structure)
        self.weight_graph = zeroed_grid(tot_nodes)
        self.dummy_weights = []
        for i in range(len(self.structure)):
            for _ in range(self.structure[i]):
                if i == 0:
                    self.dummy_weights.append(0)
                else:
                    self.dummy_weights.append(random.uniform(-3, 3))

        self.inputs = []
        self.layers = []
        self.outputs_values = []

        self.node_count = 0

        curr_start = 0
        next_start = 0

        for l in range(len(structure)):
            self.layers.append([])
            curr_start = next_start
            next_start += structure[l]
            current_layer_size = structure[l]

            # Handle adding things for last layer.
            if l == (len(structure) - 1):
                for _ in range(current_layer_size):
                    self.layers[-1].append(self.node_count)
                    self.node_count += 1
                break
            
            # any layer thats not the last one
            else:
                next_layer_size = structure[l + 1]
                for i in range(current_layer_size):

                    # Keep track of what nodes are in what layers
                    self.layers[-1].append(self.node_count)

                    # Keep track of input layer
                    if l == 0:
                        self.inputs.append(i)

                    for j in range(next_layer_size):
                        # print("Since I'm fucking dumb let's do this shit the hard way.\n"
                        #     "i is %d, j is %d, curr_start is %d, next_start is %d.\n"
                        #     "The dimensions of the weight graph is %dx%d\n" % (i, j, curr_start, next_start, len(self.weight_graph), len(self.weight_graph[0])))
                        self.weight_graph[i + curr_start][j + next_start] = random.random()

                    self.node_count += 1
        # print("Okey dokey my dudes we just initialized a hell of a neural net!! Here's some cool info about this bad boi:\n"
        #       "The NN as defined by its weights:\n%s\nHere it is as defined by its layers:\n%s" % (str(self), reduce(lambda x, y: str(x) + "\n" + str(y), self.layers)))

    def get_vectors_forward(self, node_num, prev_layer, activations):
        v1 = []
        v2 = []
        for elt in prev_layer:
            v1.append(self.weight_graph[elt][node_num])
            v2.append(activations[elt])
#       print("Forward prop vectors: %s, %s" % (v1, v2))

        return v1, v2

    def get_vectors_backward(self, node_num, next_layer, deltas):
        v1 = []
        v2 = []

        for elt in next_layer:
            v1.append(self.weight_graph[node_num][elt])
            v2.append(deltas[elt])

#       print("Back prop vectors: %s, %s" % (v1, v2))
        return v1, v2

    def forward_propagate(self, data):
        activations = []
        for i in range(len(self.inputs)):
            activations.append(data[i])


        #potentially should count down to highest value in current layer istead of actual node number!!
        # Calculate activations for all other layers
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for node_num in layer:
                w_vector, a_vector = self.get_vectors_forward(node_num, self.layers[i - 1], activations)
                in_j = dot_product(w_vector, a_vector) + self.dummy_weights[node_num]
                # if i == len(self.layers) - 1:
                    # print(logistic(in_j))
                activations.append(logistic(in_j))

        self.outputs_values = [activations[n] for n in self.layers[-1]]
        # print("Output values are", self.outputs_values)

    def predict_class(self):
        prediction = self.outputs_values[0]
        print("I predict %f" % prediction)
        return round(prediction)

    def back_propagation_learning(self, training):
        alpha = .1
        num_nodes = self.layers[-1][-1] + 1
        for epoch in range(self.epochs):
            # print("We're on epoch %d\n%s" % (epoch, str(self)))
            # print("We're on epoch %d" % epoch)
            for x, y in training:
                activations = []
                deltas = [0] * num_nodes

                # Initialize activations for the input layer
                for i in range(len(self.inputs)):
                    activations.append(x[i])


                #potentially should count down to highest value in current layer istead of actual node number!!
                # Calculate activations for all other layers
                for i in range(1, len(self.layers)):
                    layer = self.layers[i]
                    for node_num in layer:
                        w_vector, a_vector = self.get_vectors_forward(node_num, self.layers[i - 1], activations)
                        in_j = dot_product(w_vector, a_vector) + self.dummy_weights[node_num]
                        activations.append(logistic(in_j))

                # Initialize delta for output layer
                for i in range(len(self.layers[-1])):
                    n = self.layers[-1][i]
                    # print("Fucker you have an issue here bc n is %d and deltas is %d things long" % (n, len(deltas)))
                    delta = activations[n] * (1 - activations[n]) * (y[i] - activations[n])
                    deltas[n] = delta


                # Calculate deltas for all other layers
                for i in range(len(self.layers) - 2, -1, -1):
                    # print("Current layer is %d" % i)
                    # print("Prev layer is %d" % (i+1))
                    layer = self.layers[i]
                    for node_num in layer:
#                       print("Getting error for node %d" % node_num)
                        w_vector, d_vector = self.get_vectors_backward(node_num, self.layers[i + 1], deltas)
                        dot = dot_product(w_vector, d_vector)
                        delta = activations[node_num] * (1 - activations[node_num]) * dot
                        deltas[node_num] = delta


                # Use deltas to update weights in the weights grid
                for layer_num in range(len(self.layers) - 1):
                    layer = self.layers[layer_num]
                    next_layer = self.layers[layer_num + 1]
                    for i in layer:
                        for j in next_layer:
                            self.weight_graph[i][j] = self.weight_graph[i][j] + alpha * activations[i] * deltas[j]

                # Update dummy weight list
                for layer in range(1, len(self.layers)):
                    for num in self.layers[layer]:
                        self.dummy_weights[num] = self.dummy_weights[num] + alpha * 1 * deltas[num]

                # print(len(activations), len(deltas))



                #print(deltas)
            # alpha = alpha * .95

    def __str__(self):
        the_string = "Our dummy weights are %s" % str(self.dummy_weights))
        the_string += "Layers:\n"
        for l in self.layers:
            the_string += str(l)
        for thing in self.weight_graph:
            for thing2 in thing:
                the_string += "%04f " % thing2
            the_string += "\n"
        return the_string

def zeroed_grid(dimension):
    grid = []
    for _ in range(dimension):
        grid.append([])
        for _ in range(dimension):
            grid[-1].append(0)
    return grid

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    # print("File name should still be %s" % FILENAME)
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom


def crossvalidate(testing, k=10, network_structures=[]):
    assert k >= 3

    # Copy testing
    testing = copy.deepcopy(testing)
    groups = []
    n = len(testing)
    x, y = testing[0]

    # make k groups with random data from our full dataset in each one
    for i in range(len(testing)):
        if i % k == 0 and len(groups) < k:
            groups.append([])
        groups[-1].append(testing.pop(random.randrange(len(testing))))

    avg_accuracy = 0
    best_accuracy = (-float('inf'), None)
    network_idx = 0

    # We will find the best of 5 networks
    while len(network_structures) < 5:
        network_structures.append(generate_hidden_layers(3, 10))

    # For each network structure find avg accuracy
    for struct in network_structures:
        structure = [len(x)] + struct + [len(y)]
        print("NN with structure %s in testing" % structure)
        nn = None

        # Make sure to give each dataset a chance to be the accuracy set
        for _ in range(k):

            # Refresh the Neural net to clear the weights
            nn = NeuralNetwork(structure, NUM_EPOCHS)
 #           best_network = groups[0]

            # Create set to check the results
            acc_group = groups[0] + groups[1]

            # Create test data set
            testing = []
            for group in groups[2:]:
                for t in group:
                    testing.append(t)

            # for thing in testing:
            #     print(thing)

            # if network_idx < len(network_structures):
            #     structure = network_structures[i]
            #     network_idx += 1
            # else:
            #     structure = generate_hidden_layers(3, 10)


            # Run back prop
            nn.back_propagation_learning(testing)

            # Judge accuracy
            acc = accuracy(nn, acc_group)
            avg_accuracy += acc

            # Shift groups so that old test set is at the end
            groups = groups[1:] + [groups[1]]

        # Update best abd avg accuracies as needed
        avg_accuracy /= k
        if avg_accuracy > best_accuracy[0]:
            best_accuracy = (avg_accuracy, nn)
        avg_accuracy = 0

    return best_accuracy



def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1
            print("^^^ BAD!!!")

    print("welp :/ we missed %d classifications. Don't I feel foolish" % true_positives)

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here


def generate_hidden_layers(num_layers=1, nodes_per_layer=10):
    assert nodes_per_layer >= 3
    struct = []
    for _ in range(random.randint(3, num_layers)):
        struct.append(random.randrange(nodes_per_layer))
    return struct


def main():
#    header, data = read_data(sys.argv[1], ",")

    print("\n\n\n\n")
    # print("File name is: %s" % FILENAME)

    header, data = read_data(FILENAME, ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]


    acc, nn = crossvalidate(training)
    print(nn)
    print(acc)
    quit()

    cutoff = int(len(training) * .8)
    test = training[:cutoff]
    training = training[cutoff:]

    x, y = training[0]
    # print(x, y)
#   structure = [len(x)] + generate_hidden_layers(num_layers=1) + [len(y)]
    structure = [len(x)] + [4, 4] + [len(y)]
    # Check out the data:
    # for example in training:
    #     print(example)

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    nn = NeuralNetwork([3, 6, 3], 10000)
    print("Before: \n%s" % str(nn))
    nn.back_propagation_learning(training)
    # for x, y in training:
    #     nn.forward_propagate(x)
    #     print("Value:    %s\nExpected: %s\n" % (list(map(lambda x: float(round(x)), nn.outputs_values)), y))
    # quit()
    print("After: \n%s" % str(nn))
    print("Testing on %d items" % len(test))
    print(accuracy(nn, test))
if __name__ == "__main__":
    main()
