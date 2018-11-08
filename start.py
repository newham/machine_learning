import neural_network as nn

from matplotlib import pyplot
import numpy
import dataset

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.5


def test_dataset():
    n = nn.NeuralNetwork(input_nodes, hidden_nodes,
                         output_nodes, learning_rate)
    # output = n.query([1, 0.5, -1.5])
    # print(output)

    data_list = dataset.get_data_list("mnist_dataset/mnist_train_100.csv")
    data = data_list[22]
    # image_array = dataset.get_img_array(data)
    # print("label", dataset.get_img_label(data))
    # pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    # pyplot.show()

    scaled_input = dataset.get_scaled_data(data)
    print(scaled_input)
    pass


def test_output():
    data_list = dataset.get_data_list("mnist_dataset/mnist_train_100.csv")
    data = data_list[22]
    all_values = dataset.get_all_values(data)
    onodes = 10
    targets = numpy.zeros(onodes)+0.01
    targets[int(all_values[0])] = 0.99

    print(targets)
    pass


def test_nn():

    input_nodes = 28*28
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.15

    n = nn.NeuralNetwork(input_nodes, hidden_nodes,
                         output_nodes, learning_rate)

    training_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_train_100.csv")

    print("start to train")
    for record in training_data_list:
        inputs = dataset.get_scaled_data(record)
        targets = numpy.zeros(output_nodes)+0.01
        all_values = dataset.get_all_values(record)
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    print("done")

    # test_record = training_data_list[11]
    # result = n.query(dataset.get_scaled_data(test_record))
    # print(dataset.get_img_label(test_record))
    # print(result)
    print("start to test")
    test_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_test_10.csv")
    scorecard = []
    right = 0
    total = 0
    for record in test_data_list:
        outputs = n.query(dataset.get_scaled_data(record))
        result = numpy.argmax(outputs)
        label = dataset.get_img_label(record)
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        total += 1

    print(scorecard)
    print("right rate=", right/total*100, "%")


# test_dataset()
test_nn()
