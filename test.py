import dataset
import random
import neural_network as nn
import numpy


def test_load_img():
    img_name = "img/6.png"
    img_data = dataset.load_img(img_name)
    print(img_data)

def test_img():
    all_data = dataset.get_data_list("mnist_dataset/mnist_train_100.csv")
    img_label, img_array = dataset.get_img_data(
        all_data[random.randint(0, 99)])
    print(img_label)
    print(img_array)
    dataset.show_img(img_array)
    img_name = "img/"+str(img_label)+".png"
    dataset.save_img(img_array, img_name)

    dataset.show_img(dataset.load_img(img_name))
    pass


def test_query():
    img_name = "img/6.png"
    img_data = dataset.load_img(img_name)
    print(img_name)

    input_nodes = 28*28
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.2

    n = nn.NeuralNetwork(input_nodes, hidden_nodes,
                         output_nodes, learning_rate)

    n.load("mnist_dataset/w_input_hidden.txt",
           "mnist_dataset/w_hidden_output.txt")

    result = n.query(img_data)

    print(result)


test_query()
# test_img()
# test_load_img()
