import dataset
import random
import neural_network as nn
import numpy
import os


def query_img():
    input_nodes = 28*28
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.2

    n = nn.NeuralNetwork(input_nodes, hidden_nodes,
                         output_nodes, learning_rate)

    n.load("mnist_dataset/w_input_hidden.txt",
           "mnist_dataset/w_hidden_output.txt")

    img_dir = "img/"

    files = os.listdir(img_dir)
    wrong_list = []
    for img_name in files:
        img_data = dataset.load_img(img_dir + img_name)
        result = n.query(img_data)
        label = int(os.path.splitext(img_name)[0].split("_")[1])
        if result == label:
            print(img_name, "->", result, "-> √")
        else:
            print(img_name, "->", result, "-> x")
            wrong_list.append(img_name)

    print("right rate:", str(int((1-len(wrong_list)/len(files))*100)) +
          "%", "wrong list:", wrong_list)


def main():
    query_img()


if __name__ == '__main__':
    main()
