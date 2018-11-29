import neural_network as nn

from matplotlib import pyplot
import numpy
import dataset
from base import print_process

# 训练神经网络并测试，采用 mnist_train_100.csv （100组）作为训练数据，mnist_test_10.csv（10组）作为测试
# 想要提高正确率，请下载mnist_train.csv（6万组）作为训练数据


def test_nn_pooling():
    # 设置初始化参数，采用的是mnist数据集，为28*28的手写数字图像，隐含层100，输出层10代表0~9的数字，学习率初始设为0.2
    row = 14
    col = 14
    input_nodes = row*col
    hidden_nodes = 50
    output_nodes = 10
    learning_rate = 0.2

    n = nn.NeuralNetwork(input_nodes, hidden_nodes,
                         output_nodes, learning_rate)

    # 第一步：开始训练
    print("start to train")
    train = True
    if train is True:
        # 训练方法1：用数据训练，采用较小的训练数据集
        training_data_list = dataset.get_data_list(
            "mnist_dataset/mnist_train.csv")
        count = 0  # 用于打印进度
        size = len(training_data_list)  # 用于打印进度
        for index, record in enumerate(training_data_list):
            label, inputs = dataset.get_scaled_data(record)
            inputs = dataset.pooling(inputs.reshape(
                (28, 28)), 2, 2).reshape(row*col)
            # print(inputs)
            targets = numpy.zeros(output_nodes)+0.01
            targets[label] = 0.99
            n.train(inputs, targets)
            # 打印进度
            print_process(index, size)

        # 将最终的权值矩阵保存
        numpy.savetxt("w_input_hidden_pooling.txt", n.w_input_hidden)
        numpy.savetxt("w_hidden_output_pooling.txt", n.w_hidden_output)
    else:
        print("load data done")
        # 训练方法2：直接导入训练的结果（适用于已经有训练结果，即权值矩阵）
        n.load("w_input_hidden_pooling.txt", "w_hidden_output_pooling.txt")

    # 第二步：开始测试训练后的神经网络
    print("start to test")
    test_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_test.csv")
    scorecard = []  # 记分牌，保存每个测试数据的测试结果
    right = 0  # 正确总数
    size = len(test_data_list)  # 用于打印进度
    for index, record in enumerate(test_data_list):
        label, inputs = dataset.get_scaled_data(record)
        inputs = dataset.pooling(inputs.reshape(
            (28, 28)), 2, 2).reshape(row*col)
        result = n.query(inputs)
        # 对比神经网络预测结果和标签
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        # 打印进度
        print_process(index, size)
    # 打印正确率
    print("right rate=", right/len(test_data_list)*100, "%")
    pass


def test_nn():
    # 设置初始化参数，采用的是mnist数据集，为28*28的手写数字图像，隐含层100，输出层10代表0~9的数字，学习率初始设为0.2
    input_nodes = 28*28
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.2

    n = nn.NeuralNetwork(input_nodes, hidden_nodes,
                         output_nodes, learning_rate)

    # 第一步：开始训练
    print("start to train")
    train = True
    if train is True:
        # 训练方法1：用数据训练，采用较小的训练数据集
        training_data_list = dataset.get_data_list(
            "mnist_dataset/mnist_train.csv")
        count = 0  # 用于打印进度
        size = len(training_data_list)  # 用于打印进度
        for index, record in enumerate(training_data_list):
            label, inputs = dataset.get_scaled_data(record)
            targets = numpy.zeros(output_nodes)+0.01
            targets[label] = 0.99
            n.train(inputs, targets)
            # 打印进度
            print_process(index, size)

        # 将最终的权值矩阵保存
        numpy.savetxt("w_input_hidden.txt", n.w_input_hidden)
        numpy.savetxt("w_hidden_output.txt", n.w_hidden_output)
    else:
        print("load data done")
        # 训练方法2：直接导入训练的结果（适用于已经有训练结果，即权值矩阵）
        n.load("mnist_dataset/w_input_hidden.txt",
               "mnist_dataset/w_hidden_output.txt")

    # 第二步：开始测试训练后的神经网络
    print("start to test")
    test_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_test.csv")
    scorecard = []  # 记分牌，保存每个测试数据的测试结果
    right = 0  # 正确总数
    size = len(test_data_list)  # 用于打印进度
    for index, record in enumerate(test_data_list):
        label, inputs = dataset.get_scaled_data(record)
        result = n.query(inputs)
        # 对比神经网络预测结果和标签
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        # 打印进度
        print_process(index, size)

    # 保存记分牌
    numpy.savetxt("scorecard.txt", scorecard)
    # 打印正确率
    print("right rate=", right/len(test_data_list)*100, "%")


def main():
    # test_dataset()
    test_nn()
    # test_nn_pooling()


if __name__ == '__main__':
    main()
