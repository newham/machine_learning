import dataset
import numpy
from deep_neural_network import DeepNeuralNetwork
from base import print_process


def test_query():
    layers = [28 * 28, 100, 10]
    learning_rate = 0.2

    dnn = DeepNeuralNetwork(layers, learning_rate)

    training_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_train_100.csv")

    for record in training_data_list[0:4]:
        label, inputs = dataset.get_scaled_data(record)
        targets = dataset.get_targets_data(layers[-1], label)
        outputs = dnn.layer_query(inputs, targets)


def test_dnn():
    layers = [28 * 28, 200, 100, 50, 10]
    learning_rate = 0.2

    dnn = DeepNeuralNetwork(layers, learning_rate)
    # 开始测试训练
    print("start to train")
    # 训练方法1：用数据训练，采用较小的训练数据集
    training_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_train.csv")[:]  # 测试全部数据
    size = len(training_data_list)  # 用于打印进度
    for index, record in enumerate(training_data_list):
        label, inputs = dataset.get_scaled_data(record)
        targets = dataset.get_targets_data(layers[-1], label)
        dnn.layer_train(inputs, targets)
        # 打印进度
        print_process(index, size)

    # 开始测试
    print("start to test")
    test_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_test.csv")
    scorecard = []  # 记分牌，保存每个测试数据的测试结果
    right = 0  # 正确总数
    size = len(test_data_list)  # 用于打印进度
    for index, record in enumerate(test_data_list):
        label, inputs = dataset.get_scaled_data(record)
        result = dnn.layer_query_result(inputs)
        # 对比神经网络预测结果和标签
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        # 打印进度
        print_process(index, size)

    # 打印正确率
    print("right rate=", right / len(test_data_list) * 100, "%")
    pass


def recursion(x, i):
    if x < 1000:
        print(i)
        return recursion(x * 2, i + 1)
    else:
        print(x)


def main():
    test_dnn()
    # recursion(2, 1)
    pass


if __name__ == '__main__':
    main()
