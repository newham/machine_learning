from deep_neural_network import DeepNeuralNetwork
from matplotlib import pyplot
import numpy
import dataset
import random
from base import print_process


def get_kdd_data(record):
    array = record.rstrip('\n').split(',')
    label = numpy.argmax(array[122:127])
    data = numpy.asfarray(array[0:122])*0.99+0.01
    # data = numpy.asfarray(array[0:122])
    return label, data


def get_kdd_data_CICIDS(record):
    array = record.rstrip('\n').split(',')
    label = numpy.argmax(array[78])
    data = numpy.asfarray(array[0:78])*0.99+0.01
    return label, data


def get_kdd_data_UNSW(record):
    array = record.rstrip('\n').split(',')
    label_dict = {'Normal': 0,
                  'Analysis': 1,
                  'Backdoor': 2,
                  'DoS': 3,
                  'Exploits': 4,
                  'Fuzzers': 5,
                  'Generic': 6,
                  'Reconnaissance': 7,
                  'Shellcode': 8,
                  'Worms': 9
                  }
    label = label_dict[array[39]]
    data = numpy.asfarray(array[0:39])*0.99+0.01
    return label, data


def test_kdd_CICIDS():
    # 设置初始化参数，采用的是mnist数据集，为28*28的手写数字图像，隐含层100，输出层10代表0~9的数字，学习率初始设为0.2
    layers = [78, 40, 2]
    learning_rate = 0.5

    n = DeepNeuralNetwork(layers, learning_rate)
    # labels_count = [0, 0, 0, 0, 0]
    # 第一步：开始训练
    print("start to train")
    train = True
    if train is True:
        # 训练方法1：用数据训练，采用较小的训练数据集
        _, training_data_list = dataset.get_kdd_CICIDS_data(
            "kdd/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        training_data_list = random.sample(
            training_data_list, int(0.1*len(training_data_list)))
        # training_data_list = dataset.get_data_list(
        #     "kdd/KDDTest-21-normalization.txt.csv")
        size = len(training_data_list)  # 用于打印进度
        for index, record in enumerate(training_data_list):
            label, inputs = get_kdd_data_CICIDS(record)
            targets = numpy.zeros(layers[-1])+0.01
            targets[label] = 0.99
            # targets = numpy.zeros(output_nodes)
            # targets[label] = 1
            n.layer_train(inputs, targets)
            # 打印进度
            print_process(index, size)
            # 统计标签数
            # labels_count[label] += 1
    else:
        print("load data done")
        # 训练方法2：直接导入训练的结果（适用于已经有训练结果，即权值矩阵）
        n.load("w_input_hidden_kdd.txt",
               "w_hidden_output_kdd.txt")

    # 第二步：开始测试训练后的神经网络
    print("start to test")
    _, test_data_list = dataset.get_kdd_CICIDS_data(
        "kdd/Monday-WorkingHours.pcap_ISCX.csv")
    test_data_list = random.sample(
        test_data_list, int(0.1*len(test_data_list)))
    # test_data_list = dataset.get_data_list(
    #     "kdd/KDDTrain+_22Percent-normalization.txt.csv")

    scorecard = []  # 记分牌，保存每个测试数据的测试结果
    right = 0  # 正确总数
    count = 0  # 用于打印进度
    size = len(test_data_list)  # 用于打印进度
    for index, record in enumerate(test_data_list):
        label, inputs = get_kdd_data_CICIDS(record)
        result = n.layer_query_result(inputs)
        # 对比神经网络预测结果和标签
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        # 打印进度
        print_process(index, size)
    # print(labels_count)
    # 打印正确率
    print("right rate=", right/len(test_data_list)*100, "%")


def test_kdd_UNSW():
    # 设置初始化参数，采用的是mnist数据集，为28*28的手写数字图像，隐含层100，输出层10代表0~9的数字，学习率初始设为0.2
    layers = [39, 20, 10]
    learning_rate = 0.5

    n = DeepNeuralNetwork(layers, learning_rate)
    # labels_count = [0, 0, 0, 0, 0]
    # 第一步：开始训练
    print("start to train")
    train = True
    if train is True:
        # 训练方法1：用数据训练，采用较小的训练数据集
        training_data_list = dataset.get_kdd_UNSW_data(
            "kdd/UNSW_NB15_training-set.csv")
        training_data_list = random.sample(
            training_data_list, int(0.1*len(training_data_list)))
        # training_data_list = dataset.get_data_list(
        #     "kdd/KDDTest-21-normalization.txt.csv")
        size = len(training_data_list)  # 用于打印进度
        for index, record in enumerate(training_data_list):
            label, inputs = get_kdd_data_UNSW(record)
            targets = numpy.zeros(layers[-1])+0.01
            targets[label] = 0.99
            # targets = numpy.zeros(output_nodes)
            # targets[label] = 1
            n.layer_train(inputs, targets)
            # 打印进度
            print_process(index, size)
            # 统计标签数
            # labels_count[label] += 1
    else:
        print("load data done")
        # 训练方法2：直接导入训练的结果（适用于已经有训练结果，即权值矩阵）
        n.load("w_input_hidden_kdd.txt",
               "w_hidden_output_kdd.txt")

    # 第二步：开始测试训练后的神经网络
    print("start to test")
    test_data_list = dataset.get_kdd_UNSW_data(
        "kdd/UNSW_NB15_training-set.csv")
    test_data_list = random.sample(
        test_data_list, int(0.1*len(test_data_list)))
    # test_data_list = dataset.get_data_list(
    #     "kdd/KDDTrain+_22Percent-normalization.txt.csv")

    scorecard = []  # 记分牌，保存每个测试数据的测试结果
    right = 0  # 正确总数
    count = 0  # 用于打印进度
    size = len(test_data_list)  # 用于打印进度
    for index, record in enumerate(test_data_list):
        label, inputs = get_kdd_data_UNSW(record)
        result = n.layer_query_result(inputs)
        # 对比神经网络预测结果和标签
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        # 打印进度
        print_process(index, size)
    # print(labels_count)
    # 打印正确率
    print("right rate=", right/len(test_data_list)*100, "%")


def test_kdd_NSL():
    # 设置初始化参数，采用的是mnist数据集，为28*28的手写数字图像，隐含层100，输出层10代表0~9的数字，学习率初始设为0.2
    layers = [122, 100, 10, 5]
    learning_rate = 0.5

    n = DeepNeuralNetwork(layers, learning_rate)
    labels_count = [0, 0, 0, 0, 0]
    # 第一步：开始训练
    print("start to train")
    train = True
    if train is True:
        # 训练方法1：用数据训练，采用较小的训练数据集
        training_data_list = dataset.get_data_list(
            "kdd/KDDTrain+_22Percent-normalization.txt.csv")
        # training_data_list = dataset.get_data_list(
        #     "kdd/KDDTest-21-normalization.txt.csv")
        size = len(training_data_list)  # 用于打印进度
        for index, record in enumerate(training_data_list):
            label, inputs = get_kdd_data(record)
            targets = numpy.zeros(layers[-1])+0.01
            targets[label] = 0.99
            # targets = numpy.zeros(output_nodes)
            # targets[label] = 1
            n.layer_train(inputs, targets)
            # 打印进度
            print_process(index, size)
            # 统计标签数
            labels_count[label] += 1

        # 将最终的权值矩阵保存
        print(labels_count)
    else:
        print("load data done")
        # 训练方法2：直接导入训练的结果（适用于已经有训练结果，即权值矩阵）
        n.load("w_input_hidden_kdd.txt",
               "w_hidden_output_kdd.txt")

    # 第二步：开始测试训练后的神经网络
    labels_count = [0, 0, 0, 0, 0]
    print("start to test")
    test_data_list = dataset.get_data_list(
        "kdd/KDDTest-21-normalization.txt.csv")
    # test_data_list = dataset.get_data_list(
    #     "kdd/KDDTrain+_22Percent-normalization.txt.csv")

    scorecard = []  # 记分牌，保存每个测试数据的测试结果
    right = 0  # 正确总数
    count = 0  # 用于打印进度
    size = len(test_data_list)  # 用于打印进度
    for index, record in enumerate(test_data_list):
        label, inputs = get_kdd_data(record)
        result = n.layer_query_result(inputs)
        labels_count[label] += 1
        # 对比神经网络预测结果和标签
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        # 打印进度
        print_process(index, size)
    print(labels_count)
    # 打印正确率
    print("right rate=", right/len(test_data_list)*100, "%")

def test_kdd():
    # 设置初始化参数，采用的是mnist数据集，为28*28的手写数字图像，隐含层100，输出层10代表0~9的数字，学习率初始设为0.2
    layers = [39, 20, 10]
    learning_rate = 0.5

    n = DeepNeuralNetwork(layers, learning_rate)
    # labels_count = [0, 0, 0, 0, 0]
    # 第一步：开始训练
    print("start to train")
    train = True
    if train is True:
        # 训练方法1：用数据训练，采用较小的训练数据集
        training_data_list = dataset.get_kdd_UNSW_data(
            "kdd/UNSW_NB15_training-set.csv")
        training_data_list = random.sample(
            training_data_list, int(0.1 * len(training_data_list)))
        # training_data_list = dataset.get_data_list(
        #     "kdd/KDDTest-21-normalization.txt.csv")
        size = len(training_data_list)  # 用于打印进度
        for index, record in enumerate(training_data_list):
            label, inputs = get_kdd_data_UNSW(record)
            targets = numpy.zeros(layers[-1]) + 0.01
            targets[label] = 0.99
            # targets = numpy.zeros(output_nodes)
            # targets[label] = 1
            n.layer_train(inputs, targets)
            # 打印进度
            print_process(index, size)
            # 统计标签数
            # labels_count[label] += 1
    else:
        print("load data done")
        # 训练方法2：直接导入训练的结果（适用于已经有训练结果，即权值矩阵）
        n.load("w_input_hidden_kdd.txt",
               "w_hidden_output_kdd.txt")

    # 第二步：开始测试训练后的神经网络
    print("start to test")
    test_data_list = dataset.get_kdd_UNSW_data(
        "kdd/UNSW_NB15_training-set.csv")
    test_data_list = random.sample(
        test_data_list, int(0.1 * len(test_data_list)))
    # test_data_list = dataset.get_data_list(
    #     "kdd/KDDTrain+_22Percent-normalization.txt.csv")

    scorecard = []  # 记分牌，保存每个测试数据的测试结果
    right = 0  # 正确总数
    count = 0  # 用于打印进度
    size = len(test_data_list)  # 用于打印进度
    for index, record in enumerate(test_data_list):
        label, inputs = get_kdd_data_UNSW(record)
        result = n.layer_query_result(inputs)
        # 对比神经网络预测结果和标签
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        # 打印进度
        print_process(index, size)
    # print(labels_count)
    # 打印正确率
    print("right rate=", right / len(test_data_list) * 100, "%")


def test_data():
    training_data_list = dataset.get_data_list(
        "kdd/KDDTrain+_22Percent-normalization.txt.csv")
    print(training_data_list[0])
    label, data = get_kdd_data(training_data_list[0])
    print(data)
    print(len(data), label)


def main():
    # test_kdd_UNSW()
    # test_kdd_CICIDS()
    test_kdd()
    # test_data()


if __name__ == '__main__':
    main()
