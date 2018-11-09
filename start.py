import neural_network as nn

from matplotlib import pyplot
import numpy
import dataset


def print_process(count, size):  # 打印进度，每10%打印'>'
    count += 1
    if count/size >= 0.1:
        print('>', end="")
        count = 0
    return count


def test_nn():
    # 设置初始化参数，采用的是mnist数据集，为28*28的手写数字图像，隐含层100，输出层10代表0~9的数字，学习率初始设为0.2
    input_nodes = 28*28
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.2

    n = nn.NeuralNetwork(input_nodes, hidden_nodes,
                         output_nodes, learning_rate)

    # 开始训练，采用较小的训练数据集
    training_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_train_100.csv")

    print("start to train")
    count = 0
    size = len(training_data_list)
    for record in training_data_list:
        inputs = dataset.get_scaled_data(record)
        targets = numpy.zeros(output_nodes)+0.01
        all_values = dataset.get_all_values(record)
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        # 打印进度
        count = print_process(count, size)

    print("done")
    # 将最终的权值矩阵保存
    numpy.savetxt("w_input_hidden.txt", n.w_input_hidden)
    numpy.savetxt("w_hidden_output.txt", n.w_hidden_output)

    # 开始测试训练后的神经网络
    print("start to test")
    test_data_list = dataset.get_data_list(
        "mnist_dataset/mnist_test_10.csv")
    scorecard = []
    right = 0
    total = 0
    count = 0
    size = len(test_data_list)
    for record in test_data_list:
        outputs = n.query(dataset.get_scaled_data(record))
        result = numpy.argmax(outputs)
        label = dataset.get_img_label(record)
        # 对比神经网络预测结果和标签
        if label == result:
            scorecard.append(1)
            right += 1
        else:
            scorecard.append(0)
        total += 1

        # 打印进度
        count = print_process(count, size)

    print("done")
    numpy.savetxt("scorecard.txt", scorecard)
    print("right rate=", right/total*100, "%")


def main():
  # test_dataset()
    test_nn()


if __name__ == '__main__':
    main()
