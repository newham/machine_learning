from numpy import random, dot, array, transpose
from scipy import special


class NeuralNetwork:
    # init
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        print("start to init,input_nodes =", input_nodes, ",hidden_nodes =",
              hidden_nodes, ",output_nodes =", output_nodes, ",learning_rate =", learning_rate)

        # 初始化参数
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 设置权重矩阵
        # 方法1.随机生成权重矩阵
        # self.w_input_hidden = random.rand(
        #     self.hidden_nodes, self.input_nodes)-0.5
        # self.w_hidden_output = random.rand(
        #     self.output_nodes, self.hidden_nodes)-0.5

        # 方法2.正态分布
        self.w_input_hidden = random.normal(
            0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.w_hidden_output = random.normal(
            0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # print(self.w_input_hidden)
        # print(self.w_hidden_output)

        # 抑制函数
        self.activation_function = lambda x: special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = self.list_to_array(inputs_list)
        targets = self.list_to_array(targets_list)

        # 隐含层的输入参数=w_hidden_output × inputs
        hidden_inputs = dot(self.w_input_hidden, inputs)
        # 计算从隐含层的抑制函数值
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算隐含层到输出层的输入
        final_inputs = dot(self.w_hidden_output, hidden_outputs)
        # 计算输出层的抑制函数
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = dot(self.w_hidden_output.T, output_errors)

        # 更新权重
        self.w_hidden_output += self.learning_rate * \
            dot((output_errors*final_outputs*(1.0-final_outputs)),
                transpose(hidden_outputs))

        self.w_input_hidden += self.learning_rate * \
            dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
                transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = self.list_to_array(inputs_list)
        # 隐含层的输入参数=w_hidden_output × inputs
        hidden_inputs = dot(self.w_input_hidden, inputs)
        # 计算从隐含层的抑制函数值
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算隐含层到输出层的输入
        final_inputs = dot(self.w_hidden_output, hidden_outputs)
        # 计算输出层的抑制函数
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def list_to_array(self, list):
        return array(list, ndmin=2).T
