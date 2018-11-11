from numpy import random, dot, array, transpose, loadtxt, argmax
from scipy import special


class NeuralNetwork:
    # 初始化参数
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        print("start to init,input_nodes =", input_nodes, ",hidden_nodes =",
              hidden_nodes, ",output_nodes =", output_nodes, ",learning_rate =", learning_rate)

        # 初始化参数
        self.input_nodes = input_nodes  # 输入层神经元数
        self.hidden_nodes = hidden_nodes  # 隐含层神经元数
        self.output_nodes = output_nodes  # 输出层神经元数（分类结果数）
        self.learning_rate = learning_rate  # 学习率（误差变化比）

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

        # 激活函数
        self.activation_function = lambda x: special.expit(x)

        pass

    # 训练
    def train(self, inputs_list, targets_list):
        inputs = self.list_to_array(inputs_list)
        targets = self.list_to_array(targets_list)

        # 隐含层的输入参数=w_hidden_output × inputs
        hidden_inputs = dot(self.w_input_hidden, inputs)
        # 计算从隐含层的激活函数值
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算隐含层到输出层的输入
        final_inputs = dot(self.w_hidden_output, hidden_outputs)
        # 计算输出层的激活函数
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = dot(self.w_hidden_output.T, output_errors)

        # 反向传播误差，更新权重矩阵
        # 更新：隐含层->输出层
        self.w_hidden_output += self.learning_rate * \
            dot((output_errors*final_outputs*(1.0-final_outputs)),
                transpose(hidden_outputs))
        # 更新：输入层->隐含层
        self.w_input_hidden += self.learning_rate * \
            dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),
                transpose(inputs))

        pass

    # 测试
    def query(self, inputs_list):
        inputs = self.list_to_array(inputs_list)
        # 隐含层的输入参数=w_hidden_output × inputs
        hidden_inputs = dot(self.w_input_hidden, inputs)
        # 计算从隐含层的激活函数值
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算隐含层到输出层的输入
        final_inputs = dot(self.w_hidden_output, hidden_outputs)
        # 计算输出层的激活函数
        final_outputs = self.activation_function(final_inputs)
        # 根据激活函数得到结果
        result = argmax(final_outputs)
        return result

    def query_img(self, img_array):
        # 隐含层的输入参数=w_hidden_output × inputs
        hidden_inputs = dot(self.w_input_hidden, img_array)
        # 计算从隐含层的激活函数值
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算隐含层到输出层的输入
        final_inputs = dot(self.w_hidden_output, hidden_outputs)
        # 计算输出层的激活函数
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    # 转置矩阵
    def list_to_array(self, list):
        return array(list, ndmin=2).T

    # 导入权值矩阵，跳过训练
    def load(self, w_input_hidden_file, w_hidden_output_file):
        self.w_input_hidden = loadtxt(w_input_hidden_file)
        self.w_hidden_output = loadtxt(w_hidden_output_file)
