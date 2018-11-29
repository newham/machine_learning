from numpy import random, dot, array, transpose, loadtxt, argmax, shape
from scipy import special


class DeepNeuralNetwork:
    # 初始化参数
    def __init__(self, layers, learning_rate):
        print("start to init, layers = ", layers,
              ", learning_rate = ", learning_rate)

        # 初始化参数
        self.layers = layers
        self.learning_rate = learning_rate  # 学习率（误差变化比）
        self.layer_count = len(layers)

        # 初始化权重矩阵
        self.w_list = []
        self.layer_inputs = []

        # 设置权重.正态分布
        for index, layer in enumerate(self.layers):
            if index+1 < self.layer_count:
                next_layer = self.layers[index+1]
                # 隐含层1~n的权重矩阵
                w = random.normal(
                    0.0, pow(next_layer, -0.5), (next_layer, layer))
                self.w_list.append(w)

        # 激活函数
        self.activation_function = lambda x: special.expit(x)
        pass

    def layer_query_result(self, inputs_list):
        inputs = self.list_to_array(inputs_list)
        outputs = self.layer_query(inputs, 0, False)
        return argmax(outputs)

    def layer_query(self, inputs, w_index, is_train=False):
        if w_index == 0:
            self.layer_inputs.append(inputs)
        if w_index < len(self.w_list):
            outputs = self.get_layer_outputs(
                self.w_list[w_index], inputs)  # 隐含层的输入存入
            if is_train:
                self.layer_inputs.append(outputs)
            return self.layer_query(outputs, w_index+1, is_train)
        else:
            return inputs

    def get_layer_outputs(self, w, inputs):
        outputs = dot(w, inputs)  # 隐含层的输入存入
        return self.activation_function(outputs)

    def layer_query_errors(self, inputs_list, targets_list):
        targets = self.list_to_array(targets_list)
        inputs = self.list_to_array(inputs_list)
        outputs = self.layer_query(inputs, 0, True)
        errors = targets - outputs
        return errors

    def layer_bp(self, errors, w_index):
        if w_index >= 0:
            # 先计算上一层的误差
            next_errors = dot(self.w_list[w_index].T, errors)
            # 计算本层误差梯度矩阵
            layer_input = self.layer_inputs[w_index+1]
            previous_layer_input = self.layer_inputs[w_index]
            w_errors = self.learning_rate * dot((errors*layer_input*(1.0-layer_input)),
                                                transpose(previous_layer_input))
            # 更新权重
            self.w_list[w_index] += w_errors
            # 递归(反向传播)
            return self.layer_bp(next_errors, w_index - 1)
        else:
            # ！！！！将中间层输入清空
            self.layer_inputs = []
            return

    def layer_train(self, inputs_list, targets_list):
        # 1.得到误差
        errors = self.layer_query_errors(inputs_list, targets_list)
        # 2.反向传播
        self.layer_bp(errors, len(self.w_list)-1)

    # 转置矩阵
    def list_to_array(self, list):
        return array(list, ndmin=2).T

    # 导入权值矩阵，跳过训练
    def load(self, w_input_hidden_file, w_hidden_output_file):
        self.w_input_hidden = loadtxt(w_input_hidden_file)
        self.w_hidden_output = loadtxt(w_hidden_output_file)
