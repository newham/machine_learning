# 机器学习-人工神经网络
### 用Python实现的人工神经网络源代码（参考书《Python 神经网络编程》[英.Tariq Rashid]）
### 对书中的代码做了一些中文注释和优化，增加：操作数据集、导入导出模型、图片测试等功能  

代码基于: Python3    

****

**先导入库**

```
pip install -r requirements.txt
```

**运行** :
```
python3 test_nn.py
```

如果你用VSCode作为开发工具，可能会遇到这个error: 
```
[pylint] E1101:Module 'scipy.special' has no 'expit' member
```
请更改你的设置：
```
"python.linting.pylintArgs": ["--extension-pkg-whitelist=scipy"]
```

如果你运行代码出现cv2错误
请注释掉所有关于`import cv2`的代码
```
# import cv2
```

代码目录结构：  

目录|说明
--|--
`kdd/kdd.zip`|kdd数据集，直接解压kdd.zip到当前目录即可
`mnist_dataset/`|100组数据的训练集，10组数据的测试集
`mnist_dataset/dataset.url`|完整的mnist 数据集的下载地址：[mnist_train.csv](https://pjreddie.com/media/files/mnist_train.csv) ,[mnist_test.csv](https://pjreddie.com/media/files/mnist_test.csv)
`mnist_dataset/w_hidden_output.txt,w_input_hidden.txt`|导出后的权重矩阵，可直接用来测试
`img/mnist/`|从mnist数据集中导出的图片，有100多张，可以用来测试 . . .
`dataset.py`|操作数据集代码
`neural_network.py`|神经网络代码
`query.py`|图像测试代码
`test_nn.py`|训练并测试神经网络代码 > 可运行
`test_dnn.py`|训练并测试多隐含层神经网络代码 > 可运行
`test_kdd.py`|BP神经网络在kdd上的应用 > 可运行
  
**不用担心100组训练数据太少 ,运行代码你会发现只用100组测试数据就能达到【60%】的正确率！！**