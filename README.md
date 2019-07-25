# Machine learning - artificial neural network （机器学习-人工神经网络）
### This project refers to a book "Python neural network programming"[author:Tariq Rashid]. I build a BP artificial neural network with python. In addition, I add some Chinese annotation for the code , and add something new: **support for Deep BP neural network** , read & write dataset , Import and export model , **picture test**, etc.
### 用Python实现的人工神经网络源代码（参考书《Python 神经网络编程》[英.Tariq Rashid]）, 对书中的代码做了一些中文注释和优化，增加：支持深度BP、操作数据集、导入导出模型、图片测试等功能  

code is written in Python3（代码基于: Python3）    

****

**1.use PIP to install required libs （先导入库）**

```
pip install -r requirements.txt
```

**run the code (运行)** :
```
python3 test_nn.py
```

if you use VSCode as your development tools ， you will meet this error ： (如果你用VSCode作为开发工具，可能会遇到这个错误)
```
[pylint] E1101:Module 'scipy.special' has no 'expit' member
```
you can modify your config file in VSCode (请更改你的设置)：
```
"python.linting.pylintArgs": ["--extension-pkg-whitelist=scipy"]
```

if you meet the error about "cv2" , you can commented the code `import cv2` (如果你运行代码出现cv2错误
请注释掉所有关于`import cv2`的代码) 
```
# import cv2
```

code directory structure (代码目录结构)：  

menu(目录)|introduction(说明)
--|--
`kdd/kdd.zip`|kdd cup 99 dataset , unzip to current folder (kdd数据集，直接解压kdd.zip到当前目录即可)
`mnist_dataset/`|100 taining data, 10 test data (100组数据的训练集，10组数据的测试集)
`mnist_dataset/dataset.url`|full mnist dataset download url (完整的mnist 数据集的下载地址)：[mnist_train.csv](https://pjreddie.com/media/files/mnist_train.csv) ,[mnist_test.csv](https://pjreddie.com/media/files/mnist_test.csv)
`mnist_dataset/w_hidden_output.txt,w_input_hidden.txt`|exported model data , can be imported by program (导出后的权重矩阵，可直接用来测试)
`img/mnist/`|100 images, which is exported from mnist dataset , used for testing (从mnist数据集中导出的图片，有100多张，可以用来测试)<img src="img/mnist/0_5.png">,<img src="img/mnist/1_2.png">,<img src="img/mnist/1_0.png">
`dataset.py`|the code to read dataset (操作数据集代码)
`neural_network.py`|BP core code (神经网络代码)
`deep_neural_network.py`|Deep BP core code (深度神经网络代码)
`query.py`|test images code (图像测试代码)
`test_nn.py`|test BP in mnist code , can run (训练并测试神经网络代码 > 可运行)
`test_dnn.py`|test Deep BP in mnist code , can run (训练并测试多隐含层神经网络代码 > 可运行)
`test_kdd.py`|test BP used in kdd99 code BP , can run (神经网络在kdd上的应用 > 可运行)
  
**Don't worry about 100 sets of training data are too little, run the code and you'll find that you can get [60%] correct with only 100 sets of test data !! (不用担心100组训练数据太少 ,运行代码你会发现只用100组测试数据就能达到【60%】的正确率！！)**

Any question contact me with E-mail [newham.cn@gmail.com](newham.cn@gmail.com) (任何问题给我发邮件吧！)