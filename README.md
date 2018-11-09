# machine_learning
## learn machine-learning, write by Python

Python verion : Python3    

import      |
----------|
matplotlib|
numpy     |
scipy     |

start :
```
python3 start.py
```

if you use VSCODE as your IDE , you may meet this error: 
```
[pylint] E1101:Module 'scipy.special' has no 'expit' member
```
solve by changing your setting：
```
"python.linting.pylintArgs": ["--extension-pkg-whitelist=scipy"]
```

`mnist_dataset/dataset.url` include the download link of mnist data set:  
[mnist_train.csv](https://pjreddie.com/media/files/mnist_train.csv)  
[mnist_test.csv](https://pjreddie.com/media/files/mnist_test.csv)  

`mnist_dataset/` include a small data set of mnist : 100 for tranning , 10 for test