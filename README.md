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