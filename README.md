# MACHINE LEARNING JAVA [![Build Status](https://travis-ci.org/nomensa/jquery.hide-show.svg)](https://travis-ci.org/nomensa/jquery.hide-show.svg?branch=master)
#### 1.1. Links
- https://www.udemy.com/courses/search/?src=ukw&q=Machine+learning+Java
#### 1.2. Machine Learning Concept
> - 1.2.1 Download **PyCharm**

![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_001..png)

> - 1.2.2 Install pyCharm successfully

![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_002.png)

> - 1.2.3 Step by step to install Python3.6
 >> - Open terminal via Ctrl+Alt+T or searching for “Terminal” from app launcher. When it opens, run command to add the PPA:
```
sudo add-apt-repository ppa:jonathonf/python-3.6
```
>> - Then check updates and install Python 3.6 via commands:
```
sudo apt-get update
sudo apt-get install python3.6
```
>> - To make python3 use the new installed python 3.6 instead of the default 3.5 release, run following 2 commands:
```
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
```
>> - Finally switch between the two python versions for python3 via command:
```
sudo update-alternatives --config python3
```
>> - After selecting version 3.6:
```
python3 -V
```

![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_003.png)

#### 1.2. Explore Example (file Code.zip)

![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_004.png)

#### 1.3. Explore PyCharm and run program **HelloWorld.py**
> - 1.3.1. Error **Please select a valid python interpreter**

![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_005.png)
![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_006.png)

#### 1.4. Python Language Basics
> - 1.4.1. Variables in Python language.

![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_007.png)

> - 1.4.2. Variable Operations and Conversions

```py
# assignment, arithmetic, and conditional operators

isVisaCredit = False
isVisaCredit = not isVisaCredit
print(isVisaCredit)

# + - * / % // += -= *= /=
bank_balance = 1000 + payCheck
print(bank_balance)
print(5%2)
print(type(5%2))
print(5/2)
print(type(5/2))
print(5//2)
print(type(5//2))

numbers = [0, 1, 2, 3]
temp = 0
for x in numbers:
    temp += x
    print(temp)

# > >= < <= == !=
print(5 != 2)
print("a" > "b")

year = 2018
print(type(2018))
year = '2018'
print(type(year))
year = "2018"
print(type(year))

# Convert from Float to Int
a = 9.81
print(int(a))
print(float(2))

# Convert number to String
y = 1255645.255
print(str(y))
print(type(str(y)))
```

> - 1.4.3. Collection Types

```py
# Collections: tuples, lists, dictionaries
# Operations

full_name = ('Nimish', "Narang")
print(full_name)
full_name = ('John', 'Smith')
print(full_name)

inventory_item = ('apple', 5)
print(inventory_item)

roster = ['Nimish', 'John', 'Kate', 'Sarah', 'Kevin']
print(roster)

inventory = {'apple': 5, 'knife': 1, 'shoes': 2}
print(inventory)

print(full_name[0])
print(roster[1:3])
```

> - 1.4.3. Operations on Collections

#### 1.5. TensorFlow 
TensorFlow is a Python-friendly open source library for numerical computation that makes **machine learning** and **deep learning** faster and easier
![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_008.jpg)


> - 1.5.1. Setup project and Project Outline

![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_009.png)

> - 1.5.1. How to Import TensorFlow to PyCharm, Install package **tensorflow**

![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_010.png)




