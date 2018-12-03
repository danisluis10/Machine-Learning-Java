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

> - 1.5.2. Constants nodes and Sessions

```py
import tensorflow as tf

# Constant nodes
# When and how to use them
# Sessions

const_1 = tf.constant(value=[[1.0, 2.0]],
                      dtype=tf.float32,
                      shape=(1, 2),
                      name='const_1',
                      verify_shape=True)

print(const_1)
```
![alt text](https://github.com/danisluis10/Machine-Learning-Java/blob/master/ai_011.png)

> - 1.5.2. Variable nodes

```py
import tensorflow as tf

# Variable nodes
# When and how to use them
# Compare to constant nodes

var_1 = tf.Variable(initial_value=[1.0],
                    trainable=False,
                    collections=None,
                    validate_shape=True,
                    caching_device=None,
                    name='var_1',
                    variable_def=None,
                    dtype=tf.float32,
                    expected_shape=(1, None),
                    import_scope=None)
session = tf.Session()
print(var_1)
init = tf.global_variables_initializer()
session.run(init)
print(session.run(fetches=var_1))

var_2 = var_1.assign(value=[2.0])
print(session.run(fetches=var_1))
print(session.run(fetches=var_2))
```

> - 1.5.3. Placeholder nodes

```py
import tensorflow as tf

# Placeholder nodes
# When and how to use them
# Compare to constant nodes and variable nodes

placeholder_1 = tf.placeholder(dtype=tf.float32,
                               shape=(1, 4),
                               name='placeholder_1')

placeholder_2 = tf.placeholder(dtype=tf.float32,
                               shape=(2, 2),
                               name='placeholder_2')

session = tf.Session()
print(placeholder_1)
print(session.run(fetches=[placeholder_1, placeholder_2],
                  feed_dict={placeholder_1: [[1.0, 2.0, 3.0, 4.0]], placeholder_2: [[1.0, 2.0], [3.0, 4.0]]}))

```

> - 1.5.3. Operation nodes

```
import tensorflow as tf

# Operation nodes
# How to perform operations on existing nodes
# Build a mini computational graph

session = tf.Session()

const_1 = tf.constant(value=[1.0])
const_2 = tf.constant(value=[2.0])
placeholder_1 = tf.placeholder(dtype=tf.float32)
# results = const_1 + const_2
results = tf.add(x=placeholder_1, y=const_2, name='results')
print(session.run(fetches=results, feed_dict={placeholder_1: [2.0]}))
# y = Wx + b
```
### y = Wx + b

```py
import tensorflow as tf

# Operation nodes
# How to perform operations on existing nodes
# Build a mini computational graph

# y = Wx + b
session = tf.Session()

W = tf.constant(value=[2.0])
b = tf.constant(value=[1.0])

x = tf.placeholder(dtype=tf.float32)
# y = W * x + b
mult = tf.multiply(x=W, y=x)
y = tf.add(x = mult, y = b)
print(session.run(fetches=y,
                  feed_dict={
                      x: [2.0, 3.0, 4.0]
                  }))

```

> - 1.5.4. Loss, Optimizers, and Training

```py
import tensorflow as tf

# loss function: actual vs expected outputs
# actual: output from our model given an input
# expected: correct output given an input

# Optimizers: change values in model to alter loss (typically to

# Values altered during training
# Model assessed during testing

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [2.0, 3.0, 4.0, 5.0]
y_actual = [1.5, 2.5, 3.5, 4.5]
loss = tf.reduce_sum(input_tensor=tf.square(x=y_train-y_actual))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = optimizer.minimize(loss=loss)
```

> - 1.5.6. **BUILDING A LINEAR REGRESSION MODEL**

```py
import tensorflow as tf

# y = Wx + b

x_train = [1.0, 2.0, 3.0, 4.0]
y_train = [-1.0, -2.0, -3.0, -4.0]

W = tf.Variable(initial_value=[1.0], dtype=tf.float32)
b = tf.Variable(initial_value=[1.0], dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32)
y_input = tf.placeholder(dtype=tf.float32)

y_output = W * x + b

loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_input))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(loss=loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

print(session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train}))

for _ in range(1000):
    session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train})

print(session.run(fetches=[loss, W, b], feed_dict={x: x_train, y_input: y_train}))
print(session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))
```
#### 1.6. ML in Android Studio Projects 

> - 1.6.1. **Setup Prebuild Estimator Model**

```py
import tensorflow as tf
import numpy as np

# y = Wx + b
# inputs = [1.0, 2.0, 3.0, 4.0]
# outputs = [-1.0, -2.0, -3.0, -4.0]

x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([-1.0, -2.0, -3.0, -4.0])

x_eval = np.array([5.0, 10.0, 15.0, 20.0])
y_eval = np.array([-5.1, -10.1, -15.1, -20.1])

feature_column = tf.feature_column.numeric_column(key='x', shape=[1])
feature_columns = [feature_column]
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x' : x_train},
                                              y=y_train,
                                              batch_size=4,
                                              num_epochs=None,
                                              shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x' : x_train},
                                              y=y_train,
                                              batch_size=4,
                                              num_epochs=None,
                                              shuffle=False)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x' : x_eval},
                                              y=y_eval,
                                              batch_size=4,
                                              num_epochs=None,
                                              shuffle=False)

estimator.train(input_fn=input_fn, steps = 1000)
```


