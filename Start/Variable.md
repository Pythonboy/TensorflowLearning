# tensorflow.Variable的用法解析

## 两种定义图变量的方法

### tf.Variable

tf.Variable.**init**(initial_value, trainable=True, collections=None, validate_shape=True, name=None)

|    参数名称    |          参数类型          |                             含义                             |
| :------------: | :------------------------: | :----------------------------------------------------------: |
| initial_value  | 所有可以转换为Tensor的类型 |                         变量的初始值                         |
|   trainable    |            bool            | 如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer |
|  collections   |            list            |    指定该图变量的类型、默认为[GraphKeys.GLOBAL_VARIABLES]    |
| validate_shape |            bool            |             如果为False，则不进行类型和维度检查              |
|      name      |           string           |     变量的名称，如果没有指定则系统会自动分配一个唯一的值     |

*虽然有一堆参数，但只有第一个参数initial_value是必需的，用法如下（assign函数用于给图变量赋值）：*

```python
In [1]: import tensorflow as tf
In [2]: v = tf.Variable(3, name='v')
In [3]: v2 = v.assign(5)
In [4]: sess = tf.InteractiveSession()
In [5]: sess.run(v.initializer)
In [6]: sess.run(v)
Out[6]: 3
In [7]: sess.run(v2)
Out[7]: 5
```



### tf.get_variable
tf.get_variable跟tf.Variable都可以用来定义图变量，但是前者的必需参数（即第一个参数）并不是图变量的初始值，而是图变量的名称。

tf.Variable的用法要更丰富一点，当指定名称的图变量已经存在时表示获取它，当指定名称的图变量不存在时表示定义它，用法如下：

```python
In [1]: import tensorflow as tf
In [2]: init = tf.constant_initializer([5])
In [3]: x = tf.get_variable('x', shape=[1], initializer=init)
In [4]: sess = tf.InteractiveSession()
In [5]: sess.run(x.initializer)
In [6]: sess.run(x)
Out[6]: array([ 5.], dtype=float32)

```



## scope如何划分命名空间

一个深度学习模型的参数变量往往是成千上万的，不加上命名空间加以分组整理，将会成为可怕的灾难。TensorFlow的命名空间分为两种，tf.variable_scope和tf.name_scope。

*下面示范使用tf.variable_scope把图变量划分为4组：*

```python
for i in range(4):
    with tf.variable_scope('scope-{}'.format(i)):
        for j in range(25):
             v = tf.Variable(1, name=str(j))

```

#### tf.variable_scope

当使用tf.get_variable定义变量时，如果出现同名的情况将会引起报错

```python
In [1]: import tensorflow as tf
In [2]: with tf.variable_scope('scope'):
   ...:     v1 = tf.get_variable('var', [1])
   ...:     v2 = tf.get_variable('var', [1])
ValueError: Variable scope/var already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:

```

而对于tf.Variable来说，却可以定义“同名”变量

```python
In [1]: import tensorflow as tf
In [2]: with tf.variable_scope('scope'):
   ...:     v1 = tf.Variable(1, name='var')
   ...:     v2 = tf.Variable(2, name='var')
   ...:
In [3]: v1.name, v2.name
Out[3]: ('scope/var:0', 'scope/var_1:0')
```

但是把这些图变量的name属性打印出来，就可以发现它们的名称并不是一样的。

如果想使用tf.get_variable来定义另一个同名图变量，可以考虑加入新一层scope，比如：

```python
In [1]: import tensorflow as tf
In [2]: with tf.variable_scope('scope1'):
   ...:     v1 = tf.get_variable('var', shape=[1])
   ...:     with tf.variable_scope('scope2'):
   ...:         v2 = tf.get_variable('var', shape=[1])
   ...:
In [3]: v1.name, v2.name
Out[3]: ('scope1/var:0', 'scope1/scope2/var:0')

```

#### tf.name_scope

当tf.get_variable遇上tf.name_scope，它定义的变量的最终完整名称将不受这个tf.name_scope的影响，如下：

```python
In [1]: import tensorflow as tf
In [2]: with tf.variable_scope('v_scope'):
   ...:     with tf.name_scope('n_scope'):
   ...:         x = tf.Variable([1], name='x')
   ...:         y = tf.get_variable('x', shape=[1], dtype=tf.int32)
   ...:         z = x + y
   ...:
In [3]: x.name, y.name, z.name
Out[3]: ('v_scope/n_scope/x:0', 'v_scope/x:0', 'v_scope/n_scope/add:0')

```



## 图变量的复用

```python
In [1]: import tensorflow as tf
In [2]: with tf.variable_scope('scope'):
   ...:     v1 = tf.get_variable('var', [1])
   ...:     tf.get_variable_scope().reuse_variables()
   ...:     v2 = tf.get_variable('var', [1])
   ...:
In [3]: v1.name, v2.name
Out[3]: ('scope/var:0', 'scope/var:0')

```



```python
In [1]: import tensorflow as tf
In [2]: with tf.variable_scope('scope'):
   ...:     v1 = tf.get_variable('x', [1])
   ...:
In [3]: with tf.variable_scope('scope', reuse=True):
   ...:     v2 = tf.get_variable('x', [1])
   ...:
In [4]: v1.name, v2.name
Out[4]: ('scope/x:0', 'scope/x:0')

```



## Variable()与placeholder()的辨异

- **tf.Variable**：主要在于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）；
  1. <font color = "red"> 声明时必须提供初始值</font>
  2. 名称的真实含义，在于变量，也即在真实训练时，其值是会改变的，自然事先需要指定初始值； 

```python
weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
            stddev=1./math.sqrt(float(IMAGE_PIXELS)), name='weights')
)
biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')

```

- **tf.placeholder**：用于得到传递进来的真实的训练样本：
  1. <font color = "red">不必指定初始值，可以在运行时，通过Session.run的函数的feed_dict参数来指定</font>
  2. **仅仅作为一种占位符**

```python
images_placeholder = tf.placeholder(tf.float32, shape=[batch_size, IMAGE_PIXELS])
labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size])

```





**参考博文：**

>[TensorFlow图变量tf.Variable的用法解析——燃烧的快感](https://blog.csdn.net/gg_18826075157/article/details/78368924 )
>
>[TensorFlow 辨异 —— tf.placeholder 与 tf.Variable](https://blog.csdn.net/lanchunhui/article/details/61712830 )