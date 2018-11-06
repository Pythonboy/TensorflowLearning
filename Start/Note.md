# 第一章

## Tensorflow基本概念

- 使用图（graphs）来表示计算任务
- 在使用称之为会话（Session）的上下文文（context）中执行图
- 使用tensor表示数据
- 使用变量（Variable）来维护状态
- 使用feed 和 fetch 可以为任意的操作赋值或者从其中获取数据



## Variable & palceholder & constant

- Variable : 变量，通常作为参数
- placeholder： 占位符，通常用于传入数据集
- constant ： 常量

[TensorFlow中的变量（Variables）](https://blog.csdn.net/chinagreenwall/article/details/80697400)

[Tensorflow学习笔记——节点（constant，placeholder，Variable）](https://www.cnblogs.com/Vulpers/p/7809276.html)



## Session

- Session : 会话，管理Tensorflow程序运行时的所有资源

[TensorFlow基础知识5-会话（session)](https://blog.csdn.net/hongxue8888/article/details/76762108)



## 降维函数 reduce

[Tensorflow中的降维函数总结：tf.reduce_*](https://blog.csdn.net/u013093426/article/details/81430374)



## Optimizers(优化器)

[tensorflow optimizers](https://www.jianshu.com/p/e6e8aa3169ca)

[tensorflow-梯度下降，有这一篇就足够了](https://segmentfault.com/a/1190000011994447)

[Tensorflow学习（四）：优化器Optimizer](https://blog.csdn.net/xierhacker/article/details/53174558)

[tensorflow优化器optimizer](https://blog.csdn.net/wuguangbin1230/article/details/71160777)



## Softmax

[Softmax回归](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)

[Tensorflow基础系列（softmax回归）](https://blog.csdn.net/ding_xiaofei/article/details/80118374)



## 损失函数

> [Tensorflow 损失函数（loss Function](https://blog.csdn.net/hongxue8888/article/details/77159772)

*重点*

```python
# 分类问题
# 交叉熵
cross_entropy = -tf.reduce_mean(y_* tf.log(tf.clip_by_value(y, 1e-10, 1.0))) 
'''
tf.clip_by_value函数可以将一个张量中的数值限制在一个范围内，这样就避免了一些运算错误（比如log0是无效的）。

y_：正确结果 
y ：预测结果
'''

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y,y_)
# TensorFlow对交叉熵和softmax回归进行了统一封装，我们可以直接使用如下代码实现使用softmax回归后的交叉熵损失函数：

# 回归问题
# 均方误差（MSE）
mse = tf.reduce_sum(tf.square(y_ -  y))
```



> [Tensorflow损失函数（loss function) 及自定义损失函数（一）](https://blog.csdn.net/limiyudianzi/article/details/80693695)
>
> [Tensorflow损失函数（loss function) 及自定义损失函数（二）](https://blog.csdn.net/limiyudianzi/article/details/80694614)

*重点*

```python
# Tensorflow内置的四个损失函数
Tensor=tf.nn.softmax_cross_entropy_with_logits(logits= Network.out, labels= Labels_onehot)

Tensor=tf.nn.sparse_softmax_cross_entropy_with_logits (logits=Network.out, labels= Labels)

Tensor=tf.nn. sigmoid_cross_entropy_with_logits (logits= Network.out, labels= Labels_onehot)

Tensor=tf.nn.weighted_cross_entropy_with_logits (logits=Network.out, labels=Labels_onehot, pos_weight=decimal_number)
```



## 前向传播和反向传播

> [使用前向传播和反向传播的神经网络代码](https://blog.csdn.net/gaoyueace/article/details/79017532)







# 第二章

 # MNIST数据集



> [Tensorflow笔记 —— 关于MNIST数据的一个简单的例子](https://blog.csdn.net/sangfengcn/article/details/78665799)

*重点*

```python
# 预测
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵 （损失函数）
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 测试
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

```



## 二次代价函数（quadratic cost)

$$C = \frac{(y - a)^2}{2}$$

- C ： 代价函数
- x : 样本
- y : 真实值
- a : $a = \sigma (z)$ ,$ z = \sum W_j * X_j + b; 表示输出值
- n : 样本总数
- $\sigma ()$ : 激活函数



