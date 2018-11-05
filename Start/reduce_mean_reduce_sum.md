# tensorflow：tf.reduce_mean()和tf.reduce_sum()

## 求最大值

```python
tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
```



## 求平均值

````python
tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
````



## 求和

```python
reduce_sum(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None
)
```



- input_tensor:表示输入 
- axis:表示在那个维度进行sum操作。 
- keep_dims:表示是否保留原始数据的维度，False相当于执行完后原始数据就会少一个维度



**Example**

```python
import tensorflow as tf
import numpy as np
x = np.asarray([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
x_p = tf.placeholder(tf.int32,[2,2,3])
y =  tf.reduce_sum(x_p,0) #修改这里
with tf.Session() as sess:
    y = sess.run(y,feed_dict={x_p:x})
    print y
 
 
axis= 0：[[ 8 10 12] [14 16 18]] 
1+7 2+8 3+7 …….. 
axis=1: [[ 5 7 9] [17 19 21]] 
1+4 2+5 3 +6 …. 
axis=2: [[ 6 15] [24 33]] 
1+2+3 4+5+6…..


```

