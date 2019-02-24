#### Requirements

+ Anaconda
+ Captcha (used to generate verification code)
+ Tensoflow

#### Install catpcha
`pip install captcha`


#### Generate verification code datasets

`python make_verification_code_datasets.py`

#### Transform datasets to tfrecord

`python to_tfrecord.py`

#### Train

`python train.py` 

Here, we used alexnet defined in nets/alexnet.py based on **multi-task learning**. We can also defined our network.

The training process can be seen:

```
Iter: 0  Loss: 2696.982  Accuracy: 0.16, 0.20, 0.28, 0.24  Learning_rate: 0.0010
Iter: 200  Loss: 2.302  Accuracy: 0.20, 0.08, 0.04, 0.12  Learning_rate: 0.0010
Iter: 400  Loss: 2.312  Accuracy: 0.16, 0.12, 0.04, 0.00  Learning_rate: 0.0010
Iter: 600  Loss: 2.320  Accuracy: 0.08, 0.16, 0.04, 0.04  Learning_rate: 0.0010
Iter: 800  Loss: 2.213  Accuracy: 0.28, 0.24, 0.08, 0.20  Learning_rate: 0.0010
Iter: 1000  Loss: 1.817  Accuracy: 0.28, 0.20, 0.32, 0.52  Learning_rate: 0.0010
Iter: 1200  Loss: 1.456  Accuracy: 0.48, 0.28, 0.56, 0.40  Learning_rate: 0.0010
Iter: 1400  Loss: 1.037  Accuracy: 0.56, 0.52, 0.60, 0.76  Learning_rate: 0.0010
Iter: 1600  Loss: 0.947  Accuracy: 0.84, 0.56, 0.36, 0.88  Learning_rate: 0.0010
Iter: 1800  Loss: 0.507  Accuracy: 0.84, 0.76, 0.76, 0.88  Learning_rate: 0.0010
Iter: 2000  Loss: 0.601  Accuracy: 0.80, 0.76, 0.68, 0.72  Learning_rate: 0.0003
Iter: 2200  Loss: 0.469  Accuracy: 0.84, 0.88, 0.76, 0.92  Learning_rate: 0.0003
Iter: 2400  Loss: 0.427  Accuracy: 0.92, 0.80, 0.80, 0.96  Learning_rate: 0.0003
Iter: 2600  Loss: 0.425  Accuracy: 0.88, 0.76, 0.88, 0.96  Learning_rate: 0.0003
Iter: 2800  Loss: 0.176  Accuracy: 0.96, 0.96, 1.00, 0.88  Learning_rate: 0.0003
Iter: 3000  Loss: 0.202  Accuracy: 0.92, 0.96, 0.84, 0.96  Learning_rate: 0.0003
Iter: 3200  Loss: 0.243  Accuracy: 0.96, 0.96, 0.92, 0.80  Learning_rate: 0.0003
Iter: 3400  Loss: 0.172  Accuracy: 0.92, 0.96, 0.92, 0.96  Learning_rate: 0.0003
Iter: 3600  Loss: 0.342  Accuracy: 0.92, 0.92, 0.92, 0.92  Learning_rate: 0.0003
Iter: 3800  Loss: 0.150  Accuracy: 0.96, 0.96, 0.92, 0.96  Learning_rate: 0.0003
Iter: 4000  Loss: 0.177  Accuracy: 0.92, 0.88, 0.96, 0.96  Learning_rate: 0.0001
Iter: 4200  Loss: 0.125  Accuracy: 1.00, 0.92, 0.96, 0.92  Learning_rate: 0.0001
Iter: 4400  Loss: 0.114  Accuracy: 1.00, 0.92, 1.00, 1.00  Learning_rate: 0.0001
Iter: 4600  Loss: 0.128  Accuracy: 1.00, 1.00, 0.84, 1.00  Learning_rate: 0.0001
Iter: 4800  Loss: 0.100  Accuracy: 1.00, 0.92, 0.96, 0.96  Learning_rate: 0.0001
Iter: 5000  Loss: 0.075  Accuracy: 1.00, 0.96, 1.00, 0.96  Learning_rate: 0.0001
Iter: 5200  Loss: 0.061  Accuracy: 0.96, 1.00, 0.96, 1.00  Learning_rate: 0.0001
Iter: 5400  Loss: 0.098  Accuracy: 1.00, 0.96, 0.88, 1.00  Learning_rate: 0.0001
Iter: 5600  Loss: 0.145  Accuracy: 1.00, 0.96, 0.88, 0.96  Learning_rate: 0.0001
Iter: 5800  Loss: 0.090  Accuracy: 1.00, 1.00, 0.96, 0.96  Learning_rate: 0.0001
Iter: 6000  Loss: 0.079  Accuracy: 1.00, 1.00, 0.92, 0.96  Learning_rate: 0.0000
```


#### Test

`python test.py`

Test results can be seen:

```
label: 0, 4, 3, 5
predict: 0, 4, 8, 5
label: 2, 8, 4, 5
predict: 2, 8, 4, 5
label: 7, 1, 0, 5
predict: 7, 1, 0, 5
label: 3, 1, 4, 1
predict: 3, 7, 4, 1
label: 3, 0, 6, 1
predict: 3, 4, 6, 1
label: 7, 7, 5, 9
predict: 7, 7, 5, 9
label: 3, 4, 6, 2
predict: 3, 4, 6, 2
label: 7, 1, 1, 2
predict: 7, 1, 2, 2
label: 0, 6, 2, 7
predict: 0, 6, 7, 7
label: 5, 8, 9, 6
predict: 5, 8, 9, 6
```

The image and prediction can be [link: verification code recognition](https://github.com/zhulf0804/Tensorflow-Learning/blob/master/tf_relearn/10-4%20%E9%AA%8C%E8%AF%81%E7%A0%81%E6%B5%8B%E8%AF%95.ipynb)


#### Reference

The code refers mainly to [https://www.bilibili.com/video/av20542427/?p=31](https://www.bilibili.com/video/av20542427/?p=31)