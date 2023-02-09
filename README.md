# 图像识别作业

实现内容：

1. 实现基于序列最小优化的SVM算法，实现线性核、多项式核、RBF核等经典核函数，使用Numba进行加速计算。
2. 基于Pytorch框架的卷积神经网络，包括无残差结构（MCNN）和有残差结构（ResMCNN）的两个版本。
3. 使用上述算法完成MNIST、FashionMNIST、CIFAR10分类任务。

训练:

```shell
# 训练卷积神经网络
bash batch_run.sh
# 训练SVM
bash train_svm.sh
```
