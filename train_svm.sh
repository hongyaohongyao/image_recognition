python train_svm.py --name mnist_rbf -k=rbf -C=1
python train_svm.py --name mnist_linear -k=linear -C=1
python train_svm.py --ds FashionMNIST --name fmnist_rbf -k=rbf -C=1
python train_svm.py --ds FashionMNIST --name fmnist_linear -k=linear -C=1
python train_svm.py --ds CIFAR10 --name cifar10_rbf -k=rbf -C=1
python train_svm.py --ds CIFAR10 --name cifar10_linear -k=linear -C=1
python train_svm.py --name mnist_rbf_c5 -k=rbf -C=5
python train_svm.py --name mnist_rbf_c10 -k=rbf -C=10
python train_svm.py --name mnist_rbf_c50 -k=rbf -C=50
python train_svm.py --name mnist_rbf_c100 -k=rbf -C=100