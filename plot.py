import matplotlib.pyplot as plt


def plot_acc(ax, labels, train_acc, test_acc):
    ax.plot(labels, train_acc, label='Train')
    ax.plot(labels, test_acc, label='Test')
    ax.set_xlabel('C')
    ax.set_ylabel('Accuracy')


if __name__ == '__main__':
    # SVM数据
    labels = ['1', '5', '10', '50', '100']
    train_acc = [0.8825, 0.9545, 0.9795, 1.00, 1.00]
    test_acc = [0.8418, 0.9051, 0.9205, 0.9375, 0.9375]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_acc(ax, labels, train_acc, test_acc)
    ax.legend()
    ax.set_title('Accuracy with Different C')
    fig.savefig('effect_of_c.jpg')
    # cnn depth
    labels = ['10', '16', '28']
    train_acc = [0.8825, 0.9545, 0.9795, 1.00, 1.00]
    test_acc = [0.8418, 0.9051, 0.9205, 0.9375, 0.9375]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    plot_acc(ax[0], labels, [99.70, 99.71, 99.39], [95.26, 95.04, 94.47])
    plot_acc(ax[1], labels, [99.65, 99.39, 99.74], [94.71, 95.20, 94.95])
    ax[0].set_title('MCNN')
    ax[1].set_title('ResMCNN')
    ax[0].legend()
    fig.savefig('effect_of_depth.jpg')
