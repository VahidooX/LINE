from utils import svm_classify, batchgen_train, LINE_loss, load_data
import numpy as np
from model import create_model
import random


if __name__ == "__main__":

    label_file = 'dblp/labels.txt'
    edge_file = 'dblp/adjedges.txt'
    epoch_num = 100
    factors = 128
    batch_size = 1000
    negative_sampling = "UNIFORM" # UNIFORM or NON-UNIFORM
    negativeRatio = 5
    split_ratios = [0.6, 0.2, 0.2]
    svm_C = 0.1

    np.random.seed(2017)
    random.seed(2017)

    adj_list, labels_dict = load_data(label_file, edge_file)
    epoch_train_size = (((int(len(adj_list)/batch_size))*(1 + negativeRatio)*batch_size) + (1 + negativeRatio)*(len(adj_list)%batch_size))
    numNodes = np.max(adj_list.ravel()) + 1
    data_gen = batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling)

    model, embed_generator = create_model(numNodes, factors)
    model.summary()

    model.compile(optimizer='rmsprop', loss={'left_right_dot': LINE_loss})

    model.fit_generator(data_gen, steps_per_epoch=epoch_train_size, epochs=epoch_num, verbose=1)

    new_X = []
    new_label = []

    keys = list(labels_dict.keys())
    np.random.shuffle(keys)

    for k in keys:
        v = labels_dict[k]
        x = embed_generator.predict_on_batch([np.asarray([k]), np.asarray([k])])
        new_X.append(x[0][0] + x[1][0])
        new_label.append(labels_dict[k])

    new_X = np.asarray(new_X, dtype=np.float32)
    new_label = np.asarray(new_label, dtype=np.int32)

    [train_acc, valid_acc, test_acc] = svm_classify(new_X, new_label, split_ratios, svm_C)

    print("Train Acc:", train_acc, " Validation Acc:", valid_acc, " Test Acc:", test_acc)
