from sklearn.datasets import make_blobs
from numpy import where
from matplotlib import pyplot

from sklearn.datasets import make_blobs
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from numpy import mean
from numpy import std
import numpy as np
import tensorflow as tf

from sklearn.datasets import make_classification


from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
dist_ori, dist_labels_ori= list(), list()
score_ori = list()

# generate samples for blobs problem with a given random seed
# generate 2d classification dataset
#X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=1)


nfeature = 4
center_set = 3
num_class = 3
n_train_jh = 1200
n_train_jh_trans = 500
random_seed = 7
#난이도 중
def samples_for_basic(seed):
    # generate samples
    #X, yy = make_blobs(n_samples=1000, centers= center_set, n_features=nfeature, cluster_std=[2, 2.2, 2], center_box = (-7, 7), random_state=seed)
    X, yy = make_blobs(n_samples=2000, centers=center_set, n_features=nfeature, cluster_std= 2, center_box=(-7, 7), random_state=seed)

    #X, yy = make_classification (n_samples=1000,  n_features=nfeature , n_redundant = 0 , n_classes =2 ,  flip_y = 0.2 ,  random_state = seed)

    # one hot encode output variable
    y = to_categorical(yy)
    # split into train and test
    n_train = n_train_jh
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    return trainX, trainy, testX, testy, X, yy

#난이도 상
def samples_for_non_easy(seed):

    # generate samples
    #print("random seed" + str(seed))
    #X, yy = make_blobs(n_samples=1000, centers=center_set, n_features=nfeature , cluster_std=[2.8, 3.2, 2.8], center_box = (-5, 5), random_state=seed)
    X, yy = make_blobs(n_samples=2000, centers=center_set, n_features=nfeature , cluster_std=3.2, center_box = (-5, 5), random_state=seed)

    #X, yy = make_classification (n_samples=1000,  n_features=nfeature , n_redundant = 0 , n_classes =2 ,  flip_y = 0.3 ,  random_state = seed)

    y = to_categorical(yy)
    # split into train and test
    n_train = n_train_jh
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    return trainX, trainy, testX, testy, X, yy

#난이도 중과 유사
def samples_for_easy(seed):

    # generate samples
    X, yy= make_blobs(n_samples=2000, centers=center_set, n_features=nfeature , cluster_std=2.5, center_box = (-8, 8) , random_state=seed)

    #X, yy = make_classification(n_samples=1000, n_features=nfeature, n_redundant=0, n_classes=2, flip_y=0.2, random_state=seed)
    y = to_categorical(yy)
    # split into train and test
    n_train = n_train_jh
    
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    
    return trainX, trainy, testX, testy , X, yy


# 난이도 중과 유사
def sample_for_test(seed, cluster_std_a = 2.5, cluster_std_b =2.5, data_range= 10):

    # generate samples
    #print("cluster a "  +  str(cluster_std_a) + " cluster b " + str(cluster_std_b)  + " data range " + str(data_range))
    cluster_str_mean = (cluster_std_a + cluster_std_b)/2

    #X, yy = make_blobs(n_samples=1000, centers=center_set, n_features=nfeature , cluster_std= [cluster_std_a, cluster_std_b, cluster_std_a], center_box = (0 -data_range,  data_range), random_state=seed)
    X, yy = make_blobs(n_samples=1000, centers=center_set, n_features=nfeature,
                       cluster_std=cluster_str_mean,
                       center_box=(0 - data_range, data_range), random_state=seed)
    #X, yy = make_classification(n_samples=1000, n_features=nfeature, n_redundant=0, n_classes=2, flip_y=0.15, random_state=seed)

    y = to_categorical(yy)
    # split into train and test
    n_train = n_train_jh_trans

    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]

    return trainX, trainy, testX, testy, X, yy


# create a scatter plot of points colored by class value
def plot_samples(X, y, titleName, classes=num_class, ):
    # plot points for each class
    for i in range(classes):
        # select indices of points with each class label
        samples_ix = where(y == i)

        # plot points for this class with a given color
        pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1])

    pyplot.title(titleName)
    pyplot.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)


def testSetPlot ():
    # generate multiple problems
    n_problems = 5
    lastidx  = 1;
    for i in range(1, n_problems):
        # specify subplot
        print(i)

        pyplot.subplot(3, 3, i)

        #pyplot.subplot(210 + i)

        # generate samples

        subTitleName = "Title"

        if i % 4 == 2:
            i = i  - 1
            _, _, _, _, X, y= samples_for_non_easy(random_seed)
            #ax[0, 1].set_title("similar set")
            subTitleName = "Difficult Data set"

        elif i % 4 == 3:

            _, _, _, _, X, y= samples_for_easy(random_seed)
            #ax[1, 1].set_title("big difference")
            subTitleName = "Easy Dataset"


        elif i % 4 ==1:
            #ax[0, 0].set_title("basic set")

            _, _, _, _, X, y = samples_for_basic(random_seed)
            subTitleName = "Basic set"

        elif i % 4 == 0:
            # ax[0, 0].set_title("basic set")
            i = i -2
            continue
            #_, _, _, _, X, y = samples_for_seed4(i)
            #subTitleName = "Totally different  set"
        else :
            print("what")

        # scatter plot of samples
        plot_samples(X, y, subTitleName)
        lastidx = lastidx +1

    # plot figure
    #pyplot.show()

    data_type = 6

    std_var_a = 2.2
    std_var_b = 2.5
    increment_std = 0.4

    increas_r = 0.4
    range_val = 9

    for i in range(lastidx, data_type + lastidx):

        print(i)
        pyplot.subplot(3, 3, i)
        _, _, _, _, X, y = sample_for_test(random_seed, std_var_a, std_var_b, range_val)

        std_var_a = std_var_a + increment_std
        std_var_b = std_var_b + increment_std
        range_val = range_val - increas_r

        subTitleName = "test Dataset " + str(i - lastidx + 1)
        plot_samples(X, y, subTitleName)


        #pyplot.rcParams["figure.figsize"] = (20, 10)


    pyplot.show(block = False)
    pyplot.pause(1)
    pyplot.close()

    #pyplot.draw()
    #pyplot.waitforbuttonpress(0)
    #pyplot.close()


# define and fit model on a training dataset
def fit_model(trainX, trainy, testX, testy):
    # define model
    model = Sequential()
    #model.add(Dense(5, input_dim=nfeature, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(60, input_dim=nfeature, activation='relu', kernel_initializer='he_normal'))
    #model.add(Conv2D(10, (3, 3), padding='valid', input_shape=(nfeature, 1), activation='relu'))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_class, activation='softmax'))
    print(model.summary())


    # compile model print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
    return model, history



# repeated evaluation of a standalone model
def eval_standalone_model(trainX, trainy, testX, testy, n_repeats):
    scores = list()
    for _ in range(n_repeats):
        # define and fit a new model on the train dataset
        model =  fit_model_transfer(trainX, trainy)
        # evaluate model on test dataset
        _, test_acc = model.evaluate(testX, testy, verbose=0)
        scores.append(test_acc)
    return scores


# repeated evaluation of a model with transfer learning
def eval_transfer_model(trainX, trainy, testX, testy, n_fixed, n_repeats):
    scores = list()
    for _ in range(n_repeats):
        # load model
        model = load_model('model.h5')
        # mark layer weights as fixed or not trainable
        for i in range(n_fixed):
            model.layers[i].trainable = False

        # re-compile model
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        # fit model on train dataset
        model.fit(trainX, trainy, epochs=100, verbose=0)
        # evaluate model on test dataset
        _, test_acc = model.evaluate(testX, testy, verbose=0)
        scores.append(test_acc)
    return scores

def eval_ori_model(testX, testy, dataType):
    score_ori = list()
    model = load_model('model.h5')
    _, test_acc = model.evaluate(testX, testy, verbose=0)

    score_ori.append(test_acc)

    dist_ori.append(test_acc)
    dist_labels_ori.append(dataType)

    print (test_acc)

sgd = SGD(lr=1e-3 * 0.8, decay=1e-7,  momentum=0.90, nesterov=True)

def eval_transfer_model_jh_use_w_param(trainX, trainy, testX, testy, n_fixed, n_repeats):
    scores = list()

    # load model
    model = load_model('model.h5')

    weight = model.layers[2].get_weights()
    weight = np.array(weight)
    #print(weight.shape)

    # mark layer weights as fixed or not trainable
    model.layers[0].trainable = False
    model.layers[1].trainable = False
    model.layers[2].trainable = True

    # re-compile model
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())

    # fit model on train dataset
    #model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
    model.fit(trainX, trainy, epochs=50, verbose=0, batch_size=16)

    # evaluate model on test dataset
    _, test_acc = model.evaluate(testX, testy, verbose=0)

    scores.append(test_acc)
    return scores

def eval_transfer_model_jh_set_zero(trainX, trainy, testX, testy, n_fixed, n_repeats):
    scores = list()

    # load model
    model = load_model('model.h5')
    # mark layer weights as fixed or not trainable
    model.layers[0].trainable = False
    model.layers[1].trainable = False
    model.layers[2].trainable = True

    #mean = np.array([0])
    #var = mean ** 2
    #var = 0
    #mean = np.zeros(shape=(2, ), dtype=np.float32)  # 10 -zero vector 생성
    #mean = mean.tolist()
    initializers = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None)


    #idx = 2
    max_layer_num = 3
    for idx in range (1, max_layer_num):

        weight_initializer = model.layers[idx].kernel_initializer
        bias_initializer = model.layers[idx].bias_initializer
        old_weights, old_biases = model.layers[idx].get_weights()

        weight = initializers(shape=old_weights.shape)
        #weight = weight_initializer(shape=old_weights.shape)
        bias  = bias_initializer(shape=old_biases.shape)

        #weight = np.zeros(shape=old_weights.shape, dtype=np.float32)
        bias  = np.zeros(shape=old_biases.shape, dtype=np.float32)

        model.layers[idx].set_weights([
           weight,
           bias])

        #model.layers[idx].set_weights([
        #    weight_initializer(shape=old_weights.shape),
        #    bias_initializer(shape=old_biases.shape)])

    # re-compile model
    #model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())

    # fit model on train dataset
    model.fit(trainX, trainy, epochs=50, verbose=0 , batch_size=16)

    # evaluate model on test dataset
    _, test_acc = model.evaluate(testX, testy, verbose=0)

    scores.append(test_acc)
    return scores




# summarize the performance of the fit model
def summarize_model(model, history, trainX, trainy, testX, testy):
    # evaluate the model
    _, train_acc = model.evaluate(trainX, trainy, verbose=0)
    _, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()

    pyplot.show(block = False)
    pyplot.pause(1)
    pyplot.close()
    #end function




if __name__ == "__main__":

    #test / training set visualization

    testSetPlot()

    # prepare data
    trainX, trainy, testX, testy, _ , _ = samples_for_basic(random_seed)

    # fit model on train dataset
    model, history = fit_model(trainX, trainy, testX, testy)
    cnt = 0

    #for layer in model.layers:
        #print(layer.output_shape)
        #print(model.layers[cnt].get_shape)
        #cnt = cnt+1


    # evaluate model behavior
    summarize_model(model, history, trainX, trainy, testX, testy)
    # save model to file

    #model.save('model.h5')
    n_repeats = 30

    #print(model.layers[2].get_weights())

    '''
    # prepare data for problem 난이도 중간 문제 
    rainX, trainy, testX, testy, _, _ = samples_for_seed(2)
    
    dists, dist_labels = list(), list()

    # repeated evaluation of standalone model
    standalone_scores = eval_standalone_model(trainX, trainy, testX, testy, n_repeats)
    print('Standalone %.3f (%.3f)' % (mean(standalone_scores), std(standalone_scores)))
    dists.append(standalone_scores)
    dist_labels.append('standalone')
    '''

    '''
    # repeated evaluation of transfer learning model, vary fixed layers
    
    n_fixed = 3
    for i in range(n_fixed):
        scores = eval_transfer_model(trainX, trainy, testX, testy, i, n_repeats)
        print('Transfer (fixed=%d) %.3f (%.3f)' % (i, mean(scores), std(scores)))
        dists.append(scores)
        dist_labels.append('transfer f=' + str(i))

    '''
    dists, dist_labels = list(), list()

    trainX, trainy, testX, testy, _, _ = samples_for_easy(random_seed)
    scores = eval_transfer_model_jh_use_w_param(trainX, trainy, testX, testy, 0, n_repeats)
    dists.append(scores)
    dist_labels.append("easy weight reuse")
    print('easy weight reuse %.3f (%.3f)' % (mean(scores), std(scores)))


    scores = eval_transfer_model_jh_set_zero(trainX, trainy, testX, testy, 0, n_repeats)
    dists.append(scores)
    dist_labels.append("easy weight zero")
    print('easy weight init %.3f (%.3f)' % (mean(scores), std(scores)))

    eval_ori_model(testX, testy, "easyData")
    #================================================================================================================

    trainX, trainy, testX, testy, _, _ = samples_for_non_easy(random_seed)

    scores = eval_transfer_model_jh_use_w_param(trainX, trainy, testX, testy, 0, n_repeats)
    dists.append(scores)
    dist_labels.append("difficult weight reuse")
    print('difficult dataset weight reuse  %.3f (%.3f)' % (mean(scores), std(scores)))


    scores = eval_transfer_model_jh_set_zero(trainX, trainy, testX, testy, 0, n_repeats)
    dists.append(scores)
    dist_labels.append("difficult weight zero")
    print('difficult dataset weight init  %.3f (%.3f)' % (mean(scores), std(scores)))

    eval_ori_model(testX, testy, "Difficult Dataset")
    #================================================================================================================

    data_type = 10

    std_var_a = 2.4
    std_var_b = 2.8
    increment_std = 0.4

    increas_r = 0.2
    range_val = 7


    for i in range(data_type):

        #total different
        trainX, trainy, testX, testy, _, _  = sample_for_test(random_seed, std_var_a, std_var_b, range_val)

        scores = eval_transfer_model_jh_use_w_param(trainX, trainy, testX, testy, 0, n_repeats)
        print('different dataset %d reuse  %.3f (%.3f)' % (i, mean(scores), std(scores)))


        dists.append(scores)
        dist_labels.append("reuse " + str(i))

        scores = eval_transfer_model_jh_set_zero(trainX, trainy, testX, testy, 0, n_repeats)
        print('different dataset %d init  %.3f (%.3f)' % (i, mean(scores), std(scores)))

        eval_ori_model(testX, testy, "Different seed" + str(i))
        dists.append(scores)
        dist_labels.append("init " +  str(i))

        std_var_a = std_var_a + increment_std
        std_var_b = std_var_b + increment_std
        range_val = range_val + increas_r

        #end


    # box and whisker plot of score distributions
    pyplot.boxplot(dists, labels=dist_labels)
    #pyplot.figure(figsize=(100, 20))
    pyplot.show()

    #pyplot.boxplot(dist_ori, labels=dist_labels_ori)
    # pyplot.figure(figsize=(100, 20))
    #pyplot.show()

