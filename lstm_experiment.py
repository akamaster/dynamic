import matplotlib
matplotlib.use('Agg')
import lasagne
import theano
import theano.tensor as T
from lasagne.layers import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

class LastDimLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input[:,-1,:]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

class ExactSeqOutput(lasagne.layers.MergeLayer):
    def get_output_for(self, input, **kwargs):
        lstm_seq = input[0]
        seq_lens = input[1]-1
        return lstm_seq[T.arange(lstm_seq.shape[0]), T.cast(seq_lens, 'int32'), :]

    def get_output_shape_for(self, input_shapes):
        input = input_shapes[0]
        return (input[0], input[2])


def build_nn(num_units_lstm=200, num_units_dense1=50, penalty_l_dense1=0.001, penalty_l_dense2=0.001, **kwargs):
    print(locals())
    num_inputs, num_classes = 300, 4
    l_inp_words = InputLayer((None, None, num_inputs))
    batch_size, seq_length, _ = l_inp_words.shape
    l_inp_seq_lens = InputLayer((batch_size,))

    l_lstm_fwd = LSTMLayer(l_inp_words, num_units=num_units_lstm, grad_clipping=0.4)
    l_lstm_fwd_output = ExactSeqOutput([l_lstm_fwd, l_inp_seq_lens])

    #l_lstm_fwd2 = LSTMLayer(l_lstm_fwd, num_units=num_units, grad_clipping=0.4)
    #l_lstm_fwd2_output = ExactSeqOutput([l_lstm_fwd2, l_inp_seq_lens])

    l_lstm_bkw = LSTMLayer(l_inp_words, num_units=num_units_lstm, backwards=True, grad_clipping=0.4)
    l_lstm_bkw_output = ExactSeqOutput([l_lstm_bkw, l_inp_seq_lens])

    #l_lstm_bkw2 = LSTMLayer(l_lstm_bkw, num_units=num_units, backwards=True, grad_clipping=0.4)
    #l_lstm_bkw2_output = ExactSeqOutput([l_lstm_bkw2, l_inp_seq_lens])

    l_lstm_out = ConcatLayer([l_lstm_fwd_output, l_lstm_bkw_output], axis=1)

    l_dropout1 = DropoutLayer(l_lstm_out, p=0.5)
    l_dense1 = DenseLayer(l_dropout1, num_units=num_units_dense1, nonlinearity=lasagne.nonlinearities.tanh)
    l_dropout2 = DropoutLayer(l_dense1,p=0.5)
    l_dense2 = DenseLayer(l_dropout2, num_units=num_classes, nonlinearity=lasagne.nonlinearities.softmax)

    l_out = l_dense2
    y_batch = T.vector('y_batch', dtype='int64')
    network_input_params = [l_inp_words.input_var, l_inp_seq_lens.input_var, y_batch]
    output_train = get_output(l_out)
    output_test = get_output(l_out, deterministic=True)

    def loss_computation(current, target):
        objective = lasagne.objectives.categorical_crossentropy(current, target)
        objective = objective.mean()
        layers = {l_dense1: penalty_l_dense1, l_dense2: penalty_l_dense2}
        from lasagne.regularization import l2
        l2_penalty = lasagne.regularization.regularize_layer_params_weighted(layers, l2)
        objective += l2_penalty

        return objective

    objective_train = loss_computation(output_train, y_batch)
    objective_test = loss_computation(output_test, y_batch)

    accuracy_train = lasagne.objectives.categorical_accuracy(output_train, y_batch).mean()
    accuracy_test = lasagne.objectives.categorical_accuracy(output_test, y_batch).mean()

    params = lasagne.layers.get_all_params(l_out, trainable=True)
    updates = lasagne.updates.adagrad(
            objective_train, params, learning_rate=0.01)

    train = theano.function(network_input_params, [objective_train, accuracy_train, output_train],
                            updates=updates)
    test = theano.function(network_input_params, [objective_test, accuracy_test, output_test])

    return train, test, l_out


def iterate_minibatches(inputs, batchsize=10, number_of_minibatches = None, shuffle=True):
    if number_of_minibatches:
        current_batch = 0
        while True:
            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
            for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batchsize]
                else:
                    excerpt = slice(start_idx, start_idx + batchsize)
                current_batch += 1

                yield inputs[excerpt]

                if current_batch >= number_of_minibatches:
                    return
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

from lasagne.utils import floatX
from sklearn.cross_validation import StratifiedKFold
npz_file = np.load('word2vec_full_repr.npz')
data_train = floatX(npz_file['data_train'])
labels_train = npz_file['labels_train']
seq_lens_train = npz_file['seq_lens_train']

print(data_train.shape, data_train.dtype)
print(labels_train.shape, labels_train.dtype)
print(seq_lens_train.shape, seq_lens_train.dtype)


def get_actual_data(data, labels, seq_lens, minibatch_index_gen):
    for index in minibatch_index_gen:
        # microoptimization, we just need first seq_lens inputs from data,
        # therefore, max_seq_lens should be enough
        returned_val = seq_lens[index]
        max_lens = max(returned_val)
        yield data[index][:,:max_lens,:], labels[index], returned_val

path = 'nn_run/'
import gzip, pickle
import math
def worker(data_train, labels_train, seq_lens_train, worker_name='', num_iterations=1500, cv_every=100, **kwargs):
    all_passed_args = locals()
    del all_passed_args['worker_name']
    del all_passed_args['data_train']
    del all_passed_args['labels_train']
    del all_passed_args['seq_lens_train']

    filename = '__'.join('__'.join(kk+'='+str(vv) for kk, vv in v.items()) if isinstance(v, dict)
                         else k+'='+str(v)
                         for k,v in all_passed_args.items())

    cross_validation_gen = StratifiedKFold(labels_train, n_folds=3, shuffle=True)
    encoder = LabelEncoder()

    loss_train_per_cvindex = []
    loss_cv_per_cvindex = []

    early_stop_loss = []
    early_stop_iteration_number = []
    early_stop_accuracy = []
    end_of_training_accuracy = []
    end_of_training_loss = []

    for cv_index, (index_train, index_cv) in enumerate(cross_validation_gen):
        targets_training_actual = encoder.fit_transform(labels_train)
        train_proc, test_proc, network = build_nn(**kwargs)

        convergence_data_train = []
        convergence_data_cv_y = []
        convergence_data_cv_x = []
        convergence_data_cv_acc = []
        targets_cv = encoder.transform(labels_train[index_cv])

        for i, (d_batch, t_batch, s_batch) in enumerate(get_actual_data(data_train,
                                                         targets_training_actual,
                                                         seq_lens_train,
                                                         iterate_minibatches(index_train,
                                                                             batchsize=20,
                                                                             number_of_minibatches=(num_iterations+1)))):
            loss, acc, pred = train_proc(d_batch,s_batch,t_batch)
            print("W:",worker_name,"cv_ind:",cv_index, i, loss, acc)
            convergence_data_train.append(loss)

            if i%cv_every == 0:
                loss_cv, acc_cv, pred_cv = test_proc(data_train[index_cv],
                                                     seq_lens_train[index_cv],
                                                     targets_cv)
                print("W:",worker_name,"cv_ind:",cv_index, "CROSS VAL:", loss_cv, acc_cv)
                convergence_data_cv_y.append(loss_cv)
                convergence_data_cv_x.append(i)
                convergence_data_cv_acc.append(acc_cv)

        fig = plt.figure()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.ylim([0,2])
        plt.title(' '.join(filename.split('__')))
        plt.plot(convergence_data_train, color='green')
        plt.plot(convergence_data_cv_x, convergence_data_cv_y, '--rx')
        fig.savefig(path+filename+'__cv_index='+str(cv_index)+'.png', dpi=400)
        plt.close(fig)

        loss_train_per_cvindex.append(convergence_data_train)
        loss_cv_per_cvindex.append((convergence_data_cv_x, convergence_data_cv_y))

        end_of_training_accuracy.append(convergence_data_cv_acc[-1])
        end_of_training_loss.append(convergence_data_cv_y[-1])


        min_loss = min(convergence_data_cv_y)
        ind_min_loss = convergence_data_cv_y.index(min_loss)
        early_stop_loss.append(min_loss)
        early_stop_accuracy.append(convergence_data_cv_acc[ind_min_loss])
        early_stop_iteration_number.append(cv_every*ind_min_loss)


    params = {'train':loss_train_per_cvindex, 'cv': loss_train_per_cvindex}
    tmp_file = gzip.open(path+filename+'.pklz', mode='wb')
    pickle.dump(params, file=gzip.open(path+filename+'.pklz', mode='wb'))
    tmp_file.close()

    print(end_of_training_loss)
    print(end_of_training_accuracy)
    print(early_stop_loss)
    print(early_stop_accuracy)
    print(early_stop_iteration_number)

    return np.mean(end_of_training_accuracy), np.mean(end_of_training_loss), \
           np.mean(early_stop_accuracy), np.mean(early_stop_loss), np.mean(early_stop_iteration_number)

#print(worker('asdf', data_train, labels_train, seq_lens_train, num_iterations=1500, num_units_lstm=100, num_units_dense1=50))

grid_search_params = { 'num_units_lstm':[50,100],#,200,300],
                       'num_units_dense1':[50]}#,100,200,300] }

import itertools

all_combinations = list((dict(zip(grid_search_params.keys(), p)) for p in itertools.product(*grid_search_params.values())))

for i, p in enumerate(all_combinations):
    p['worker'] = 'W'+str(i)


def worker_helper(params):
    return worker(data_train, labels_train, seq_lens_train, **params)

from multiprocessing import Pool
p = Pool()
f = open('logs_nn.txt', mode='w')
print(all_combinations,file=f)
results = np.array(p.map(worker_helper, all_combinations))
print(results,file=f)

# best_early_stop_loss = results[:,3].max()
# ind = results[:,2].argmax()
# best_early_stop_it = results[ind,4]
# best_early_stop_acc = results[ind,2]
#
# print('Best early stop occured with loss={.4}, it={}, acc={}'
#       .format(best_early_stop_loss, best_early_stop_it, best_early_stop_acc))
#
# best_end_of_training_loss = results[:,1].max()
# ind = results[:,1].argmax()
# best_end_of_training_acc = results[]


