#!/usr/bin/env python
"""CEVAE model on IHDP
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from datasets import IHDP
from evaluation import Evaluator
import numpy as np
import time
from scipy.stats import sem
from nn import build_models
from keras.utils import to_categorical

from sklearn.utils import class_weight
from argparse import ArgumentParser
from utils import batch_generator
import distutils.util as ut

parser = ArgumentParser()
parser.add_argument('-reps', type=int, default=10)
parser.add_argument('-earl', type=int, default=10)
parser.add_argument('-iterations', type=int, default=10000)
parser.add_argument('-print_every', type=int, default=1000)
parser.add_argument('-DANN', type=lambda x: bool(ut.strtobool(x)), default=False)
args = parser.parse_args()

args.true_post = True

dataset = IHDP(replications=args.reps)

scores = np.zeros((args.reps, 3))
scores_test = np.zeros((args.reps, 3))

batch_size = 128
SAMPLING_ITERATIONS = int(args.iterations)
n_neurons = 64

for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
    print('\nReplication {}/{}, DANN:{}'.format(i + 1, args.reps, args.DANN))
    (xtr, ttr, ytr), (y_cftr, mu0tr, mu1tr) = train
    (xva, tva, yva), (y_cfva, mu0va, mu1va) = valid
    (xte, tte, yte), (y_cfte, mu0te, mu1te) = test

    print(sum(ttr), sum(tva), sum(tte))

    # xm, xs = np.mean(xtr, axis=0), np.std(xtr, axis=0)
    # xtr, xva, xte = (xtr - xm) / xs, (xva - xm) / xs, (xte - xm) / xs

    evaluator_test = Evaluator(yte, tte, y_cf=y_cfte, mu0=mu0te, mu1=mu1te)

    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate(
        [ytr, yva], axis=0)
    evaluator_train = Evaluator(yalltr, talltr, y_cf=np.concatenate([y_cftr, y_cfva], axis=0),
                                mu0=np.concatenate([mu0tr, mu0va], axis=0), mu1=np.concatenate([mu1tr, mu1va], axis=0))

    model, regressor_model, domain_classification_model, embeddings_model = build_models(shape=(xalltr.shape[1],),
                                                                                         n_neurons=n_neurons)

    # zero mean, unit variance for x and y during training

    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

    cw_s = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(ttr), np.squeeze(ttr))))  #
    cw_t = {0: cw_s[1], 1: cw_s[0]}
    print(cw_s, cw_t, np.unique(ttr))
    index_0 = ttr[:, 0] > 0.5
    index_1 = ttr[:, 0] < 0.5
    cttr = to_categorical(ttr)
    # batches_0 = batch_generator([xtr[index_0], ttr[index_0],cttr[index_0] , ytr[index_0]], batch_size)
    # batches_1 = batch_generator([xtr[index_1], ttr[index_1], cttr[index_1], ytr[index_1]], batch_size)
    batches = batch_generator([xtr, ttr, cttr, ytr], batch_size)

    t0 = time.time()

    for j in range(SAMPLING_ITERATIONS):

        # X_batch_0, t_batch_0, tc_batch_0, y_batch_0 = next(batches_0)
        # X_batch_1, t_batch_1, tc_batch_1, y_batch_1 = next(batches_1)
        #
        # X_batch = np.concatenate([X_batch_0, X_batch_1])
        # t_batch = np.concatenate([t_batch_0, t_batch_1])
        #
        # tc_batch = np.concatenate([tc_batch_0, tc_batch_1])
        # y_batch = np.concatenate([y_batch_0, y_batch_1])

        X_batch, t_batch, tc_batch, y_batch = next(batches)

        if (args.DANN):

            features = embeddings_model.predict(X_batch)
            stats2 = domain_classification_model.train_on_batch([features], tc_batch, class_weight=[None, cw_s])



            adv_weights = []
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    adv_weights.append(layer.get_weights())

            stats = model.train_on_batch([X_batch, t_batch], [y_batch, 1 - tc_batch, ], class_weight=[None, cw_t])

            k = 0
            for layer in model.layers:
                if (layer.name.startswith("do")):
                    layer.set_weights(adv_weights[k])
                    k += 1

        else:
            stats = regressor_model.train_on_batch([X_batch, t_batch], y_batch)

        if j % args.print_every == 0:
            y0 = model.predict([xalltr, np.zeros(shape=(len(xalltr), 1))])[0]
            y1 = model.predict([xalltr, np.ones(shape=(len(xalltr), 1))])[0]
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score_train = evaluator_train.calc_stats(y1, y0)
            rmses_train = evaluator_train.y_errors(y0, y1)

            y0t = model.predict([xte, np.zeros(shape=(len(xte), 1))])[0]
            y1t = model.predict([xte, np.ones(shape=(len(xte), 1))])[0]
            y0t, y1t = y0t * ys + ym, y1t * ys + ym
            score_test = evaluator_test.calc_stats(y1t, y0t)

            print("Epoch: {}/{},, ite_tr: {:0.3f}, ate_tr: {:0.3f}, pehe_tr: {:0.3f}, " \
                  "rmse_f_tr: {:0.3f}, rmse_cf_tr: {:0.3f}, ite_te: {:0.3f}, ate_te: {:0.3f}, pehe_te: {:0.3f}, " \
                  "dt: {:0.3f}".format(j + 1, SAMPLING_ITERATIONS, score_train[0], score_train[1], score_train[2],
                                       rmses_train[0], rmses_train[1], score_test[0], score_test[1], score_test[2],
                                       time.time() - t0), stats)
            t0 = time.time()

    y0 = model.predict([xalltr, np.zeros(shape=(len(xalltr), 1))])[0]
    y1 = model.predict([xalltr, np.ones(shape=(len(xalltr), 1))])[0]
    y0, y1 = y0 * ys + ym, y1 * ys + ym
    score = evaluator_train.calc_stats(y1, y0)
    rmses_train = evaluator_train.y_errors(y0, y1)

    scores[i, :] = score

    y0t = model.predict([xte, np.zeros(shape=(len(xte), 1))])[0]
    y1t = model.predict([xte, np.ones(shape=(len(xte), 1))])[0]
    y0t, y1t = y0t * ys + ym, y1t * ys + ym
    score_test = evaluator_test.calc_stats(y1t, y0t)
    scores_test[i, :] = score_test

    print('Replication: {}/{}, tr_ite: {:0.3f}, tr_ate: {:0.3f}, tr_pehe: {:0.3f}' \
          ', te_ite: {:0.3f}, te_ate: {:0.3f}, te_pehe: {:0.3f}'.format(i + 1, args.reps,
                                                                        score[0], score[1], score[2],
                                                                        score_test[0], score_test[1], score_test[2]), j,
          stats)

print('DANN model total scores')
means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
print('train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}, train PEHE: {:.3f}+-{:.3f}' \
      ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))

means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
print('test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}, test PEHE: {:.3f}+-{:.3f}' \
      ''.format(means[0], stds[0], means[1], stds[1], means[2], stds[2]))
