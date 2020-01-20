import numpy as np
import argparse
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file

sys.path.append('../')
print(sys.path)

from cs_ope_estimator import ope_estimators


def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='Covariate Shift Adaptation for Off-policy Evaluation and Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', type=str, default='staimage',
                        help='Name of dataset')
    parser.add_argument('--sample_size', '-s', type=int, default=1000,
                        help='Sample size')
    parser.add_argument('--num_trials', '-n', type=int, default=200,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['satimage', 'vehicle', 'pendigits'],
                        help="Presets of configuration")
    args = parser.parse_args(arguments)

    if args.preset == 'satimage':
        args.sample_size = 1000
        args.dataset = 'satimage'
        args.num_trials = 200
    elif args.preset == 'vehicle':
        args.sample_size = 1000
        args.dataset = 'vehicle'
        args.num_trials = 200
    elif args.preset == "pendigits":
        args.sample_size = 1000
        args.dataset = 'pendigits'
        args.num_trials = 200
    return args

def data_generation(data_name, N):
    X, Y = load_svmlight_file('data/%s'%data_name)
    X = X.toarray()
    X = X/X.max(axis=0)
    Y = np.array(Y, np.int64)

    N_train = np.int(N*0.7)
    N_test= N - N_train

    perm = np.random.permutation(len(X))

    X, Y = X[perm[:N]], Y[perm[:N]]

    if data_name == 'satimage.scale':
        Y = Y - 1

    classes = np.unique(Y)

    Y_matrix = np.zeros(shape=(N, len(classes)))

    for i in range(N):
        Y_matrix[i, Y[i]] = 1

    prob = 1/(1+np.exp(-(X[:,0]+X[:,1]+X[:,2]+X[:,3]+X[:,4]+X[:,5]+X[:,6]+0.1*np.random.normal(size=len(X)))))
    rand = np.random.uniform(size=len(X))

    prob_base = prob

    for C in range(100000):
        C /= 1000
        eval_samplesize = np.sum(prob*C > rand)
        if eval_samplesize > N_train:
            break
        prob_base = prob*C

    train_test_split = prob_base > rand
    
    if np.sum(train_test_split) != N_train:
        N_train = np.sum(train_test_split)
        N_test = N - N_train
    
    return X, Y, Y_matrix, train_test_split, classes, N, N_train, N_test

def behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha=0.7):
    N = len(X)
    num_class = len(classes)
    
    X_train = X[train_test_split]
    Y_train = Y[train_test_split]
    
    classifier = LogisticRegression(random_state=0, penalty='l2', C=0.1, solver='saga', multi_class='multinomial',).fit(X_train, Y_train)
    predict = np.array(classifier.predict(X), np.int64)

    pi_predict = np.zeros(shape=(N, num_class))

    for i in range(N):
        pi_predict[i, predict[i]] = 1

    pi_random = np.random.uniform(size=(N, num_class))
    
    pi_random = pi_random.T
    pi_random /= pi_random.sum(axis=0)
    pi_random = pi_random.T
    
    
    pi_behavior = alpha*pi_predict + (1-alpha)*pi_random
        
    pi_evaluation = 0.9*pi_predict + 0.1*pi_random
        
    return pi_behavior, pi_evaluation

def true_value(Y_matrix, pi_evaluation, N):
     return np.sum(Y_matrix*pi_evaluation)/N
    
def main(arguments):
    args = process_args(arguments)

    data_name = args.dataset
    num_trials = args.num_trials
    sample_size = args.sample_size

    if data_name == 'satimage':
        data_name = 'satimage.scale'
    
    alphas = [0.7, 0.4, 0.0]

    tau_list = np.zeros(num_trials)
    res_ipw3_list = np.zeros((num_trials, len(alphas)))
    res_dm_list = np.zeros((num_trials, len(alphas)))
    res_dml1_list = np.zeros((num_trials, len(alphas)))
    res_dml2_list = np.zeros((num_trials, len(alphas)))

    res_ipw3_sn_list = np.zeros((num_trials, len(alphas)))
    res_dml1_sn_list = np.zeros((num_trials, len(alphas)))
    res_dml2_sn_list = np.zeros((num_trials, len(alphas)))

    for trial in range(num_trials):
        X, Y, Y_matrix, train_test_split, classes, N, N_train, N_test = data_generation(data_name, sample_size)

        X_train, X_test = X[train_test_split], X[~train_test_split]

        Y_matrix_train, Y_matrix_test = Y_matrix[train_test_split], Y_matrix[~train_test_split]

        pi_behavior, pi_evaluation  = behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha=0.7)

        pi_behavior_train = pi_behavior[train_test_split]
        pi_evaluation_train, pi_evaluation_test = pi_evaluation[train_test_split], pi_evaluation[~train_test_split]

        tau = true_value(Y_matrix_test, pi_evaluation_test, N_test)
        tau_list[trial] = tau

        for idx_alpha in  range(len(alphas)):    
            alpha = alphas[idx_alpha]
            pi_behavior, pi_evaluation  = behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha=alpha)

            perm = np.random.permutation(N_train)

            X_seq_train, Y_matrix_seq_train, pi_behavior_seq_train, pi_evaluation_seq_train = X_train[perm], Y_matrix_train[perm], pi_behavior_train[perm], pi_evaluation_train[perm]

            Y_historical_matrix = np.zeros(shape=(N_train, len(classes)))
            A_historical_matrix = np.zeros(shape=(N_train, len(classes)))

            for i in range(N_train):
                a = np.random.choice(classes, p=pi_behavior_seq_train[i])
                Y_historical_matrix[i, a] = Y_matrix_seq_train[i, a]
                A_historical_matrix[i, a] = 1

            estimators = ope_estimators(X_seq_train, A_historical_matrix, Y_historical_matrix, X_test, classes, pi_evaluation_seq_train, pi_evaluation_test)
            res_ipw3 = estimators.ipw(self_norm=False)
            res_ipw3_sn = estimators.ipw(self_norm=True)
            res_dm = estimators.dm()
            res_dml1 = estimators.dml(self_norm=False, method='Lasso')
            res_dml2 = estimators.dml(self_norm=False, method='Ridge')
            res_dml1_sn = estimators.dml(self_norm=True, method='Lasso')
            res_dml2_sn = estimators.dml(self_norm=True, method='Ridge')

            print('True:', tau)
            print('IPW3:', res_ipw3)
            print('IPW3_SN:', res_ipw3_sn)
            print('DM:', res_dm)
            print('DML1:', res_dml1)
            print('DML1_SN:', res_dml1_sn)
            print('DML2:', res_dml2)
            print('DML2_SN:', res_dml2_sn)

            res_ipw3_list[trial, idx_alpha] = res_ipw3
            res_ipw3_sn_list[trial, idx_alpha] = res_ipw3_sn
            res_dm_list[trial, idx_alpha] = res_dm
            res_dml1_list[trial, idx_alpha] = res_dml1
            res_dml1_sn_list[trial, idx_alpha] = res_dml1_sn
            res_dml2_list[trial, idx_alpha] = res_dml2
            res_dml2_sn_list[trial, idx_alpha] = res_dml2_sn

            np.savetxt("exp_results/true_value.csv", tau_list, delimiter=",")
            np.savetxt("exp_results/res_ipw3.csv", res_ipw3_list, delimiter=",")
            np.savetxt("exp_results/res_ipw3_sn.csv", res_ipw3_sn_list, delimiter=",")
            np.savetxt("exp_results/res_dm.csv", res_dm_list, delimiter=",")
            np.savetxt("exp_results/res_dml1.csv", res_dml1_list, delimiter=",")
            np.savetxt("exp_results/res_dml1_sn.csv", res_dml1_sn_list, delimiter=",")
            np.savetxt("exp_results/res_dml2.csv", res_dml2_list, delimiter=",")
            np.savetxt("exp_results/res_dml2_sn.csv", res_dml２_sn_list, delimiter=",")

        tau_list[trial] = tau
    
if __name__ == '__main__':
    main(sys.argv[1:])

    