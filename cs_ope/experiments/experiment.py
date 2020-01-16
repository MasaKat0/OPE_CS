import numpy as np
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file


def data_generation(data_name):
    X, Y = load_svmlight_file('data/%s'%data_name)
    X = X.toarray()
    X = X/X.max(axis=0)
    Y = np.array(Y, np.int64)

    N = len(X)
    N_train = np.int(N*0.7)
    N_test= N - N_train

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
    
    return X, Y, Y_matrix, train_test_split, prob, classes, N, N_train, N_test

def behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha=0.7):
    N = len(X)
    num_class = len(classes)
    
    X_train, X_test = X[train_test_split], X[~train_test_split]
    Y_train, Y_test = Y[train_test_split], Y[~train_test_split]
    
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
    
def experiment(data_name):
    num_trials = 100
alphas = [0.7, 0.4, 0.0]

tau_list = np.zeros(num_trials)
res_ipw3_list = np.zeros((num_trials, len(alphas)))
res_dm_list = np.zeros((num_trials, len(alphas)))
res_dr1_list = np.zeros((num_trials, len(alphas)))
res_dr2_list = np.zeros((num_trials, len(alphas)))
res_dml1_truew_list = np.zeros((num_trials, len(alphas)))
res_dml2_truew_list = np.zeros((num_trials, len(alphas)))

for trial in range(num_trials):
    X, Y, Y_matrix, train_test_split, x_prob, classes, N, N_train, N_test = data_generation(data_name)

    X_train, X_test = X[train_test_split], X[~train_test_split]
    Y_train, Y_test = Y[train_test_split], Y[~train_test_split]
    Y_matrix_train, Y_matrix_test = Y_matrix[train_test_split], Y_matrix[~train_test_split]

    pi_behavior, pi_evaluation  = behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha=0.7)

    pi_behavior_train, pi_behavior_test = pi_behavior[train_test_split], pi_behavior[~train_test_split]
    pi_evaluation_train, pi_evaluation_test = pi_evaluation[train_test_split], pi_evaluation[~train_test_split]

    tau = true_value(Y_matrix_test, pi_evaluation_test, N_test)

    for idx_alpha in  range(len(alphas)):    
        alpha = alphas[idx_alpha]
        pi_behavior, pi_evaluation  = behavior_and_evaluation_policy(X, Y, train_test_split, classes, alpha=alpha)

        perm = np.random.permutation(N_train)

        X_seq_train, Y_matrix_seq_train, pi_behavior_seq_train, pi_evaluation_seq_train = X_train[perm], Y_matrix_train[perm], pi_behavior_train[perm], pi_evaluation_train[perm]

        Y_historical_matrix = np.zeros(shape=(N_train, len(classes)))
        A_historical_matrix = np.zeros(shape=(N_train, len(classes)))

        for i in range(N_train):
            a = np.random.choice(classes, p=pi_behavior[i])
            Y_historical_matrix[i, a] = 1
            A_historical_matrix[i, a] = 1

        res_ipw3 = ipw(Y_historical_matrix, X_seq_train, X_test, classes, pi_evaluation_train, A_hst=A_historical_matrix)
        res_dm = dm(Y_historical_matrix, X_seq_train, X_test, pi_evaluation_test, classes)
        res_dr1 = dr(Y_historical_matrix, A_historical_matrix, X_seq_train, X_test, pi_evaluation_seq_train, pi_evaluation_test, classes, pi_behavior=pi_behavior_seq_train, method='Lasso')
        res_dr2 = dr(Y_historical_matrix, A_historical_matrix, X_seq_train, X_test, pi_evaluation_seq_train, pi_evaluation_test, classes, pi_behavior=pi_behavior_seq_train, method='Ridge')
        res_dml1 =dml(Y_historical_matrix, A_historical_matrix, X_seq_train, X_test, pi_evaluation_seq_train, pi_evaluation_test, classes, pi_behavior=pi_behavior_seq_train, method='Lasso')
        res_dml2 =dml(Y_historical_matrix, A_historical_matrix, X_seq_train, X_test, pi_evaluation_seq_train, pi_evaluation_test, classes, pi_behavior=pi_behavior_seq_train, method='Ridge')
        res_dml1_truew =dml(Y_historical_matrix, A_historical_matrix, X_seq_train, X_test, pi_evaluation_seq_train, pi_evaluation_test, classes, pi_behavior=pi_behavior_seq_train, w_est=True, method='Lasso')
        res_dml2_truew=dml(Y_historical_matrix, A_historical_matrix, X_seq_train, X_test, pi_evaluation_seq_train, pi_evaluation_test, classes, pi_behavior=pi_behavior_seq_train, w_est=True, method='Ridge')

        print(res_ipw3)
        print(res_dm)
        print(res_dr1)
        print(res_dr2)
        print(res_dml1_truew)
        print(res_dml2_truew)

        res_ipw3_list[trial, idx_alpha] = res_ipw3
        res_dm_list[trial, idx_alpha] = res_dm
        res_dr1_list[trial, idx_alpha] = res_dr1
        res_dr2_list[trial, idx_alpha] = res_dr2
        res_dml1_truew_list[trial, idx_alpha] = res_dml1_truew
        res_dml2_truew_list[trial, idx_alpha] = res_dml2_truew

        np.savetxt("exp_results/res_ipw3.csv", res_ipw3_list, delimiter=",")
        np.savetxt("exp_results/res_dm.csv", res_dm_list, delimiter=",")
        np.savetxt("exp_results/res_dr1.csv", res_dr1_list, delimiter=",")
        np.savetxt("exp_results/res_dr2.csv", res_dr2_list, delimiter=",")
        np.savetxt("exp_results/res_dml1.csv", res_dml1_truew_list, delimiter=",")
        np.savetxt("exp_results/res_dml2.csv", res_dml2_truew_list, delimiter=",")

    tau_list[trial] = tau
    
if __name__ == '__main__':
    args = sys.argv
    data_name = args[1]
    experiment(data_name)

    