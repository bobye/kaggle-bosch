import numpy as np
import time
from tqdm import tqdm
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import cluster
import pyximport
pyximport.install()
import mcc

start_time=time.time()
print('Load basic data ... ')
numeric_feat=np.load('../data/numeric_feat.npz')['arr_0']
label=np.load('../data/label.npy')
print('Finished: {} minutes'.format(round((time.time() - start_time)/60, 2)))

print('Create 2-fold cv ... ')
X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
    numeric_feat, label, stratify=label, test_size=0.5, random_state=777)
print('Finished: {} minutes'.format(round((time.time() - start_time)/60, 2)))


tree_number=100
random_seed=777
fn=0.01
clf=ensemble.RandomForestClassifier(n_estimators=tree_number, random_state=random_seed, verbose=1, n_jobs=4, oob_score=True, class_weight={1:100, 0:1}) # ~10 minutes

def cascade(X_train, X_test, y_train, y_test):
    print('Create cascade ... ')
    clf.fit(X_train, y_train)
    y_test_pred=clf.predict_proba(X_test)[:,1]
    true_counts=sum(y_test)
    print('Test true counts:' + str(true_counts))
    threshold=0
    for i in range(tree_number):
        FN=sum(y_test_pred[y_test==1] <= i/tree_number)
        if FN > fn * true_counts:
            threshold= i / tree_number
            break
    print('Test FN: '+ str(FN))
    test_filter=y_test_pred > threshold
    remaining_counts=sum(y_test_pred > threshold)
    print('Test Remaining: ' + str(remaining_counts))
    clf.fit(X_test, y_test)
    y_train_pred=clf.predict_proba(X_train)[:,1]
    train_filter=y_train_pred > threshold
    print('Train true counts:' + str(sum(y_train)))
    print('Train FN: '+ str(sum(y_train_pred[y_train==1]<=threshold)))
    print('Train Remaining: ' + str(sum(y_train_pred > threshold)))       
    return train_filter, test_filter, y_train_pred, y_test_pred



train_filter=np.full(len(y_train), True, np.bool)
test_filter=np.full(len(y_test), True, np.bool)
y_train_pred=np.full(len(y_train), 0, np.int32)
y_test_pred=np.full(len(y_test), 0, np.int32)

def run_level(X_train0, X_test0, y_train0, y_test0, level=0, max_level=np.inf, best_mcc=0):
    print('****************************************************************************')
    print('Current level:' + str(level))
    params={'n_estimators': tree_number * 2 ** level, 'class_weight': {1:100/(level+1), 0:1}}
    clf.set_params(**params)
    
    train_filter1, test_filter1, y_train_pred0, y_test_pred0 = cascade(X_train0, X_test0, y_train0, y_test0)
    y_train_pred[~train_filter]=0
    y_train_pred[train_filter]=y_train_pred0
    y_test_pred[~test_filter]=0
    y_test_pred[test_filter]=y_test_pred0
    current_mcc=(mcc.eval_mcc(y_train, y_train_pred) + mcc.eval_mcc(y_test, y_test_pred))/2
    print('MCC:' + str(current_mcc))
    if current_mcc > best_mcc and level < max_level:
        best_mcc=current_mcc
        X_train1=X_train0[train_filter1]
        y_train1=y_train0[train_filter1]
        X_test1=X_test0[test_filter1]
        y_test1=y_test0[test_filter1]
        train_filter[train_filter]=train_filter1
        test_filter[test_filter]=test_filter1
        run_level(X_train1, X_test1, y_train1, y_test1, level=level+1, best_mcc=best_mcc)
        

run_level(X_train, X_test, y_train, y_test)










