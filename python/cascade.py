import numpy as np
import time
from tqdm import tqdm
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import cluster
from sklearn import metrics
import pyximport
pyximport.install()
import mcc

UseAllFeature=True
WriteFeatureFilt=False

start_time=time.time()
print('Load basic data ... ')
numeric_feat=np.load('../data/numeric_feat.npz')['arr_0']
#feature_filter_ttest=np.load('../data/feature_filter_ttest.npy')
#numeric_feat=numeric_feat[:, feature_filter_ttest]
date_feat=np.load('../data/date_feat.npy')
label=np.load('../data/label.npy')
feat=np.hstack((numeric_feat, date_feat))

#numeric_hidden=np.load('../data/numeric_hidden.npz')['arr_0']
#feat=np.hstack((feat, numeric_hidden))
print('Finished: {} minutes'.format(round((time.time() - start_time)/60, 2)))


feature_filter=np.full(feat.shape[1], False, np.bool)
if not UseAllFeature:
    feature_filter=np.load('../data/feature_filter.npy')
    feat=feat[:, feature_filter]

print('Create 2-fold cv ... ')
X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
    feat, label, stratify=label, test_size=0.5, random_state=777)
print('Finished: {} minutes'.format(round((time.time() - start_time)/60, 2)))

del(numeric_feat)
del(date_feat)
del(feat)
del(label)


tree_number=100
random_seed=777
fn=0.1
clf=ensemble.RandomForestClassifier(n_estimators=tree_number, max_features=0.1, \
                                    random_state=random_seed, verbose=1, n_jobs=16, oob_score=True, class_weight={1:100, 0:1})

def cascade(X_train, X_test, y_train, y_test):
    print('Create cascade ... ')
    clf.fit(X_train, y_train)
    if UseAllFeature:
        cutoff_importance=np.percentile(clf.feature_importances_, 90)
        global feature_filter
        feature_filter = feature_filter | np.array(clf.feature_importances_ >= cutoff_importance)
    ytep=clf.predict_proba(X_test)[:,1]
    true_counts=sum(y_test)
    print('Test true counts:' + str(true_counts))
    threshold=0
    total_tn=len(clf.estimators_)
    for i in range(total_tn):
        FN=sum(ytep[y_test==1] <= i/total_tn)
        if FN > fn * true_counts:
            threshold= i / total_tn
            break
    print('Test FN: '+ str(FN))
    tef=ytep > threshold
    remaining_counts=sum(ytep > threshold)
    print('Test Remaining: ' + str(remaining_counts))
    print('Test postive rate: ' + str((true_counts - FN) / remaining_counts))
    clf.fit(X_test, y_test)
    if UseAllFeature:
        cutoff_importance=np.percentile(clf.feature_importances_, 90)
        global feature_filter
        feature_filter = feature_filter | np.array(clf.feature_importances_ >= cutoff_importance)
    ytrp=clf.predict_proba(X_train)[:,1]
    trf=ytrp > threshold
    true_counts = sum(y_train)
    FN=sum(ytrp[y_train==1]<=threshold)
    remaining_counts=sum(ytrp > threshold)
    print('Train true counts:' + str(true_counts))
    print('Train FN: '+ str(FN))
    print('Train Remaining: ' + str(remaining_counts))       
    print('Train postive rate: ' + str((true_counts - FN) / remaining_counts))
    return trf, tef, ytrp, ytep



train_filter=np.full(len(y_train), True, np.bool)
test_filter=np.full(len(y_test), True, np.bool)
y_train_pred=np.full(len(y_train), 0, np.float)
y_test_pred=np.full(len(y_test), 0, np.float)

def run_level(X_train0, X_test0, y_train0, y_test0, level=0, max_level=np.inf, best_mcc=0):
    print('****************************************************************************')
    print('Current level:' + str(level))
    params={'n_estimators': min(max(sum(y_train0==1), sum(y_test0==1)), round(tree_number * 2 ** level)), 'class_weight': {1:100/(level+1), 0:1}}
    clf.set_params(**params)
    train_filter1, test_filter1, y_train_pred0, y_test_pred0 = cascade(X_train0, X_test0, y_train0, y_test0)
#    if not max(y_train_pred0) > 0: raise AssertionError
#    if not sum(train_filter) == len(y_train_pred0): raise AssertionError
    y_train_pred[~train_filter]=0
    y_train_pred[train_filter]=y_train_pred0
    y_test_pred[~test_filter]=0
    y_test_pred[test_filter]=y_test_pred0
    best_prob0, tmp0, tmp1 = mcc.eval_mcc(y_train, y_train_pred, show = True) 
    best_prob1, tmp0, tmp1 = mcc.eval_mcc(y_test, y_test_pred, show = True)    
    best_prob = (best_prob0 + best_prob1) / 2
    current_mcc = (metrics.matthews_corrcoef(y_train, (y_train_pred > best_prob).astype(int)) + \
                   metrics.matthews_corrcoef(y_test, (y_test_pred > best_prob).astype(int))) / 2
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
if UseAllFeature and WriteFeatureFilt:
    np.save('../data/feature_filter.npy', feature_filter)
