import pandas as pd
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
CreateLBPrediction=False
WriteFeatureFilt=False
filename='../submission/rf'

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


X_lb=np.zeros((0,feat.shape[1]))
if CreateLBPrediction:
    print('Load basic LB data ... ')
    numeric_feat=np.load('../data/lb_numeric_feat.npz')['arr_0']
    date_feat=np.load('../data/lb_date_feat.npy')
    feat=np.hstack((numeric_feat, date_feat))
    X_lb = feat

## remove variables to save memory
del(numeric_feat)
del(date_feat)
del(feat)
del(label)

#######################################################################################
tree_number=100
random_seed=777
fn=0.1


def rf_class(X, y=None, todo="predict"):
    level=rf_class.level
    tree_numbers=rf_class.tree_numbers 
    if todo=="fit":
        params={'n_estimators': tree_numbers, 'class_weight': {1:100/(level+1), 0:1}}
        rf_class.clf.set_params(**params)
        rf_class.clf.fit(X, y)
        if UseAllFeature:
            cutoff_importance=np.percentile(rf_class.clf.feature_importances_, 90)
            global feature_filter
            feature_filter = feature_filter | np.array(rf_class.clf.feature_importances_ >= cutoff_importance)
        return None
    elif todo=="predict":
        return rf_class.clf.predict_proba(X)[:,1]
rf_class.clf=ensemble.RandomForestClassifier(n_estimators=tree_number, max_features=0.1, \
                                             random_state=random_seed, verbose=1, n_jobs=16, oob_score=True, class_weight={1:100, 0:1})
        
        
        

def cascade(X_train, X_test, y_train, y_test, X_lb=None):
    print('Create cascade ... ')
    rf_class(X_train, y_train, todo="fit")
    ytep=rf_class(X_test, todo="predict")
    true_counts=sum(y_test)
    print('Test true counts:' + str(true_counts))
    threshold=0
    total_tn=sorted(np.unique(ytep))
    for val in total_tn:
        FN=sum(ytep[y_test==1] <= val)
        if FN > fn * true_counts:
            threshold= val
            break
    print('Test FN: '+ str(FN))
    tef=ytep > threshold
    remaining_counts=sum(ytep > threshold)
    print('Test Remaining: ' + str(remaining_counts))
    print('Test postive rate: ' + str((true_counts - FN) / remaining_counts))
    rf_class(X_test, y_test, todo="fit")
    ytrp=rf_class(X_train, todo="predict")
    trf=ytrp > threshold
    true_counts = sum(y_train)
    FN=sum(ytrp[y_train==1]<=threshold)
    remaining_counts=sum(ytrp > threshold)
    print('Train true counts:' + str(true_counts))
    print('Train FN: '+ str(FN))
    print('Train Remaining: ' + str(remaining_counts))       
    print('Train postive rate: ' + str((true_counts - FN) / remaining_counts))

    if X_lb is not None:
        print('Create cascade_lb ... ')
        rf_class(np.vstack((X_train, X_test)), np.concatenate((y_train, y_test)), todo="fit")
        ylbp=rf_class(X_lb, todo="predict")
        lbf=ylbp > threshold

    if X_lb is None:
        return trf, tef, ytrp, ytep
    else:
        return trf, tef, lbf, ytrp, ytep, ylbp
        

train_filter=np.full(len(y_train), True, np.bool)
test_filter=np.full(len(y_test), True, np.bool)
y_train_pred=np.full(len(y_train), 0, np.float)
y_test_pred=np.full(len(y_test), 0, np.float)

lb_filter=np.full(X_lb.shape[0], True, np.bool)
y_lb_pred=np.full(X_lb.shape[0], 0, np.float)

def run_level(X_train0, X_test0, y_train0, y_test0, X_lb0=None, \
              level=0, max_level=np.inf, best_mcc=0):
    print('****************************************************************************')
    print('Current level:' + str(level))
    ## set parameters for random forest
    rf_class.level = level
    rf_class.tree_numbers =min(max(sum(y_train0==1), sum(y_test0==1)), round(tree_number * 2 ** level))
    ## call cascade classification
    if CreateLBPrediction:
        train_filter1, test_filter1, lb_filter1, y_train_pred0, y_test_pred0, y_lb_pred0 = cascade(X_train0, X_test0, y_train0, y_test0, X_lb0)
    else:
        train_filter1, test_filter1, y_train_pred0, y_test_pred0 = cascade(X_train0, X_test0, y_train0, y_test0)
    ## update predictions using returned filter
    y_train_pred[~train_filter]=0
    y_train_pred[train_filter]=y_train_pred0
    y_test_pred[~test_filter]=0
    y_test_pred[test_filter]=y_test_pred0
    if CreateLBPrediction:
        y_lb_pred[~lb_filter]=0
        y_lb_pred[lb_filter]=y_lb_pred0
    ## estimate the best mcc on 2-fold validation
    best_prob, current_mcc, tmp1 = mcc.eval_mcc(np.concatenate((y_train, y_test)), np.concatenate((y_train_pred, y_test_pred)), show = True)
    if CreateLBPrediction:
        prediction=(y_lb_pred > best_prob).astype(int)
        lb_id=pd.read_csv('.../data/lb_ID.csv')['Id'].values
        submission=pd.DataFrame({'Id':lb_id, 'Response':prediction})
        submission.to_csv(filename + '_level' + str(level) + '.csv', index=False)
    print('MCC:' + str(current_mcc))
    ## if it improves, start a new level of cascade
    if current_mcc > best_mcc and level < max_level:
        best_mcc=current_mcc
        ## filter data
        X_train1=X_train0[train_filter1]
        y_train1=y_train0[train_filter1]
        X_test1=X_test0[test_filter1]
        y_test1=y_test0[test_filter1]
        if CreateLBPrediction:
            X_lb1=X_lb0[lb_filter1]
        ## update global filter
        train_filter[train_filter]=train_filter1
        test_filter[test_filter]=test_filter1
        if CreateLBPrediction:
            lb_filter[lb_filter]=lb_filter1
        ## to go
        if CreateLBPrediction:
            run_level(X_train1, X_test1, y_train1, y_test1, X_lb0=X_lb1,
                      level=level+1, best_mcc=best_mcc)            
        else:
            run_level(X_train1, X_test1, y_train1, y_test1,
                      level=level+1, best_mcc=best_mcc)
        
if CreateLBPrediction:
    run_level(X_train, X_test, y_train, y_test, X_lb)
else:
    run_level(X_train, X_test, y_train, y_test)
    
if UseAllFeature and WriteFeatureFilt:
    np.save('../data/feature_filter.npy', feature_filter)
