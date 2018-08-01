import datetime
import csv
import itertools
import operator
import numpy as np
import datetime
import scipy as sp
import xgboost as xgb
import os
from sklearn import cross_validation
from matplotlib import pylab as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
import data_process
#for test
import pandas as pd

def GetWeight(y):
	w = np.zeros(y.shape, dtype=float)
	ind = y!= 0
	w[ind] = 1.0/ (y[ind]**2)
	return w
def rmspe_xgb(y_pred, y):
	y = y.get_label()
	y = np.expm1(y)
	y_pred = np.expm1(y_pred)
	w = GetWeight(y)
	rmspe = np.sqrt(np.mean(w * (y - y_pred)**2))
	return "rmspe_xgb", rmspe

def rmspe(y_pred, y):
	w = GetWeight(y)
	rmspe = np.sqrt(np.mean(w * (y - y_pred)**2))
	return rmspe

def train_xgb(train, features, features_non_numeric):
	
	depth = 13
	eta = 0.01
	ntrees = 3000
	mcw = 3
	params = {"objective": "reg:linear",
		  "booster": "gbtree",
		  "eta": eta,
		  "max_depth": depth,
		  "min_child_weight": mcw,
		  "subsample": 0.9,
		  "colsample_bytree": 0.7,
		  "silent": 1
		  }
	print "Params: ", params
	print "ntrees: ", ntrees
	print "Features: ", features
	
	#Train with local split
	tsize = 0.35
	X_train, X_test = cross_validation.train_test_split(train, test_size=tsize)
	print "Complete Data split!"
	dtrain = xgb.DMatrix(X_train[features], np.log1p(X_train['Sales']))
	print "Complete Transfer training data to DMatrix..."
	dvalid = xgb.DMatrix(X_test[features], np.log1p(X_test['Sales']))
	print "Complete Transfer test data to DMatrix..."
	watchlist = [(dvalid, 'dvalid'), (dtrain, 'dtrain')]
	print "Start Training..."
	xgb_train_md = xgb.train(params, dtrain, ntrees, evals=watchlist, early_stopping_rounds=100, feval=rmspe_xgb, verbose_eval=True)
	"Save model..."
	xgb_train_md.save_model('xgb_train_md.model')

def test_xgb(test, features):
	
	gbm = xgb.Booster({'nthread':4})
	print "load xgb training model..."
	gbm.load_model("xgb_train_md.model")
	print "make prediction..."
	prediction = gbm.predict(xgb.DMatrix(test[features]))
	indices = prediction < 0
	prediction[indices] = 0
	submission = pd.DataFrame({'Id': test['Id'], 'Sales': np.expm1(prediction)})
	print submission.head()
	if not os.path.exists('result/'):
		os.makedirs('result/')
	print "save to submission.csv..."
	submission.to_csv("./result/submission.csv", index=False)
	
'''			
if __name__ == "__main__":
	print "loading raw data..."
	train, test, features, features_non_numeric = data_process.read_data()
	print "clean data and do feature engineering..."
	train, test, features, features_non_numeric = data_process.process_data(train, test, features, features_non_numeric)
	print "xgb train..."
	#train_xgb(train, features, features_non_numeric)
	#print "Complete Training..."
	print "xgb test..."		
	test_xgb(test, features)
	print "Complete XGB test..."
'''
