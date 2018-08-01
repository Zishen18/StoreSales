import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
import warnings
warnings.filterwarnings("ignore")

def read_data():
	# read raw data , split numeric and non-numeric features
	
	store = pd.read_csv('../data/store.csv')
	train = pd.read_csv('../data/train.csv')
	test = pd.read_csv('../data/test.csv')
	train = pd.merge(train, store, on='Store', how='left')
	test = pd.merge(test, store, on='Store', how='left')
	train['StateHoliday'] = train['StateHoliday'].astype('str')
	test['StateHoliday'] = test['StateHoliday'].astype('str')	
	features = test.columns.tolist()
	numeric_dtype = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	features_numeric = test.select_dtypes(include=numeric_dtype).columns.tolist()
	features_non_numeric = [feat for feat in features if feat not in features_numeric]
	return train, test, features, features_non_numeric	

def process_data(train, test, features, features_non_numeric):
	# Feature Engineering and Selection
	train = train[train['Sales'] > 0]
	
	for data in [train, test]:
		data['year'] = data.Date.apply(lambda x: x.split('-')[0])
		data['year'] = data['year'].astype(float)
		data['month'] = data.Date.apply(lambda x: x.split('-')[1])
                data['month'] = data['month'].astype(float)
		data['day'] = data.Date.apply(lambda x: x.split('-')[2])
                data['day'] = data['day'].astype(float)
		
		#promo interval "Jan, Apr, Jul, Oct"
		data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
		data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)	
		data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
		data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Apr" in x else 0)
		data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "May" in x else 0)
		data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jun" in x else 0)
		data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jul" in x else 0)
		data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Aug" in x else 0)
		data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Sep" in x else 0)
		data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Oct" in x else 0)
		data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Nov" in x else 0)
		data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Dec" in x else 0)

	##Feature Set.
	noisy_features = ['Id', 'Date']
	features = [c for c in features if c not in noisy_features]
	features_non_numeric = [c for c in features_non_numeric if c not in noisy_features]
		
	features.extend(['year', 'month', 'day'])
		
	#Fill NA
	class DataFrameImputer(TransformerMixin):

		def __init__(self):

			"""
			Impute missing values.
			Columns of dtype are imputed with the most frequent value in column.
			Columns of other type are imputed with mean of column
			 """

		def fit(self, X, y=None):
			self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], index=X.columns)
			return self

		def transform(self, X, y=None):
			return X.fillna(self.fill)

	print "Start Filling Missing data..."
	train = DataFrameImputer().fit_transform(train)
	test = DataFrameImputer().fit_transform(test)
	print "Complete filling missing data..."

	#Preprocess non-numeric data..."
	le = LabelEncoder()
	for feat in features_non_numeric:
		le.fit(list(train[feat]) + list(test[feat]))
		train[feat] = le.transform(train[feat])
		test[feat] = le.transform(test[feat])
	print "normalize numeric data..."
	scaler = StandardScaler()
	columns = set(features) - set(features_non_numeric) - set([])
	print columns
	'''
	print "Train Data head: "
	print train.head()
	print "Test Data Head: "
	print test.head()
	'''
	for col in columns:
		train_list = list(train[col])
		test_list = list(test[col])
		scaler.fit(train_list + test_list)
		train[col] = scaler.transform(train[col])
		test[col] = scaler.transform(test[col])
	return train, test, features, features_non_numeric
	
	
