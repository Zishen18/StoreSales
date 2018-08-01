import xgboost as xgb
from matplotlib import pylab as plt
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import data_process
from sklearn import cross_validation


def create_feature_map(features):
	
	outfile = open('xgb.fmap', 'w')
	for i, feat in enumerate(features):
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
	outfile.close()


def importance_feat(features):
	print "loading xgb train model..."	
	xgbm = xgb.Booster({'nthread':4})
	xgbm.load_model("xgb_train_md.model")
	print "create feature map..."
	create_feature_map(features)
	print "get feature importance..."
	importance = xgbm.get_fscore(fmap='xgb.fmap')
	print "importance: "
	print importance
	print "sort importance..."
	importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
	print "importance: "
	print importance 
	
	df = pd.DataFrame(importance, columns=['feature', 'fscore'])
	df['fscore'] = df['fscore'] / df['fscore'].sum()
	print df.head()
	print "plot feature importance..."
	#featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6,10))
	featp = df.plot(kind='bar', x='feature', y='fscore', legend=True, figsize=(10, 6))
	plt.title('XGBoost Feature Importance')
	#plt.xlabel('relative importance')
	fig_featp = featp.get_figure()
	fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)

def Correlation_Matrix_plot(data):
	features = data.columns.tolist()
	correlations = data.corr()
	print "Plot correlation matrix..."
	fig = plt.figure(figsize=(12,12))
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	num_feat = len(features)
	ticks = np.arange(0,num_feat,1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(features, rotation=90)
	ax.set_yticklabels(features)
	fig.savefig('correlation_matrix.png') 
	print "Correlation Maxtrix Figure saved!"

def Scatter_plot(data):
	features = data.columns.tolist()
	xname = features[0]
	yname = 'Sales'
	condition = data[features[2]].tolist()
	size = len(condition)
	color = ['red' if condition[i] == 0 else 'green' for i in range(size)]
	print "color size: ", len(color)

	x = data[xname].tolist()
	y = data['Sales'].tolist()
	print "x size: ", len(x)
	print "y size: ", len(y)
	plt.scatter(x, y, c=color)
	plt.xlabel(xname)
	plt.ylabel(yname)
	# add legend
	classes = ['0', '1']
	class_colours = ['r', 'g']
	recs = []
	for i in range(len(class_colours)):
		recs.append(mpatches.Rectangle((0,0), 1, 1, fc=class_colours[i]))
	plt.legend(recs, classes, loc='upper left')
	plt.show()
	

train, test, features, features_non_numeric = data_process.read_data()

train, test, features, features_non_numeric = data_process.process_data(train, test, features, features_non_numeric)

tsize = 0.001
dtrain, dtest = cross_validation.train_test_split(train, test_size=tsize)

#importance_feat(features)

#Correlation_Matrix_plot(train)

features = ['Customers', 'Sales','Promo']
data = dtest[features]

Scatter_plot(data)
