# Predicting likelihood of US government adopting contractionary fiscal policy with Kernel SVM

# Importing the libraries
import pandas as pd
import datapungi_fed as dpf

# Importing the dataset
data = dpf.data("API Key")
# Federal Debt Held by Foreign and International Investors as Percent of Gross Domestic Product
X1 = data.series('HBFIGDQ188S')/100 # quarterly
# Long term government bond yield: 10-year: Main (Including Benchmark)
X2 = data.series('IRLTLT01USQ156N').pct_change(fill_method ='ffill') #quarterly
# Trade Weighted U.S. Dollar Index: Broad, Goods and Services
X3 = ((1+data.series('DTWEXBGS').pct_change(fill_method ='ffill')).resample('Q').prod())-1 #convert daily time series to quarterly
X3.index = X3.index + pd.DateOffset(days=1)
# Real GDP growth rate
X4 = data.series('A191RL1Q225SBEA') #quarterly


# Real Government Consumption Expenditures and Gross Investment
Y = data.series('GCEC1').pct_change(fill_method ='ffill') # quarterly

# Preparing the dataframe
fulldf = pd.concat([X1,X2,X3,X4,Y],axis=1).dropna()

fulldf['foreign_debt_lag2'] = fulldf['HBFIGDQ188S'].shift(2)
fulldf['10_yr_gvt_bond_yield_lag1'] = fulldf['IRLTLT01USQ156N'].shift(1)
fulldf['exchange_rate_lag1'] = fulldf['DTWEXBGS'].shift(1)
fulldf['real_gdp_growth_lag1'] = fulldf['A191RL1Q225SBEA'].shift(1)

fulldf.loc[(fulldf.GCEC1 < 0), "gov_predict"] = 1
fulldf['gov_predict'] = fulldf['gov_predict'].fillna(0)
fulldf = fulldf.drop(['HBFIGDQ188S','IRLTLT01USQ156N','DTWEXBGS','A191RL1Q225SBEA','GCEC1'],axis=1).dropna()

print(fulldf)

X = fulldf.iloc[:, :-1].values
y = fulldf.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results with the latest
y_pred = classifier.predict(X_test)

# Compare predicted results with actual test results
print(y_pred)
print(y_test)
