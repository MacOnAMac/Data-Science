import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer

def get_score(n_estimators):
    """Return the average MAE over "x" CV folds of random forest model.

    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', RandomForestRegressor(n_estimators,
                                                                  random_state=0))
                                  ])

    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()




" Read in data "
train_data = pd.read_csv('house_train.csv', index_col='Id')
test_data = pd.read_csv('house_test.csv', index_col='Id')

"Do EDA on data to maximise good data usage "
# Remove rows with missing target and separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]


# Preprocessing for numerical and categorical data
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Select categorical columns with relatively low cardinality, look at data set cardinality to decide this number
categorical_cols = [cname for cname in train_data.columns if train_data[cname].nunique() < 10 and
                        train_data[cname].dtype == "object"]


# Keep selected columns only
my_cols = categorical_cols + numeric_cols
X = train_data[my_cols].copy()
X_test = test_data[my_cols].copy()

"apply preprocessing to numeric and categorical data"
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#calculate the scoare from various estimator sizes
results = {}
for i in range(1, 9):
    results[50 * i] = get_score(50 * i)

print("min test MAE =", min(results.values()))
plt.plot(list(results.keys()), list(results.values())) #Plot MEA from cross val as a funciton of estimators

#Min val is 300, use for "optimised piepline"

optimised_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', RandomForestRegressor(300,
                                                                  random_state=1))
                                  ])


"Run final model on the test data"
optimised_pipeline.fit(X, y)
predictions = optimised_pipeline.predict(X_test)


plt.show()

