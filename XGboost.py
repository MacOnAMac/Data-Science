import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

"Read in data"
train_data = pd.read_csv('house_train.csv', index_col='Id')
test_data = pd.read_csv('house_test.csv', index_col='Id')

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

"Do EDA on data to maximise good data usage "
# Remove rows with missing target and separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

#Split to test and train
X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)



"Preprocess training data"

#split data into numeric and cetegoric

catergoric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"] # Select categorical columns with low cardinality


numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)


# Keep selected columns only
my_cols = catergoric_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = test_data[my_cols].copy()

"Create pipeline"
numerical_transformer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, catergoric_cols)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', my_model)
                           ])
preprocessor.fit(X_valid)
X_valid_transformed = preprocessor.transform(X_valid)


pipeline.fit(X_train, y_train, model__early_stopping_rounds=5,
             model__eval_set=[(X_valid_transformed, y_valid)],
             model__verbose=False)


# Preprocessing of validation data, get predictions
preds = pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE_pipeline:', score)


plt.show()
