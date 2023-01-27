
"Import Python modules used for plotting, manipulating data and cleaning dataframe"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

"Define functions to use "


def remove_outliers(df, columns, n_std):
    # Taken from https://stephenallwright.com/remove-outliers-pandas/
    for col in columns:
        # print('Working on column: {}'.format(col)) #Useful for printing where outliers brake down if data set has not been cleaned (MAC)

        mean = df[col].mean()
        sd = df[col].std()

        df = df[(df[col] <= mean + (n_std * sd))]
    return df


def data_set_plot(title):
    fig, axes = plt.subplots(2, 6)
    fig.suptitle(title, fontsize=14)
    sns.histplot(data_set['ComponentAge'], color='blue', kde=True, ax=axes[0, 0])
    sns.histplot(data_set['MonthlyRunTime'], color='blue', kde=True, ax=axes[0, 1])
    sns.histplot(data_set['FlowRate'], color='blue', kde=True, ax=axes[0, 2])
    sns.histplot(data_set['MaxOutputRate'], color='blue', kde=True, ax=axes[0, 3])
    sns.histplot(data_set['Sensor1'], color='blue', kde=True, ax=axes[0, 4])
    sns.histplot(data_set['Sensor2'], color='blue', kde=True, ax=axes[0, 5])
    sns.histplot(data_set['Sensor3'], color='blue', kde=True, ax=axes[1, 0])
    sns.histplot(data_set['Sensor4'], color='blue', kde=True, ax=axes[1, 1])
    sns.histplot(data_set['Sensor5'], color='blue', kde=True, ax=axes[1, 2])
    sns.histplot(data_set['Sensor5.1'], color='blue', kde=True, ax=axes[1, 3])
    sns.histplot(data_set['DaysSinceMaintenance'], color='blue', kde=True, ax=axes[1, 4])


"================================================================================================"
"STEP 1 - Load data sets"
data_set = pd.read_csv('data1.csv')

# print(data_set) #Checks Data set is loaded

"Create columns for individual operations if needed and check dtypes"
columns = list(data_set)

print("Initial data types")
print(data_set.dtypes)  # Sensor 3 needs further investigation as dtype is object -
"================================================================================================"
"STEP 2 - Check data before exploratory analysis"

"Use pandas describe to get preliminary stats about data"
pd.DataFrame.describe(data_set)

"Explore data via plots and compare with stats"


data_set_plot(title="Variables before before EDA")
## NOTES ##
"Gaussian data: ComponentAge, FlowRate, Monthly RunTime, Sensor1, Sensor4, Sensor5, and DaysSince"
"NON Guassian data: Sensor 2&3, Max output, Sensor 5.1"
"Location not a numerical value, consider changing to A=0, B=1 as could be important, location could cause damage/faults"
"Location failures / Total at location; A - 1061/4898 = 21%, B - 217/1599 = 13%. Location A more likely"
"But not an very significantly more liley to fault at A"
"Component Age - Av of 0.5, but max of 2. Possible outlier"
"Runtime - Seems reasonable"
"Flow rate - All fine"
"OPX Volume -is not well documented - Moslty empty cells, omit from model for fear of skew"
"Max output - s.t.d is half of the mean, warrents a closer look for large values skewing data set like the max at 440 "
"Sensors - Sensor2, std too close to mean with large max, omit column for fear of skewing model OR treat via quantile"
"Days since maintenance - Large value possible skewing, but important feature if looking for failure like component age, consider clearing outliers"
"Some missing values, 6497 is max entries"

"Max misssing number of fields is ~10, ~0.2% of the data entries."

"================================================================================================"
"STEP 3 - Clean up data"

"Drop data columns that are unreliable or not useful for model"
columns_to_drop = ['OPXVolume', 'Location']  # Could create Location as a dummy variable but no value added?

data_set.drop(columns_to_drop, inplace=True, axis=1)

"Print to check colums have been dropped "
# print(data_set)
pd.DataFrame.describe(data_set)

"Look for non-sensicle input values like NaN"
data_set = data_set.dropna()  # One row has been dropped with NaN in
columns = list(data_set)  # Reset the columns as have removed some
data_set.drop(data_set.index[(data_set["Sensor3"] == "-")], axis=0, inplace=True)  # Remove invalid data entry

data_set["Sensor3"] = pd.to_numeric(data_set["Sensor3"],
                                    downcast="float")  # Removes str from column as flaged as error

print("Data types after screening")
print(data_set.dtypes)  # Check data type is correct and rows < rows before

"Remove rows with outliers as to not skew model learning as defined by values greater than 3 s.d. away from mean (assuming Gaussian behaviour)"
data_set = remove_outliers(data_set, columns, 3)  # Printed metrics again and deviations are smaller
# Sensor 5 is more Guassian if a percentile sweep is done

"Clean non Gaussian data by doing a percentile sweep, Check stats and graphs for confirmaiton on percentile value"
cols = ['Sensor2', 'Sensor3', 'MaxOutputRate', 'Sensor5.1']

Q1 = data_set[cols].quantile(0.25)  # Set lower quantile
Q3 = data_set[cols].quantile(0.75)  # Set upper quantile
IQR = Q3 - Q1  # Interquantile range

data_set = data_set[~((data_set[cols] < (Q1 - 1.5 * IQR)) | (data_set[cols] > (Q3 + 1.5 * IQR))).any(
    axis=1)]  # Remove values outside quantile for each column


data_set_plot(title="Variables before after EDA")

"All looks reasonable BAR Sensor 2, which doesn't still seems unreliable"
"could do a second percentile sweep, at the lower end of the spectrum creating a broad Guassian or remove"
"Decided to remove instead of over manipulating the data"
data_set.drop(['Sensor2'], inplace=True, axis=1)

"Now, Data should be ready for applying model"

"================================================================================================"
"STEP 4 - Binary claissification model"

"Count number of responses of each pass/fail"
data_set['Target'].value_counts()  # Reveals count of failures / target cases

X = data_set.loc[:, :]  # Sets a variable for all input data for model
Y = data_set.loc[:, 'Target']  # create a variable of target data only

"Create training and test data sets"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)  # splits test data to 25%

"Normalise results to protect against data leakage"

ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

"create model variable from imported sklearn"
model = LogisticRegression()

"Fit the model to the training data for ML"
fitted_train_data = model.fit(X_train, Y_train)

"Use the model to predict from the test data"
predictions = model.predict(X_test)

"================================================================================================"
"STEP 5: check model predictions - Easy to use the confusion matrix to compare the scenarios ie TN etc, and then calculate accuracy"

TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()

print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)

"Use these to calculate an accuracy score"
accuracy = (TP + TN) / (TP + FP + TN + FN)

print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))  # Returns 1 as expected since no FP or FN

"Can use the full classificaiton report as well"

classification_report(Y_test, predictions)

plt.show()

# Final notes: Long life parts are expected to fail, and those with long periods of without maintenance checks are issues as well
# Despite disregarding this data for the model this should be flagged
# perhaps and if >this age and >than this last checked raised as a potential hazard
