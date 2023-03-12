import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from tensorflow import keras
from keras import layers

data = pd.read_csv('data1.csv')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


print("dtypes before EDA =", data.dtypes)
data.drop(data.index[(data["Sensor3"] == "-")], axis=0, inplace=True)

"Separate Target and variables"
X = data.copy()
y = X.pop('Target')


features_num = [
    "ComponentAge", "MonthlyRunTime",
    "FlowRate", "OPXVolume",
    "MaxOutputRate", "Sensor1", "Sensor2", "Sensor3",
    "Sensor4", "Sensor5",
    "Sensor5.1", "DaysSinceMaintenance",
]
features_cat = [
    "Location",
]

"Try meadian Value"
transformer_num = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

"Split test and train"
X_train, X_valid, y_train, y_valid = \
    train_test_split(X, y, stratify=y, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]

"Create model"
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])

#Add optimiser, loss and metic
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")


plt.show()
