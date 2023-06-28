from madlan_data_prep import prepare_data
import numpy as np

# Clean the DataFrame using the clean_dataframe function
df = prepare_data('output_all_students_Train_v10.xlsx')

#############################Elastic Net Model#################################

##train data###

excluded_columns = [7]  # Index of columns to exclude

# Select all columns except those in the excluded_columns list
included_columns = [col for col in range(df.shape[1]) if col not in excluded_columns]

X =  df.iloc[:, included_columns].values
y =df.iloc[:,7].values

# Handle Null values in numerical columns
X_Numerical = df.iloc[:, [2,3,10,11,12,13,15,16,17,18,21,23,24]].values 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_Numerical)
X_Numerical = imputer.transform(X_Numerical)

X[:, [2,3,9,10,11,12,14,15,16,17,20,22,23]] = X_Numerical


# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#one hot encoder:
from sklearn.preprocessing import OneHotEncoder    
    
# Select the relevant categorical columns for one-hot encoding
X_train_cat = X_train[:, [0, 1, 13, 19, 24]]
X_test_cat = X_test[:, [0, 1, 13, 19, 24]]

# Initialize the encoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit and transform the categorical columns in the training data
X_train_encoded = encoder.fit_transform(X_train_cat)

# Transform the categorical columns in the testing data
X_test_encoded = encoder.transform(X_test_cat)

# Convert the encoded data to dense arrays
X_train_encoded = X_train_encoded.toarray()
X_test_encoded = X_test_encoded.toarray()


# Concatenate the encoded data with the numerical columns
X_train = np.concatenate((X_train[:, [2, 3, 9, 10, 11, 12, 14, 15, 16, 17, 20, 22, 23]], X_train_encoded), axis=1)
X_test = np.concatenate((X_test[:, [2, 3, 9, 10, 11, 12, 14, 15, 16, 17, 20, 22, 23]], X_test_encoded), axis=1)

#standartization:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[:,[0,1,10,11,12]] = scaler.fit_transform(X_train[:,[0,1,10,11,12]])
X_test[:,[0,1,10,11,12]] = scaler.transform(X_test[:,[0,1,10,11,12]])

########################elastic net###########################################
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV

# Create an instance of the Elastic Net model with cross-validation
elastic_net_cv = ElasticNetCV(
    alphas=[0.1, 0.5, 1.0, 10.0, 14.29999999999, 20.0],  # List of alpha values to consider
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.901],  # List of l1_ratio values to consider
    cv=10,  # Number of folds for cross-validation
)

# Train the model on the training data
elastic_net_cv.fit(X_train, y_train)

# Make predictions on the test set
y_pred = elastic_net_cv.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Retrieve the best alpha and l1_ratio values from cross-validation
best_alpha = elastic_net_cv.alpha_
best_l1_ratio = elastic_net_cv.l1_ratio_
print("Best alpha:", best_alpha)
print("Best l1_ratio:", best_l1_ratio)

import pickle
pickle.dump(elastic_net_cv, open("trained_model.pkl","wb"))
pickle.dump(X, open("X.pkl","wb"))