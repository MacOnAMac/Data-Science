# Data-Science-
Some tests for Data modelling skills and exploring data handling



################### Titanic model ##################
The titanic model takes in survival data from the Titanc ship crash
Inputs:
  - Passenger Id
  - Name
  - Ticket
  - Fare
  - Passenger Class
  - SibSp (number of siblings or spouses onboard)
  - Parch (number of parents of chidlren onboard)
  - Age 
  - Sex
  - Cabin 
 Target = Survived 


Initial EDA is done, and a pipeline is used to impute the missing data within limits (ie not too many high cardinality variables or mostly missing columns)
There is a nueral network model (Takes ~30m to run due to grid search optimisation being dense) which is overkill for this simple dataset. 
Theres also a much quicker random forest model abailable.



################### Mechanical fluid pump (data1.csv) ##################
Neural Network gives an example of using a neural network on a set of data monitoring faliure of mechanical parts as the target for a binary clsasifier. 
