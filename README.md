# Kaggle-Competition
This is a Kaggle Competition completed by Chen Dong from Boston University

# Datasets
File descriptions:

train.csv - 1,697,533 unique reviews from Amazon Movie Reviews, with their associated star ratings and metadata. It is not necessary to use all reviews, or metadata for training. Some reviews will be missing a value in the 'Score' column. That is because, these are the scores you want to predict.

test.csv - Contains a table with 300,000 unique reviews. The format of the table has two columns; i) 'Id': contains an id that corresponds to a review in train.csv for which you predict a score ii) 'Score': the values for this column are missing since it will include the score predictions. You are required to predict the star ratings of these Id using the metadata in train.csv.

sample.csv - a sample submission file. The 'Id' field is populated with values from test.csv. Kaggle will only evaluate submission files in this exact same format.
Data fields:

ProductId - unique identifier for the product

UserId - unique identifier for the user

HelpfulnessNumerator - number of users who found the review helpful

HelpfulnessDenominator - number of users who indicated whether they found the review helpful

Score - rating between 1 and 5

Time - timestamp for the review

Summary - brief summary of the review

Text - text of the review

Id - a unique identifier associated with a review

# What we want to do?
The purpose is predict the missing values of a given dataset.

# How to start?
You should clone the repo and install python 3. Then, use the set-up file to check that you are ready to go. 

You can modify the codes that I provide. I use the gradient boosting decision tree to decrease the loss by 70 percent which is less than 1 now.

You can also use the confusion matrix to adjust your model accordingly.

# Reminder
The actual dataset which contains 1.7 millions of reviews are too large to be commited to Github (1.71G). So, I provide the test, sample, and the submission file which can provide you a view of the data. 

If you wish to obtain the full dataset, please email cdong27@bu.edu for the access.

After you get the dataset, please keep in mid the dataset is really, really large, and it takes me 10 hours to finish one feature extraction step. So, be patient and efficient. 
