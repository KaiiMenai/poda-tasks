# Principles of Data Analytics Tasks

by Kyra Menai Hamilton (PhD)

## Setup

Sign up for a free GitHub Account.
Go to the Repository page in GitHub through the browser.
Click on the GREEN Code button.
Click on the Codespaces tab.
Click Create New Codespace on main.
Laugh at the random generated Codespace name.

## Technologies

1. Python
2. Git
3. GitHub
4. Codespaces
5. Jupyter

## Notes and Tips

To save changes made to the README file:

- Click on Source Control option on the left.
- In Message box, enter what the changes were i.e. changed README.
- Click on the Commit button and this will save changes to the project, this will save the state of the project as it is now.

To sync the changes with GitHub:

- Following saving changes to the README file, through the Source Control and Commit functions, there should now be a Sync Changes button.
- Click on the Sync Changes button, this will then sync the committed changes from the development environment (github.dev) with GitHub.

IF Codespaces throws 403 ERROR when attempting COMMIT:

- Check login details.
- Check Access and Permissions.
- Honestly just easier to make a new repository and copy everything over.

*** SPLIT - put the appropriate model/whatsit for the analysis type under each task header to make it easier for reader.

## Summary of Content in Jupyter Notebook

### Libraries for the tasks

(Python. 2025. Built-in types. https://docs.python.org/3/library/stdtypes.html)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### Task 1: Source the Data Set

Importing the Iris dataset from https://gist.github.com/curran/a08a1080b88344b0c8a7 using https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html to make Python be able to read the csv easier.#

### Task 2: Explore the Data Structure

Dataset Loaded in.
150 samples, across 5 variables including 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'.
First and last 5 rows were printed using: sliced_df = pd.concat([df.head(5), df.tail(5)])

### Task 3

### Task 4

### Task 5

### Task 6

### Task 7

### Task 8

corr_matrix = df.iloc[:, :4].corr()
    -  where df.iloc[:, :4]: selects the first four columns of the df (dataframe), these were assumed to be numerical features (sepal length, sepal width, petal length, and petal width).
    - in this : selects all rows, and :4 selects columns from index 0 to 3 (exclusive).

.corr(): - this calculates the correlation matrix for the selected columns. The correlation matrix shows the pairwise correlation coefficients between the features, with values ranging from:

-  1: Perfect positive correlation.
-  0: No correlation.
- -1: Perfect negative correlation.

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm') - where sns.heatmap(corr_matrix, ...): is what creates a heatmap visualisation of the correlation matrix using Seaborn, annot=True: shows the correlation values (numerical) inside each cell of the heatmap, and cmap='coolwarm': Specifies the colour map for the heatmap - Cool colours (blues) represent negative correlations, and warm colours (reds) represent positive correlations.

### Task 9

The packages used to conduct the Logistic Regression and get R<sup>2<sup> were:

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

### Task 10

For the Pairwise plot, the pairplot from the package seaborn was used:
import seaborn as sns
pairplot = sns.pairplot(dataframe, hue='class', height=desired height for plot) - where dataframe is the data to be plotted, class is the categorical variable, and height is the desired height of the subplot.

## References
