# Principles of Data Analytics Tasks

author: Kyra Menai Hamilton (PhD)

## Technologies

1. Python
2. Git
3. GitHub
4. Codespaces
5. Jupyter

## Summary of Content in Jupyter Notebook

### Background

The iris dataset has been extensively used as a training dataset, a learning dataset, and used for developing machine learning techniques. Originally sourced in 1936 by Fisher (1936), the iris dataset has been used multiple times, with a number of different iterations available online, some have differing (often noted as wrong) datapoints, as in the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/53/iris).

Fisher (1936) made the iris dataset wellknown following the use of the dataset in the paper titled "The use of multiple measurements in taxonomic problems". In the paper the multivariate iris dataset was used as an example of linear discriminant analysis, that is, a statistical method to find a linear combination of features that can either characterise or separate a number (two or more) of classes of objects or events from one another (https://en.wikipedia.org/wiki/Linear_discriminant_analysis, https://www.ibm.com/think/topics/linear-discriminant-analysis).

Anderson (1935) used the dataset for quantifying the morphological differences and variation between the iris species. An interesting point about the dataset is that collected two of the three species, here referring to the versicolor and virginica iris species samples, from the same pasture, on the same dat, and also measured at that time with the same piece of equipment (apparatus). This is an interesting point for the following analysis as often time the virginica and versicolor species seem to converge and not be as easy to separate as the setosa species (see task 4 histograms, task 5 scatter plots, etc.).

### Libraries for the tasks

(Python. 2025. Built-in types. https://docs.python.org/3/library/stdtypes.html)

```ruby
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
```

### Task 1: Source the Data Set

Importing the Iris dataset from https://gist.github.com/curran/a08a1080b88344b0c8a7 using https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html to make Python be able to read the csv easier.

### Task 2: Explore the Data Structure

Dataset Loaded in.
150 samples, across 5 variables including 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'.
First and last 5 rows were printed using: ```sliced_df = pd.concat([df.head(5), df.tail(5)])```

### Task 3

Summary statistics for the whole dataset and for each species was done using the ```df.describe()``` function. 
This was modified for each species so that the species could be separated from one another:

```ruby
setosa_stats = df[df['species'] == 'setosa'].describe()
versicolor_stats = df[df['species'] == 'versicolor'].describe()
virginica_stats = df[df['species'] == 'virginica'].describe()
```

Class distributions were also explored using ```df['species'].value_counts()``` to see if the data was evenly distributed between the three species. 50 samples were seen for each of the species, Setosa, Virginica, and Versicolor.

In order to get a full view of the differences between the species for each of the features a one way ANOVA was run for each of the features, separated by species using ```f_oneway```:
```ruby
setosa = df[df['species'] == 'setosa']['sepal_length']
versicolor = df[df['species'] == 'versicolor']['sepal_length']
virginica = df[df['species'] == 'virginica']['sepal_length']
```

Running the ANOVA in addition to the summary statistics gave an oversight as to if there were differences between the species for each of the features.

### Task 4

Histograms were plotted for each of the features. All plots were put into one "figure" to make the data easier to read and compare using ```fig, axes = plt.subplots(2, 2, figsize=(12, 10))```.
The histograms were plotted using ```sns.histplot(data=df, x="feature", hue="species", kde=False, ax=axes[0, 0], bins=15)```

- ```sns.histplot``` refers to the plot to be made,
- where ```data=df``` was the iris dataframe, 
- ```x="feature"``` where ```"feature"``` was sepal length/width or petal length/width,
- ```hue="species"``` would colour code the plot points by species, and
- ```ax=axes[0, 0]``` referred to the subplot.

### Task 5

Scatter plots were made for sepal length vs sepal width and petal length vs petal width, these were all plotted on one "figure" for ease of observation. This was done using seaborn: ```sns.scatterplot(ax=axes[0], data=df, x='feature1', y='feature2', hue='species', s=100)```

- where ```sns.scatterplot``` refers to the plot to be run,
- ```ax=axes[0]``` refers to the subplot,
- ```data=df``` was the iris dataframe,
- ```x="feature1"``` where ```"feature1"``` was sepal length or petal length,
- ```y="feature2"``` where ```"feature2"``` was sepal width or petal width, and
- ```hue="species"``` would colour code the plot points by species.

### Task 6

There were two ways to plot regression lines to the scatter plots from task 5 using ```seaborn``` or ```numpy```. For the purposes of completing the task initially a regression line was added to the plots from task 5 using the ```np.polyfit``` module from ```numpy.polyfit```.

Through using numpy, the regression line is manually calculated and added to the plots. This gives the user greater control and customisability of the regression lines.
```ruby
# Scatter plot with regression line for sepal length vs width
sns.scatterplot(ax=axes[0], data=df, x='feature1', y='feature2', hue='species', s=100)
# Custom function for regression line
for species in df['species'].unique():
    subset = df[df['species'] == species]
    # Fit a linear regression line using numpy.polyfit
    slope, intercept = np.polyfit(subset['feature1'], subset['feature2'], 1)
    # Generate regression line points
    x_vals = np.linspace(subset['feature1'].min(), subset['feature1'].max(), 100)
    y_vals = slope * x_vals + intercept
    # Plot the regression line
    axes[0].plot(x_vals, y_vals, label=f'{species} Regression')
```

- where ```sns.scatterplot``` refers to the plot to be run,
- ```ax=axes[0]``` refers to the subplot,
- ```data=df``` was the iris dataframe,
- ```x="feature1"``` where ```"feature1"``` was sepal length or petal length,
- ```y="feature2"``` where ```"feature2"``` was sepal width or petal width, and
- ```hue="species"``` would colour code the plot points by species.

The custom function  for plotting the regression line subsets the data by 'species' to give individual species regression lines.

Following the use of ```numpy``` a plot and regression lines by species was also plotted using ```sns.regplot``` from ```seaborn```. This automatically fits and plots the regression line, it is easier to use compared to numpy, but has less customisability.

Principal Component Analysis (PCS) was conducted on the data. This was used to see where most of the variability in the data came from. The data needed to be standardised for the analysis and this was done through a scaled transformation of the data:
```ruby
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The PCA showed that the majority of the variance within the data was due to the petal features (72.8 %) in PC1 (first principal components).

### Task 7

Boxplots were used to see the spread of the data and whether there were outliers. These were done for each of the features and colour coded for the species.

```ruby
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] # Define feature names and their corresponding titles
titles = ['Sepal Length by Species', 'Sepal Width by Species', 
          'Petal Length by Species', 'Petal Width by Species']

plt.figure(figsize=(12, 8))
for i, feature in enumerate(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):
    ax = plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, hue='species', data=df, ax=ax)
    ax.set_title(titles[i])
    ax.set_xlabel('Species'.title())  # Capitalize the first letter of each word
    ax.set_ylabel(feature.replace('_', ' ').title())  # Capitalize the first letter of each word
```

In the code:

- ```for i, feature in enumerate(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']):``` separates by feature and labels the figure appropriately in ```ax.set_title(titles[i])```,
- ```sns.boxplot``` states the plot to be conducted,
- ```x='species'``` the data will be separated by species,
- the ```y=feature``` is plotted against the species
- ```hue='species'``` the data will be colour coded by species,
- ```data=df``` is the iris dataset, and
- ```ax=ax``` refers to the subplot.

### Task 8

```ruby
corr_matrix = df.iloc[:, :4].corr() :
```

- where ```df.iloc[:, :4]:``` selects the first four columns of the df (dataframe), these were assumed to be numerical features (sepal length, sepal width, petal length, and petal width).
- in this code, ```:``` selects all rows, and ```:4``` selects columns from index 0 to 3 (exclusive).
- ```.corr():``` calculates the correlation matrix for the selected columns.

The correlation matrix shows the pairwise correlation coefficients between the features, with values ranging from:

-  1: Perfect positive correlation.
-  0: No correlation.
- -1: Perfect negative correlation.

```ruby
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm') :
```

- where ```sns.heatmap(corr_matrix, ...):``` is what creates a heatmap visualisation of the correlation matrix using Seaborn.
- ```annot=True:``` shows the correlation values (numerical) inside each cell of the heatmap.
- ```cmap='coolwarm':``` specifies the colour map for the heatmap.

In the plot,

- cool colours (**blues**) represent **negative** correlations, and
- warm colours (**reds**) represent **positive** correlations.

### Task 9

The packages used to conduct the Logistic Regression and get R<sup>2</sup> were:

```ruby
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
```

All plots (two subplots for sepal features and petal features) were placed into one "figure" using ```fig, axes = plt.subplots(1, 2, figsize=(20, 8))```.

```ruby
# Feature1 vs Feature2
X_feature = df[['feature1']]
y_feature = df['feature2']
model_feature = LinearRegression()
model_feature.fit(X_feature, y_feature)
y_feature_pred = model_feature.predict(X_feature)
r2_feature = r2_score(y_feature, y_feature_pred)

sns.scatterplot(ax=axes[0], data=df, x='feature1', y='feature2', hue='species', s=100)
sns.regplot(ax=axes[0], data=df, x='feature1', y='feature2', scatter=False, color='red')
axes[0].set_title('Feature1 vs Feature2 by Species')
axes[0].set_xlabel('Feature1 (cm)')
axes[0].set_ylabel('Feature2 (cm)')
axes[0].legend(title='Species')
axes[0].grid(True)
axes[0].text(0.05, 0.95, f'R² = {r2_feature:.2f}', transform=axes[0].transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
```



### Task 10

For the Pairwise plot, the pairplot from the package seaborn was used:

```ruby
import seaborn as sns
pairplot = sns.pairplot(dataframe, hue='class', height=desired height for plot) 
```

- Where ```dataframe``` is the data to be plotted,
- ```class``` is the categorical variable, and
- ```height``` is the desired height of the subplot.

## Conclusion

The size of the dataset is a limiting factor when it comes to model accuracy, reliability, and consistent repeatability. The data does well at demonstrating what a linear based dataset can show through various forms of analysis.

For example for a number of the plots (histograms, scatter plots, linear regression, box plots) the Setosa species clearly clusters separately to the Versicolor and Virginica species, for both sepal and petal features. For petal features all species separated from one another, indicating that petal features were more distinguishable between species. Following principal component analysis (PCA) this was solidified as it was found that PC1 (first principal component) was responsible for 72.8 % of the variability seen within the data, the PC1 was referring to the petal length and width features. The feature correlation matrix heat map also displayed this, where petal length vs petal width resulted in r = 0.96 indicating that the petal length and width showed strong positive correlation and that the size often varied together. This was further corroborated following the linear regression analysis where petal length vs petal width gave an R<sup>2</sup> value of 0.93, indicating that most of the variance is petal width can be explained by petal length. Some predictions were made for petal width following an 80 % - 20 % split in the data for training and testing, respectively. However, in order to have a mode reliable method for predicting the species using a linear regression (or logistic regression) model, a larger sample population is essential in order to accurately visualise and calculate the nuances between such species based on their features.

## References

### Academic Sources

Anderson, E. (1935) 'The irises of the Gaspé peninsula', *Bulletin of the American Iris Society*, 59, pp. 2-5.

Cheeseman, P. et al. (1988) *AUTOCLASS II conceptual clustering system finds 3 classes in the data*, MLC Proceedings, pp. 54-64. Available at: https://cdn.aaai.org/AAAI/1988/AAAI88-108.pdf

Dasarathy, B.V. (1980) 'Nosing around the neighborhood: a new system structure and classification rule for recognition in partially exposed environments', *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-2(1), pp. 67-71. Available at: https://www.academia.edu/30910064/Nosing_Around_the_Neighborhood_A_New_System_Structure_and_Classification_Rule_for_Recognition_in_Partially_Exposed_Environments

Duda, R.O. and Hart, P.E. (1973) *Pattern Classification and Scene Analysis*. New York: John Wiley & Sons. Available at: https://www.semanticscholar.org/paper/Pattern-classification-and-scene-analysis-Duda-Hart/b07ce649d6f6eb636872527104b0209d3edc8188

Fisher, R.A. (1936) 'The use of multiple measurements in taxonomic problems', *Annual Eugenics*, 7(Part II), pp. 179-188. Available at: https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x

Fisher, R.A. (1950) *Contributions to Mathematical Statistics*. New York: Wiley & Co.

Gates, G.W. (1972) 'The reduced nearest neighbor rule', *IEEE Transactions on Information Theory*, 18(3), pp. 431-433. Available at: https://ieeexplore.ieee.org/document/1054809

Hamilton, K.M. (2022) *Drug resistance and susceptibility in sheep nematodes: fitness and the role of anthelmintic combinations in resistance management*. PhD Thesis. University College Dublin, Teagasc, and AgResearch.

James, G., Witten, D., Hastie, T. and Tibshirani, R. (2013) *An Introduction to Statistical Learning*. New York: Springer. Available at: https://link.springer.com/book/10.1007/978-1-0716-1418-1

---

### Information Sources (Non-Academic)

**Analytics Vidhya**  
(2020) 'Confusion matrix in machine learning'. Available at: https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/  
(2024) 'Pair plots in machine learning'. Available at: https://www.analyticsvidhya.com/blog/2024/02/pair-plots-in-machine-learning/

**Built In**  
(no date) 'Seaborn pairplot'. Available at: https://builtin.com/articles/seaborn-pairplot

**Bytemedirk**  
(no date) 'Mastering iris dataset analysis with Python'. Available at: https://bytemedirk.medium.com/mastering-iris-dataset-analysis-with-python-9e040a088ef4

**Datacamp**  
(no date) 'Simple linear regression tutorial'. Available at: https://www.datacamp.com/tutorial/simple-linear-regression

**Datatab**  
(no date) 'Linear regression tutorial'. Available at: https://datatab.net/tutorial/linear-regression

**GeeksforGeeks**  
(no date) 'Exploratory data analysis on iris dataset'. Available at: https://www.geeksforgeeks.org/exploratory-data-analysis-on-iris-dataset/  
(no date) 'How to show first/last n rows of a dataframe'. Available at: https://stackoverflow.com/questions/58260771/how-to-show-firstlast-n-rows-of-a-dataframe *(Note: Corrected URL to Stack Overflow)*  
(no date) 'Iris dataset'. Available at: https://www.geeksforgeeks.org/iris-dataset/  
(no date) 'Interpretations of histogram'. Available at: https://www.geeksforgeeks.org/interpretations-of-histogram/  
(no date) 'ML mathematical explanation of RMSE and R-squared error'. Available at: https://www.geeksforgeeks.org/ml-mathematical-explanation-of-rmse-and-r-squared-error/  
(no date) 'Python basics of pandas using iris dataset'. Available at: https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/

**How.dev**  
(no date) 'How to perform the ANOVA test in Python'. Available at: https://how.dev/answers/how-to-perform-the-anova-test-in-python

**IBM**  
(no date) 'Introduction to linear discriminant analysis'. Available at: https://www.ibm.com/think/topics/linear-discriminant-analysis  
(no date) 'Linear regression'. Available at: https://www.ibm.com/think/topics/linear-regression  
(no date) 'Logistic regression'. Available at: https://www.ibm.com/think/topics/logistic-regression

**Investopedia**  
(no date) 'R-squared'. Available at: https://www.investopedia.com/terms/r/r-squared.asp

**Kachiann**  
(no date) 'A beginners guide to machine learning with Python: Iris flower prediction'. Available at: https://medium.com/@kachiann/a-beginners-guide-to-machine-learning-with-python-iris-flower-prediction-61814e095268

**Kulkarni, M.**  
(no date) 'Heatmap analysis using Python seaborn and matplotlib'. Available at: https://medium.com/@kulkarni.madhwaraj/heatmap-analysis-using-python-seaborn-and-matplotlib-f6f5d7da2f64

**Medium**  
(no date) 'Exploratory data analysis of iris dataset'. Available at: https://medium.com/@nirajan.acharya777/exploratory-data-analysis-of-iris-dataset-9c0df76771df  
(no date) 'Pairplot visualization'. Available at: https://medium.com/analytics-vidhya/pairplot-visualization-16325cd725e6  
(no date) 'Regression model evaluation metrics'. Available at: https://medium.com/%40brandon93.w/regression-model-evaluation-metrics-r-squared-adjusted-r-squared-mse-rmse-and-mae-24dcc0e4cbd3

**Mizanur**  
(no date) 'Cleaning your data: handling missing and duplicate values'. Available at: https://mizanur.io/cleaning-your-data-handling-missing-and-duplicate-values/

**Newcastle University**  
(no date) 'Box and whisker plots'. Available at: https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/data-presentation/box-and-whisker-plots.html

**Nick McCullum**  
(no date) 'Python visualization: boxplot'. Available at: https://www.nickmccullum.com/python-visualization/boxplot/

**Numpy**  
(no date) 'numpy.polyfit'. Available at: https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

**Pandas**  
(no date) 'pandas.read_csv'. Available at: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

**Python Documentation**  
(no date) 'Built-in types'. Available at: https://docs.python.org/3/library/stdtypes.html

**ResearchGate**  
(no date) 'Classification of Iris Flower Dataset using Different Algorithms'. Available at: https://www.researchgate.net/publication/367220930_Classification_of_Iris_Flower_Dataset_using_Different_Algorithms

**RSS**  
(no date) 'Common statistical terms'. Available at: https://rss.org.uk/resources/statistical-explainers/common-statistical-terms/

**Scikit-learn**  
(no date) 'Classification report'. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html  
(no date) 'LabelEncoder'. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html  
(no date) 'LinearRegression'. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html  
(no date) 'LogisticRegression'. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html  
(no date) 'PCA example with iris dataset'. Available at: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html

**Seaborn**  
(no date) 'Pairplot'. Available at: https://seaborn.pydata.org/generated/seaborn.pairplot.html  
(no date) 'Regplot'. Available at: https://seaborn.pydata.org/generated/seaborn.regplot.html  
(no date) 'Scatterplot'. Available at: https://seaborn.pydata.org/generated/seaborn.scatterplot.html

**Slidescope**  
(no date) 'ANOVA example using Python pandas on iris dataset'. Available at: https://slidescope.com/anova-example-using-python-pandas-on-iris-dataset/#:~:text=We%20then%20convert%20the%20dataset,p-value%20for%20the%20test

**Stack Overflow**  
(no date) 'How to show first/last n rows of a dataframe'. Available at: https://stackoverflow.com/questions/58260771/how-to-show-firstlast-n-rows-of-a-dataframe

**Toxigon**  
(no date) 'Best practices for data cleaning and preprocessing'. Available at: https://toxigon.com/best-practices-for-data-cleaning-and-preprocessing  
(no date) 'Guide to data cleaning'. Available at: https://toxigon.com/guide-to-data-cleaning  
(no date) 'Introduction to seaborn for data visualization'. Available at: https://toxigon.com/introduction-to-seaborn-for-data-visualization  
(no date) 'Seaborn data visualization guide'. Available at: https://toxigon.com/seaborn-data-visualization-guide

**UCI Machine Learning Repository**  
(no date) 'Iris dataset'. Available at: https://archive.ics.uci.edu/dataset/53/iris

**Wikipedia**  
(no date) 'Linear discriminant analysis'. Available at: https://en.wikipedia.org/wiki/Linear_discriminant_analysis

**WV State University**  
(no date) 'Scholarly vs. non-scholarly articles'. Available at: https://wvstateu.libguides.com/c.php?g=813217&p=5816022

**Gist**  
(no date) 'Iris dataset CSV'. Available at: https://gist.githubusercontent.com/

# END