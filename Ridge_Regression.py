#!/usr/bin/env python
# coding: utf-8

# # Ridge Regression
# 
# This tutorial builds upon the work performed in the previous tutorial "*Least_Squares_Regression.ipynb*". The data used will be the same
# 
# ### Learning Objectives
# 
# - Test both Python and mathematical competencies in ridge regression
# - Calculate ridge regression weights using linear algebra
# - Understand how to standardize data and its working 
# - Process data for regularized methods 
# - Implement ridge regression from scratch
# - Familiarize with the concept of hyperparameter tuning

# Before coding an algorithm, we will take a look at our data using Python's `pandas`. For visualizations, we'll use `matplotlib`.
# Let's import the necessary libraries and load the datasets by using the pandas pd.read_csv() function.

# In[1]:


### Import the necessary modules

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)  

### Read in the data
tr_path = 'data/train.csv'

data = pd.read_csv(tr_path)


# We begin by performing some basic exploratory data analysis by using the function head() and the attribute columns.
# 

# In[2]:


data.head()


# Next, let's plot the relationship between our variables of interest: the price for each house and the above ground living area in square feet.
# 
# We can do so by creating a scatter plot using matplotlib.

# In[3]:


data.plot('GrLivArea', 'SalePrice', kind = 'scatter', marker = 'x');


# ### Coding Ridge Regression   
# 
# #### Preprocessing
# Before implementing ridge regression, it is important to mean-center our target variable and mean-center and standardize observations. We will do this by using the following formulas:  
# #### Mean Center Target
# $$y_{cent} = y_0 - \bar{y}$$
# 
# #### Standardize Observations
# $$X_{std} = \frac{X_0-\bar{X}}{s_{X}}$$
# 
# Where $\bar{X}$ is the sample mean of X and $s_{X}$ is the sample standard deviation of X. 

# We will create a function called "standardize" that accepts, as input a list of numbers and returns a list where those values have been standardized.

# In[6]:


def standardize( num_list):
    
    output = []
    x_mean = np.mean(num_list)
    x_std = np.std(num_list)
    for x in num_list:
        x_standard = (x - x_mean) / x_std
        output.append(x_standard)
    return output


# Below we will create a function which will preprocess our data by performing:
# * mean subtraction from $y$,
# * dimension standardization for $x$.
# 
# The formulas to Mean Center Target and Standardize Observations are given above.

# In[7]:


def preprocess_for_regularization(data, y_column_name, x_column_names):
    
    df = data.copy()
    df = df.loc[:, x_column_names + [y_column_name]]
    
    for x in x_column_names:
        df[x] = standardize(df[x])
        
    y_mean = df[y_column_name].mean()
    df[y_column_name] = df[y_column_name].apply(lambda z: z - y_mean)
    return df


# Then we will test our function

# In[8]:


data = pd.read_csv(tr_path).head()
prepro_data = preprocess_for_regularization(data,'SalePrice', ['GrLivArea','YearBuilt'])

print(prepro_data)


# Next, we will implement the equation for ridge regression using the closed form equation:  
# 
# $$w_{RR}=(\lambda+X^TX)^{-1}X^Ty$$

# We will create a  function called "ridge_regression_weights" that takes, as input, three inputs: two matricies corresponding to the x inputs and y target and a number (int or float) for the lambda parameter
# 
# The function should return a numpy array of regression weights
# 
#  The following steps must be accomplished:
# 
# Ensure the number of rows of each the X matrix is greater than the number of columns.
# If not, transpose the matrix.
# Ultimately, the y input will have length n. Thus the x input should be in the shape n-by-p
# 
# *Prepend* an n-by-1 column of ones to the input_x matrix
# 
# We will use the above equation to calculate the least squares weights. This will involve creating the lambda matrix - a p+1-by-p+1 matrix with the "lambda_param" on the diagonal

# In[9]:


def ridge_regression_weights(input_x, output_y, lambda_param):
    
    if input_x.shape[0] < input_x.shape[1]:
        input_x = input_x.T
          
    ones_matrix = np.ones((input_x.shape[0], 1), dtype=int)
    input_x = np.concatenate((ones_matrix, input_x), axis=1)
    lambda_matrix = lambda_param * np.identity(min(input_x.shape))
    
    weights = np.matmul(np.linalg.inv(lambda_matrix + np.matmul(input_x.T, input_x)), np.matmul(input_x.T, output_y))
   
    return weights


# ### Selecting the $\lambda$ parameter
# 
# For our final function before looking at the `sklearn` implementation of ridge regression, we will create a hyperparameter tuning algorithm.
# 
# In ridge regression, we must pick a value for $\lambda$. We have some intuition about $\lambda$ from the equations that define it: small values tend to emulate the results from Least Squares, while large values will reduce the dimensionality of the problem.
# 
# For this tutorial, we will solve a simple problem on finding a value that minimizes the list returned by the function.

# In[10]:


### `hidden` takes a single number as a parameter (int or float) and returns a list of 1000 numbers
### the input must be between 0 and 50 exclusive

def hidden(hp):
    if (hp<=0) or (hp >= 50):
        print("input out of bounds")
    
    nums = np.logspace(0,5,num = 1000)
    vals = nums** 43.123985172351235134687934
    
    user_vals = nums** hp
    
    return vals-user_vals


# In[11]:


hidden(10)


# The below function will be similar to `hidden` created above. Like 'hidden', the passed function will take a single argument, a number between 0 and 50 exclusive and then, the function will return a numpy array of 1000 numbers.
# 
# Your function should return the value that makes the mean of the array returned by 'passed_func' as close to 0 as possible.

# In[12]:


def minimize( passed_func):
    # Create values to test
    test_vals = list(np.linspace(.1,49.9, 1000))
    
    # Find mean of returned array from function
    ret_vals = [abs(np.mean(passed_func(x))) for x in test_vals]
    
    # Find smallest mean
    min_mean = min(ret_vals)
    
    # Return the test value that creates the smallest mean
    return test_vals[ret_vals.index(min_mean)] 
    


# In the case of ridge regression, you would be searching lambda parameters to minimize the validation error.
# 
# See below for an example of using the functions built above that automatically perform hyperparameter tuning using mean absolute deviation.

# In[15]:


def lambda_search_func(lambda_param):
    
    # Define X and y
    # with preprocessing
    df = preprocess_for_regularization(data.head(50),'SalePrice', ['GrLivArea','YearBuilt'])
    
    y_true = df['SalePrice'].values
    X = df[['GrLivArea','YearBuilt']].values
    
    # Calculate Weights then use for predictions
    weights = ridge_regression_weights(X, y_true, lambda_param )
    y_pred = weights[0] + np.matmul(X,weights[1:])
    
    # Calculate Residuals
    resid = y_true - y_pred
    
    # take absolute value to tune on mean-absolute-deviation
    # Alternatively, could use:
    # return resid **2-S
    # for tuning on mean-squared-error
    
    return abs(resid)


# In[14]:


minimize(lambda_search_func)


# ### Ridge Regression in `sklearn` 
# 
# In the below code, I will show you how to implement Ridge regression in `sklearn`.
# 
# We will use the function `LinearRegression` from `sklearn` to instantiate the classifier `lr`.
# We will use the function `Ridge` from `sklear` to instantiate the classifier `reg`. For this classifier, set the parameter `alpha=100000`. Use the `Ridge` function to instantiate another classifier, `reg0`, but, this time, set `alpha=0`.
# 
# **NOTE: Note, the "alpha" parameter defines regularization strength. Lambda is a reserved word in `Python` -- Thus "alpha" instead**

# In[17]:


from sklearn.linear_model import Ridge, LinearRegression

lr = LinearRegression()
reg = Ridge(alpha=100000)
reg0 = Ridge(alpha=0)
X = data[['GrLivArea', 'YearBuilt']]
y = data['SalePrice']

for m, name in zip([lr, reg, reg0], ["LeastSquares","Ridge alpha = 100000","Ridge, alpha = 0"]):
    m.fit(X,y)
    print(name, "Intercept:", m.intercept_, "Coefs:",m.coef_,"\n")


# In[ ]:




