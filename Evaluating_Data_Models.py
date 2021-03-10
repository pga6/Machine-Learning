#!/usr/bin/env python
# coding: utf-8

# ## Evaluating Data Models
# 

# ### Learning Objectives:
# 
# * Become familiar with Scikit-Learn's "bunch" object datasets (load_boston, load_breast_cancer)
# 
# 
# * Identifying the baseline accuracy / error that naturally exists from the data set.
# 
# 
# * Using the Breast Cancer dataset (load_breast_cancer):
# 
#     - Prototype basic classification models (LogReg, KNN, DecTrees).
#     
#     - Determine if our models outperform the baseline accuracy (is this model better than just guessing the **majority class**?
#     
#     - Evaluate the best performing model based on classification metrics (Accuracy, Precision, Recall, F1-Score, ROC, AUC)
#     
#     
# * Using the Boston Housing dataset (load_boston):
# 
#     - Prototype basic regression models (LinReg, Ridge, LASSO)
#     
#     - Determine if our models outperform the baseline error (is this model better than just guessing the **mean** of the target variable?)
#     
#     - Evaluate the best performing model based on regression metrics (R^2 Accuracy, RMSE)

# ### Part I. Classification using the Breast Cancer Dataset

# In[1]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split


# In[3]:


pd.set_option("max.columns", 40)


# In[4]:


#Prepare Cancer Dataset
lbc = load_breast_cancer()

cancer_df = pd.DataFrame(data = lbc.data, columns = lbc.feature_names)
cancer_df['target'] = lbc.target
cancer_df.head()


# In[5]:


## View documentation for breast cancer data:
print(lbc.DESCR)


# In[6]:


print(lbc.target_names)


# In[7]:


cancer_df.target.value_counts()


# * From the documentation (.DESCR):
# 
#     > :Class Distribution: 212 - Malignant, 357 - Benign
#     
# * We know that Benign = 1, Malignant = 0
#     * In other words, "1" is non-cancerous, "0" is cancerous
#     * If we divide these numbers by the total number of samples (569), we can get the % of each class.
#         - We can use the keyword arguement "normalize = True" to do this!

# In[8]:


cancer_df.target.value_counts(normalize = True)


# #### Baseline for a Classification model:
# 
# * For a classification model, the "majority class" present in the target variable is the baseline accuracy.
# * From the above calculation we can conclude:
#     - Probability that the tumor is Malignant ("0") is ~37%
#     - Probability that the tumor is Benign ("1") is ~63%
#     
# * **Since the majority class is "Benign" (357 samples out of 569), the baseline accuracy is 62.7%**
#     - For a useful model, we need to beat 62.7%
#     - Otherwise, if we were to just guess the probability of the majority class for each sample, our predictions would be correct only ~62.7% of the time.
#     - A model that performs worse that the baseline is *worse than arbitrarily guessing the probability of the most common class in the data*.

# Note:  Typically, the convention for class labels are positives as "1's" and negatives as "0's."  This dataset is old and they use the "backwards" notation, but here we can explicitly define "postives" as "malignant" and "negatives" as "benign" as is the language used in medical testing.
# 
# ### Classification metrics:
# 
# ### $\\ Accuracy = \frac {True Positives + True Negatives}{N_{samples}}$
# 
# ### $\\ False Positive Rate = \frac {False Positives}{N_{negatives}}$
# 
# ### $\\ True Positive Rate = \frac {True Positives}{N_{positives}}$
# 
# ### $\\ Precision = \frac {True Positives}{TruePositives + FalsePositives}$
# 
# ### $\\ Recall = TPR =  \frac {True Positives}{FalseNegatives + TruePositives}$
# 
# ### $\\ F1-Score = 2 \frac {precision  *  recall}{precision + recall}$

# * Accuracy isn't everything!
#     - Often times wrong predictions matter *more* than total accuracy, depending on what is misclassified.
#     - Accuracy doesn't tell you how well a model can differentiate between false positives and false negatives.
#     
#     
# * When evaluating a classification model, mind what labels a model is mislabeling.
#     - For this cancer diagnosing, a false negative (diagnoses bengin when a tumor is actually cancerous/malignant) is a more costly result than a false positive (labeled as malignant than actually benign).
#     - A model than is more likely to wrongly diagnose a patient as being cancer free when they really do have cancer would be awful!
#     
#     
# * To adress these kinds of errors, we use the True Positve Rate, False Positive Rate, Precision, and Recall to quantify *how* a model is misclassifying samples.
# 
# 
# **True Positive Rate (Recall)**: % of positive predictions are correctly classified as positive.
# - (This tumor was diagnosed with as malignant, and really is malignant.  The patient has cancer and classified to have cancer.)
#         
#         
#         
# **False Positive Rate**: % of positive predictions that are incorrectly classified as negatives.
# - (This tumor was diagnosed as benign, but really is malginant.  The patient has cancer but diagnoses as cancer-free).
#         
#         
#         
# **Precision**: The "positive predictive power," the % of positive predictions compared to the total of number of all both false and true positives.
# - (If precision is low, then this model is poor at correctly classifying malignant tumors.

# In[9]:


cancer_df.corr()['target'].sort_values()[:-1]


# In[10]:


plt.figure(figsize=(8,8))
cancer_df.corr()['target'].sort_values()[:-1].plot(kind = 'barh')


# ### Let's start modeling with a few basic models
# 1. Logistic Regression 
# 2. K-Nearest Neighbors (KNN)
# 3. Decision Trees
# 

# In[11]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


# In[12]:


# Train Test Split
X = cancer_df.drop(columns = ['target'])
y = cancer_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# Declare Model Objects
logreg = LogisticRegression()
knn    = KNeighborsClassifier()
tree   = DecisionTreeClassifier()

# Fit training data

logreg.fit(X_train, y_train)
knn.fit(X_train, y_train)
tree.fit(X_train, y_train);


# In[13]:


def print_scores(model, model_name):
    print("{} R^2 Scores : ".format(model_name))
    print("Training Set : {:.2%}".format(model.score(X_train, y_train)))
    print("Testing Set  : {:.2%}\n".format(model.score(X_test, y_test)))
    
def print_report(model):
    print(classification_report(model.predict(X_test), y_test))

def plot_roc_curve(model):
    
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    model_auc = auc(fpr,tpr)
    
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1])
    plt.xlabel('FPR', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.ylabel('TPR', fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title('ROC Curve\nAUC : ' + str(model_auc.round(3)), fontsize = 18)
    plt.plot(fpr, tpr) 
    

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    ## From the Scikit-Learn Documentation
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def show_all_scores(model, model_name):
    print("------ Showing all results for {} --------\n".format(model_name))
    print(model, "\n")
    print_scores(model, model_name)
    print_report(model)
    plot_confusion_matrix(y_test, model.predict(X_test), classes = lbc.target_names, normalize = False)
    plot_roc_curve(model)


# In[14]:


show_all_scores(logreg, "LogReg")


# In[15]:


show_all_scores(knn, "KNN")


# In[16]:


show_all_scores(tree, "DecTree")


# ### Conclusions:
# 

# ### Part II. Regression using the Boston Housing Dataset

# In[17]:


#Prepare Cancer Dataset
boston = load_boston()

boston_df = pd.DataFrame(data = boston.data, columns = boston.feature_names)
boston_df['MEDV'] = boston.target
boston_df.head()


# In[18]:


# View documentation for boston data:
print(boston.DESCR)


# * From the documentation (.DESCR):
# 
#     > MEDV     Median value of owner-occupied homes in $1000's
#     
# * This target variable for this dataset is the value of each home, "MEDV."
# * This is a continuous variable, expressed in units of thousands.
# * Note: This data came from a study published in 1978 and not adjusted for inflation, so the overall prices of these homes will look lower than they would in 2019 Boston, MA.

# In[19]:


boston_df['MEDV'].hist(bins = 30)


# In[20]:


boston_df['MEDV'].describe()


# <a id="evaluation-metrics-for-regression-problems"></a>
# ### Regression metrics:
# 
# **Mean Squared Error (MSE)**:
# 
# ### $\frac {1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2$
# 
# **Root Mean Squared error (RMSE)**:
# 
# ### $\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$
# 
# **R^2 (Coefficient of Determination)**:
# 
# ### $\\ R^2 = 1 - \frac {Sum of Square Errors}{Total Sum of Squares}$
# 
# 
# ### $\\ R^2 = 1 - \frac {\sum_{i=1}^n(y_i-\hat{y}_i)^2}{\sum_{i=1}^n(y_i-\bar{y})^2}$
# 
# 
# ### $\\ R^2 = 1 - \frac {MSE}{Var(y)}$
# 
# 
# * R^2 Accuracy is a _normalized_ view of the mean squared error.
#     - Mean Squared Error is the sum of all of the errors for each prediction made.
#     - OK to use R^2 for general model training (gauging overall performance, checking for overfitting), but not interpretable for a continuous value!

# * We'll be using RMSE mostly, since it's the closest in scale and units of the target variable.
#     - Interpret as "for every prediction I make, I can expect the value to be this amount over or under the real value."
#     - Somewhat of a relative value, best for comparing models between each other.
#     - The closer to 0, the better.
#     - Recall, it incorporates the sum of *all of the squared errors* for the predictions (the red lines in the figure above, which are the distinaces between the actual y values from the predicted y values (y-hat).

# #### Baseline for a Regression model:
# 
# * For a regression model, the error from using the mean of the target variable as your prediction for every value is the baseline accuracy.
#     - This is equivalent to just guessing the average home price for every data point as your prediction.
#     - If a regression model can produce a lower RMSE than the baseline RMSE, then we can conclude it is better than the baseline (better than guessing the mean).
#    

# In[21]:


plt.figure(figsize=(6,6))
boston_df.corr()['MEDV'].sort_values()[:-1].plot(kind = 'barh')


# In[22]:


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error


# In[23]:


# Train Test Split
X = boston_df.drop(columns = ['MEDV'])
y = boston_df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# Declare Model Objects

linreg = LinearRegression()
lasso  = Lasso()
ridge  = Ridge()

linreg.fit(X_train, y_train)
lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train);


# In[24]:


def print_reg_scores(model, model_name):
    print("------ Showing all results for {} --------\n".format(model_name))
    print(model, "\n")
    print("{} R^2 Scores  ".format(model_name))
    print("Training Set : {:.2%}".format(model.score(X_train, y_train)))
    print("Testing Set  : {:.2%}\n".format(model.score(X_test, y_test)))
    
    y_train_preds = model.predict(X_train)
    y_test_preds  = model.predict(X_test)
    
    y_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_preds))
    y_test_rmse  = np.sqrt(mean_squared_error(y_test, y_test_preds))
    
    print("{} RMSE ".format(model_name))
    print("Training Set : {:.3}".format(y_train_rmse))
    print("Testing Set  : {:.3}\n".format(y_test_rmse))


# In[25]:


mean_MEDV = boston_df['MEDV'].mean()


# In[26]:


mean_MEDV


# In[27]:


train_baseline_array = [mean_MEDV for i in range(0, y_train.shape[0])]
test_baseline_array  = [mean_MEDV for i in range(0, y_test.shape[0])]


# In[28]:


def calc_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[29]:


print("Baseline RMSE:")
print("Training Set : {:.3}".format(calc_rmse(y_train, train_baseline_array)))
print("Testing Set  : {:.3}".format(calc_rmse(y_test, test_baseline_array)))


# In[30]:


print_reg_scores(linreg, "LinReg")


# In[31]:


print_reg_scores(lasso, "LASSO")


# In[32]:


print_reg_scores(ridge, "Ridge")


# In[33]:


def plot_residuals(model, model_name):
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    plt.figure(figsize = (8,6))
    plt.scatter(y_train, y_train_pred,
                c='steelblue', marker = 'o', edgecolor = 'white', label = 'Training data')
    plt.scatter(y_test, y_test_pred,
                c='limegreen', marker = 'o', edgecolor = 'white', label = 'Test data')
    plt.plot([i for i in range(0, int(max(boston_df['MEDV'])))],
             [i for i in range(0, int(max(boston_df['MEDV'])))], c = 'black', marker ='.')
    plt.xlabel('Predicted Values (MEDV)', fontsize = 13)
    plt.ylabel('Test Values (MEDV)',fontsize = 13)
    plt.title("Residuals plot for {}".format(model_name), fontsize= 14)
    plt.legend(loc = 'upper left', fontsize = 13)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()


# In[34]:


plot_residuals(linreg, "LinReg")


# In[35]:


plot_residuals(lasso, "LASSO")


# In[36]:


plot_residuals(ridge, "Ridge")


# In[ ]:




