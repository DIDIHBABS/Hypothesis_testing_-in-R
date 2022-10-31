#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats.stats import pearsonr
import scipy.stats


# In[2]:


#Dataset= pd.read_excel("D:\Downloads\WIID_31MAY2021_0 (1).xlsx") 
Dataset=pd.read_excel("D:\Downloads\Big Data and ML\wiidcountry.xlsx")


# In[3]:


Dataset.head(2)


# In[4]:


Dataset2 = Dataset[["year","country","gini","population","gini_std", "incomegroup"]]


# In[5]:


Dataset2.describe()


# In[31]:


Dataset2.describe()


# In[6]:


Dataset2['logarithm_base10'] = np.log10(Dataset2['population'])
Dataset2.head(5)


# In[9]:


Dataset2.info()


# In[12]:


Dataset2.drop_duplicates()


# In[10]:


Dataset2 =Dataset2.dropna()


# In[11]:


Dataset2.isnull().sum()


# In[7]:


Dataset2.head(2)


# In[ ]:





# # Visualization 

# In[13]:


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = [20, 10]
plt.scatter(Dataset2['population'], Dataset2['gini_std'], s = 60) # 's' for 's'ize of data points.
plt.ylabel('Gini Index', fontsize = 14)
plt.xlabel('Population Size', fontsize = 14)
plt.title('Gini Index vs. Population', fontsize = 20)


# In[20]:


sns.regplot(Dataset2['population'], Dataset2['gini_std'], ci = None)


# In[14]:


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = [20, 10]
plt.scatter(Dataset2['logarithm_base10'], Dataset2['gini_std'], s = 60) # 's' for 's'ize of data points.
plt.ylabel('Gini Index', fontsize = 14)
plt.xlabel('Population Size', fontsize = 14)
plt.title('Gini Index vs. Natural Log of Population', fontsize = 20)


# In[30]:


sns.regplot(Dataset2['logarithm_base10'], Dataset2['gini_std'], ci = None)
plt.ylabel('Gini Index', fontsize = 14)
plt.xlabel('Population Size', fontsize = 14)
plt.title('Gini Index vs. Natural Log of Population', fontsize = 20)


# In[ ]:


sns.regplot(df['BA'], df['W'], ci = None)
plt.ylabel('Number of Wins', fontsize = 14)
plt.xlabel('Batting Average', fontsize = 14)
plt.title('Number of Wins vs. Batting Average', fontsize = 20)
plt.ylabel('Gini Index', fontsize = 14)
plt.xlabel('Population Size', fontsize = 14)
plt.title('Gini Index vs. Population', fontsize = 20)


# In[ ]:





# # Hypothesis test one . 

# In[16]:


r, p = scipy.stats.pearsonr(Dataset2["population"], Dataset2[ "gini_std"])
r,p


# In[17]:


print("r =", round(r, 3))
if p < .001:
    print("p <.001")
elif p <.01:
    print("p <.01")
elif p <.05:
    print("p <.05")
else:
    print("Not significant")


# # # Natural Log Value 

# In[16]:


rho, pho = scipy.stats.pearsonr(Dataset2["logarithm_base10"], Dataset2[ "gini_std"])
rho,pho


# In[18]:


print("rho =", round(rho, 3))
if pho < .001:
    print("pho <.001")
elif pho <.01:
    print("pho <.01")
elif pho <.05:
    print("pho <.05")
else:
    print("Not significant")


# In[21]:


Dataset2.shape


# In[12]:


import statsmodels.api as sm


# In[14]:




x =Dataset2[["population", "logarithm_base10"]]
# .corr is a pd function.
x.corr(method = 'pearson')


# In[15]:


y = Dataset2["gini_std"] # DV
x2 = Dataset2[["population", "logarithm_base10"]] # BA & RA as IVs
# Add a constant to an IV
x2 = sm.add_constant(x2)
# Run the model with y & x2
mdl2 = sm.OLS(y, x2).fit()
print(mdl2.summary())


# In[ ]:





# # Further Analysis Hypothesis 1 

# In[15]:


import statsmodels.api as sm


# In[43]:


y= Dataset2["gini_std"]
x= Dataset2["logarithm_base10"]
x=sm.add_constant(x)


# In[44]:


mdl = sm.OLS(y, x).fit()
print(mdl.summary())


# In[ ]:





# In[45]:


y= Dataset2["gini_std"]
x= Dataset2["population"]
x=sm.add_constant(x)


# In[46]:


mdl = sm.OLS(y, x).fit()
print(mdl.summary())


# In[25]:


print(mdl.rsquared_adj)


# In[114]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(x, y, train_size = 0.8, random_state = 42)
# Linear regression model 
model = sm.OLS(train_y, train_X)
model = model.fit()
print(model.summary2())


# In[115]:


predictions = model.predict(test_X)


# In[116]:


df_results = pd.DataFrame({'Actual': test_y, 'Predicted': predictions})


# In[117]:


#Plot the actual vs predicted results
sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False)
#Plot the diagonal line
d_line= np.arange(df_results.min().min(), df_results.max().max())
plt.plot(d_line, d_line, color='red', linestyle='--')
plt.show()


# In[123]:


from statsmodels.graphics.gofplots import qqplot
fig=qqplot(model.resid_pearson,line='45',fit='True')
plt.xlabel("Population")
plt.ylabel("Gini Index")
plt.show()


# In[124]:


fig, ax = plt.subplots(figsize=(5, 5))
sns.regplot(model.fittedvalues,model.resid, scatter_kws={'alpha': 0.25}, line_kws={'color': 'C2', 'lw': 2}, ax=ax)
ax.set_xlabel('predicted')
ax.set_ylabel('residuals')
plt.tight_layout()
plt.show()


# In[ ]:





# In[29]:


import statsmodels.api as sm


# In[39]:


y= Dataset2["gini_std"]
x2= Dataset2[["logarithm_base10"]]
x2=sm.add_constant(x)


# In[40]:


mdl2 = sm.OLS(y, x2).fit()
print(mdl2.summary())


# In[38]:


y_pred = mdl2.predict()
# Set the style and dimension for a scatter plot.
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid') # Set the style
plt.rcParams['figure.figsize'] = [20, 10]
# Add the predicted values in the x axis and the observed values in the

import seaborn as sns
sns.regplot(x = y_pred, y = y, ci = None, line_kws = {'color':'red'})
plt.ylabel('Observed values', fontsize = 14)
plt.xlabel('Predicted values', fontsize = 14)


# In[ ]:





# In[ ]:





# In[ ]:





# In[79]:


mdl = sm.OLS(y, x).fit()
print(mdl.summary())


# In[4]:


y= Dataset2["gini_std"]
x= Dataset2["population"]
x=sm.add_constant(x)


# In[78]:


mdl = sm.OLS(y, x).fit()
print(mdl.summary())


# In[56]:


resid = mdl.resid
# Set the style and dimension of the chart.
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = [15, 10]
# Create a histogram of residuals/errors.
resid.plot.hist(grid = True, bins = 60, edgecolor = 'white',
linewidth = 1.0)
plt.title('Residuals Distribution', fontsize = 20)


# In[82]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(Dataset2[['population']], Dataset2[['gini_std']])
predictions = lr.predict(Dataset2[['population']])


# In[85]:


import statsmodels.formula.api as sm
model = sm.ols(formula='gini_std ~ population', data=Dataset2)
fitted1 = model.fit()
fitted1.summary()


# In[83]:


mdl = sm.OLS(y, x).fit()
print(mdl.summary())


# In[ ]:





# In[ ]:





# In[16]:


from scipy.stats.stats import kendalltau


# In[19]:


corr = Dataset2.corr(method='kendall')


# In[21]:


from pylab import rcParams


# In[23]:


rcParams['figure.figsize'] = 14.7,8.27
sns.heatmap(corr, 
           xticklabels=corr.columns.values, 
           yticklabels=corr.columns.values, 
           cmap="YlGnBu",
          annot=True)


# In[25]:


import scipy.stats as stats


# In[32]:


stats.kendalltau(Dataset2['population'], Dataset2['gini_std'])


# In[34]:


stats.kendalltau(Dataset2['logarithm_base10'], Dataset2['gini'])


# In[28]:


Dataset2.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[104]:


Dataset2.reset_index(drop=True)


# In[108]:


x= Dataset3[["population",'incomegroup']]
x.corr(method = 'pearson')


# # Hypothesis 2 

# In[34]:


Dataset3= Dataset[[ "gini_std", "incomegroup", "population"]]


# In[35]:


Dataset3 = Dataset3.dropna()


# In[36]:


Dataset3.isnull().sum()


# In[37]:


Dataset4 = Dataset3[(Dataset3['incomegroup'] == "High income") | (Dataset3[ 'incomegroup'] == "Low income")]
Dataset4.head(2)


# In[24]:


Dataset4 = Dataset3[(Dataset3['incomegroup'] == "High income") ]

Dataset4.head(2)


# In[25]:


Dataset4.shape


# In[29]:


sns.boxplot(x="incomegroup" , y="gini_std", data=Dataset4,palette='rainbow')


# In[32]:


sns.violinplot(x="incomegroup", y="gini_std", data=Dataset4,palette='rainbow')


# In[33]:


sns.stripplot(x="incomegroup", y="gini_std", data=Dataset4)


# In[34]:


sns.swarmplot(x="incomegroup", y="gini_std", data=Dataset3)


# In[39]:


from scipy.stats.stats import spearmanr # Spearman’s


# In[40]:


rho, pho = spearmanr(Dataset3["incomegroup"], Dataset3["gini_std"])
print(rho, pho)


# In[53]:


rho, pho = spearmanr(Dataset4["incomegroup"], Dataset4["gini_std"])
print(rho, pho)


# In[43]:


print("rho =", round(rho, 1))
if pho < .001:
    print("pho <.001")
elif pho <.01:
    print("pho <.01")
elif pho <.05:
    print("pho <.05")
else:
    print("Not significant")


# # Further Analysis 

# In[129]:


X = Dataset2['gini'].values.reshape(-1,1)
y = Dataset2['population'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[130]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


# In[131]:


#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)


# In[132]:


y_pred = regressor.predict(X_test)


# In[133]:


Datasetn = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
Datasetn


# In[134]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




