#!/usr/bin/env python
# coding: utf-8

# In[56]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
import seaborn as sns
import os
sns.set_style('darkgrid')

# Dataset
# Step-2. Import data
os.chdir('/Users/pauline/Documents/Python')
data = pd.read_csv("Tab-Morph.csv")
data.index = data.profile

# Fit the model
mod = sm.tsa.statespace.SARIMAX(data['sedim_thick'], trend='c', order=(1,1,1))
res = mod.fit(disp=False)
print(res.summary())
