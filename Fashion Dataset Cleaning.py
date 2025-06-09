#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(r"C:\Users\megha\OneDrive\Desktop\fashion_dataset.csv")


# In[3]:


df.head()


# In[4]:


df['brand'] = df['brand'].str.strip().str.title()
df['type'] = df['type'].str.strip().str.title()


# In[5]:


df.head()


# In[6]:


duplicates = df.duplicated()
duplicates


# In[7]:


duplicates_count = duplicates.sum()
print(f"number of duplicates rows :{duplicates_count} ")


# In[8]:


df['price_usd'].describe()


# In[9]:


items_over_5000 = df[df['price_usd']>5000]
count_items_over_5000 = len( items_over_5000)


# In[10]:


print(items_over_5000)


# In[11]:


print(count_items_over_5000)


# In[12]:


top_brands = df['brand'].value_counts().head(10)
print(top_brands)


# In[13]:


average_price_by_brand = df.groupby('brand')['price_usd'].mean().sort_values(ascending=False).head(10)
print(average_price_by_brand)


# In[14]:


avg_price_by_type = df.groupby('type')['price_usd'].mean()


# In[15]:


print(avg_price_by_type)


# In[16]:


df['description_words'] = df['description'].str.lower().str.split()

# Get first two words as possible color
df['possible_color'] = df['description_words'].apply(
    lambda x: ' '.join(x[:2]) if len(x) > 1 else x[0] if x else 'Unknown')


# In[17]:


df['color'] = df['possible_color'].apply(
    lambda x: x if len(x.split()) == 1 else x if x.count(' ') == 1 else 'Unknown'
)


# In[18]:


df=df.drop(columns=['possible_color','color'])


# In[19]:


df.head(20)


# In[20]:


import matplotlib

# Get a list of all named CSS colors (simple, consistent)
color_names = list(matplotlib.colors.CSS4_COLORS.keys())

extra_colors = ['taupe', 'ivory', 'champagne', 'nude', 'tan', 'off-white', 'burgundy', 'mint', 'mauve', 'peach', 'khaki', 'multicolor', 'transparent']

all_colors = set(color_names + extra_colors)


# In[21]:


# Function to find matching colors from words
df['colors_found'] = df['description_words'].apply(
    lambda word_list: [word for word in word_list if word in all_colors]
)

# Optional: Convert to string format
df['colors_combined'] = df['colors_found'].apply(
    lambda x: ', '.join(x) if x else 'Unknown'
)


# In[22]:


df.head(20)


# In[23]:


df['colors_combined'].value_counts()['Unknown']


# In[24]:


unknown_colors = df[df['colors_combined'] == 'Unknown']
print(unknown_colors.shape[0])  # Tells you how many there are
unknown_colors.head(30)


# In[25]:


# If there's at least 2 words, join the last two. Otherwise, just take the last one.
df['possible_product_type'] = df['description_words'].apply(
    lambda x: ' '.join(x[-1:]) if len(x) >= 1 else (x[-1] if x else 'Unknown')
)


# In[26]:


df.head(20)


# In[27]:


unknown_type = df[df['possible_product_type'] == 'Unknown']
print(unknown_type.shape[0])  # Tells you how many there are
unknown_type.head(30)


# In[28]:


df = df.drop(columns = ['description_words', 'colors_found'])


# In[29]:


df


# In[30]:


df = df.rename(columns = {'possible_product_type': 'product_type', 'colors_combined':'colors'})


# In[31]:


df


# In[32]:


average_price_per_brand = df.groupby('brand')['price_usd'].mean()


# In[33]:


luxury_brands = average_price_per_brand[average_price_per_brand>1000].index.tolist()


# In[34]:


luxury_brands


# In[35]:


df['is_luxury'] = df['brand'].apply(lambda x: x in luxury_brands)


# In[36]:


df


# In[37]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# In[38]:


df.rename(columns = {'colors_combined': 'colors'})


# In[39]:


df = pd.get_dummies(df, columns = ['type'], prefix='type')


# In[40]:


columns_to_use = ['price_usd', 'colors', 'product_type', 'type_Mens', 'type_Womens', 'is_luxury']
df_ml = df[columns_to_use].copy()
df_ml


# In[41]:


df_ml = pd.get_dummies(df_ml, columns = ['colors', 'product_type'],drop_first=True)


# In[42]:


print("Shape of final dataset:", df_ml.shape)
df_ml.head()


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[44]:


# Now define X and y
X = df_ml.drop('is_luxury', axis=1)
y = df_ml['is_luxury']


# In[45]:


X = df_ml.drop('is_luxury', axis = 1) # Features (everything except the label)
y = df_ml['is_luxury'] # Target (what we want to predict)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# In[ ]:





# In[47]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[48]:


y_pred = model.predict(X_test)


# In[49]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[50]:


classification = classification_report(y_test, y_pred)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# In[55]:


rf_model = RandomForestClassifier(random_state = 42)


# In[56]:


rf_model.fit(X_train, y_train)


# In[59]:


y_pred_rf = rf_model.predict(X_test)


# In[60]:


print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))


# In[61]:


from xgboost import XGBClassifier

# Initialize XGBoost
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Train
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
from sklearn.metrics import classification_report, accuracy_score
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))


# In[62]:


import matplotlib.pyplot as plt
import pandas as pd

# 1. Get feature importances
feature_importances = xgb_model.feature_importances_

# 2. Create a DataFrame to display it nicely
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importances
})

# 3. Sort by importance
importance_df = importance_df.sort_values(by='importance', ascending=False)

# 4. Show top 20 features
print(importance_df.head(20))


# In[63]:


# Plot top 20 important features
plt.figure(figsize=(12,8))
plt.barh(importance_df['feature'][:20][::-1], importance_df['importance'][:20][::-1])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.show()


# In[ ]:




