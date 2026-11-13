#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

plt.style.use("ggplot")
import nltk 
print("imported succesfully")


# In[2]:


df = pd.read_csv("Reviews.csv")


# In[3]:


df.head()    


# In[4]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[5]:


example = df['Text'][50]
print(example)


# In[6]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[7]:


# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[8]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[16]:


subset = df.sample(500, random_state=42)


# In[17]:


results = []

for i, row in tqdm(subset.iterrows(), total=len(subset)):
    try:
        text = row['Text']
        myid = row['Id']

        # VADER
        vader_result = sia.polarity_scores(text)
        vader_result = {f"vader_{k}": v for k, v in vader_result.items()}

        # RoBERTa
        roberta_result = polarity_scores_roberta(text)

        # Merge
        combined = {"Id": myid, **vader_result, **roberta_result}
        results.append(combined)

    except Exception as e:
        print(f"Broke for id {myid}: {e}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("sentiment_results_sample.csv", index=False)


# In[18]:


results_df = pd.DataFrame(results)
# pick label with highest score
results_df['roberta_label'] = results_df[['roberta_neg','roberta_neu','roberta_pos']].idxmax(axis=1)

# plot counts
sns.countplot(x='roberta_label', data=results_df)
plt.title("Distribution of RoBERTa Sentiment Predictions")
plt.show()


# In[19]:


# Plot distribution of review scores
plt.figure(figsize=(8,5))
sns.countplot(x='Score', data=df, palette="viridis")
plt.title("Distribution of Review Scores")
plt.xlabel("Score (1 = Bad, 5 = Excellent)")
plt.ylabel("Number of Reviews")
plt.show()


# In[26]:


# Merge the results with the original dataframe to get the Score column back
results_df = results_df.merge(df[['Id','Score']], on='Id', how='left')

# Map score (1–5) to a simpler label
def score_to_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

# Add new column
results_df['true_sentiment'] = results_df['Score'].apply(score_to_sentiment)

# Visualization: Compare RoBERTa predictions with true Score
plt.figure(figsize=(8,6))
sns.countplot(x="Score", hue="roberta_label", data=results_df, palette="Set2")
plt.title("RoBERTa Sentiment Predictions by Review Score")
plt.xlabel("Review Score (1–5)")
plt.ylabel("Number of Reviews")
plt.legend(title="Predicted Sentiment")
plt.show()


# In[23]:


df.head()


# In[ ]:





# In[ ]:




