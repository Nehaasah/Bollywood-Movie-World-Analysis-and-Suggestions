#!/usr/bin/env python
# coding: utf-8

# ### Importing Python Libraries

# In[1]:


import numpy as np
import pandas as pd
import altair as alt
import re
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[2]:


df = pd.read_csv('C:/Users/Dell/Desktop/PROJECT/Bollywood Movie Recommendation/bollywood_full.csv')


# In[3]:


df.info()


# In[5]:


df.head(6)


# In[9]:


df.tail(6)


# ## DataSet Exploration

# The Dataset has some columns which might not be useful for analysis. Let's drop them to have only significant data part.

# In[10]:


# Droping columns which will not be used for analysis
df = df.drop(['title_y', 'imdb_id', 'poster_path', 'wiki_link', 'tagline', 'release_date'], axis=1)


# ### Renaming for ease of reference

# In[11]:


df = df.rename(columns={'original_title': 'title', 'year_of_release':'year', 'imdb_rating':'rating', 'imdb_votes':'votes', 'wins_nominations':'awards'})


# ## Cleaning Process

# Awards columns has both awards and nominations. Separation is required for better understanding so separating them into wins and nominations. 

# In[12]:


# Data cleaning
df['awards'] = df['awards'].apply(lambda x: re.sub(r'[A-Za-z]', '', str(x)))
df[['wins', 'nominations']] = df.awards.str.split('&', expand=True)
df['wins'] = df['wins'].str.strip()
df['wins'] = df['wins'].apply(lambda x: 0 if x=='' else x)
df['wins'] = df['wins'].replace(np.nan, 0)
df['wins'] = df['wins'].astype(int)


# Actors column has list of all the actors from the movie. 
# Let's first explore dataset in regards to lead actor , hence we need only the First Actor.

# In[13]:


actors = df.actors.str.split('|', expand=True)
df['lead_actor'] = actors[0]


# Genres columns has many genres pipe seperated to which movie might belong.
# Let's consider first genre which will describe movie as main genre and then discard rest of them.
# Also Droping duplicates from the dataset

# In[14]:


df[['genre','genre2', 'genre3']] = df.genres.str.split("|",expand=True)
df = df.drop(['genres', 'awards', 'actors', 'genre2', 'genre3', 'nominations'], axis=1)

df = df.drop_duplicates()


# ### Let’s look at the movies released in each year

# In[22]:


alt.Chart(df, title='Bollywood movies released over the years').mark_circle(opacity=0.8).transform_window(
    id='rank()',
    groupby=['year']
).encode(
    alt.X('year', scale=alt.Scale(type='linear', domain=[1950, 2020])),
    alt.Y('id:O', axis=None, sort='descending'),
    tooltip=['title', 'year', 'rating'],
    color='genre'
).properties(height=500, width=800)


# ##  ;)    Drama has Always dominated bollywood.

# ### Action movies have gained momentum since 1980 and Comedy from early 2000. 

# ## Thrillers are in vogue right now.

# ===================================================================================================================== 

#  Let’s look at how many movies of each genre were made each year

# In[19]:


temp1 = df.groupby(['genre', 'year']).size().reset_index(name='count')
alt.Chart(temp1, title='Bollywood movies genre over the years').mark_bar(opacity=0.9).encode(
    alt.X('year', scale=alt.Scale(type='linear', domain=[1955, 2015])),
    alt.Y('count', stack='zero'),
    color='genre',
    tooltip=['genre', 'year']
).properties(height=500, width=800)


# - Till 1970s drama dominated bollywood. But, after that action movies started gaining momentum,Amitabh Bachchan may be the reason we will find soon ;) . 

# - Looks like 80s till 2000 was action era of bollywood.Ofcourse, Actors such as Akshay kumar, Ajay devgan, Sunil shetty, Sunny deol, Sanjay dutt were dominating the bollywood scene that time with lots of action on the big screen. 

# - From early 2000 comedy movies started gaining catching up to drama and action. From 2000 till 2010 some of the best comedy movies were made like hera pheri one of my favourites. 

# - Romantic movies count have been low compared to drama, action or comedy also for few brief period hardly any romantic movies were made. Romantic movie although low in count have been big success in bollywood.

# ### Let’s explore the movies that won awards in bollywood

# In[27]:


winners = df[df['wins']!=0]
alt.Chart(winners, title='Award winning movies over the years').mark_circle(opacity=0.8).transform_window(
    id='rank()',
    groupby=['year']
).encode(
    alt.X('year', scale=alt.Scale(type='linear', domain=[1950, 2020])),
    alt.Y('id:O', axis=None, sort='descending'),
    tooltip=['title', 'lead_actor'],
    color='genre'
).properties(height=500, width=800)


# - Wow, Drama movies dominated the award scene till late 80s. But, since 1990s we are seeing healthy mix of genre in awards.

# - But, we can clearly see that bollywood has started giving toooo many awards since 2000. Which I Think is because of diverse audience and more no of movies and even more no of different Award Shows.

# ### I always wonder Does movie that win multiple records also have high ratings? Lets end the curiousity
# 
# Checking for movies that have won more than 5 awards

# In[29]:


alt.Chart(winners[winners['wins']>5], title='Multiple award winning movies over the years').mark_circle(opacity=0.8).transform_window(
    id='rank()',
    groupby=['year']
).encode(
    alt.X('year', scale=alt.Scale(type='linear', domain=[1950, 2020])),
    alt.Y('rating', scale=alt.Scale(type='linear', domain=[0, 10])),
    tooltip=['title', 'year'],
    color='genre'
).properties(height=500, width=800)


# Ohkkkk ! It's a thing.

# But, What about votes? Does the movie that have higher rating also have most votes?

# ### Movies which won more than 5 awards with their rating and votes

# In[35]:


alt.Chart(winners[winners['wins']>=5], title='Movies that won more than 5 award with its rating and votes').mark_circle(opacity=0.8).transform_window(
    id='rank()',
    groupby=['year']
).encode(
    alt.X('year', scale=alt.Scale(type='linear', domain=[1950, 2020])),
    alt.Y('votes', scale=alt.Scale(type='log', domain=[10, 1000000])),
    tooltip=['title', 'votes' , 'rating'],
    color='genre'
).properties(height=600, width=700)


# It seems that lot of movies from early bollywood era haven’t been rate by many user. Handful of bollywood movies have more than 100000 ratings.

# 3 idios outclassing every other movie with whopping 300K votes.

# ### Who is the actor with multiple appeearence in award winning movies? Ahhha this is a great ques always.
# 
# We will find out lead actors who appeared in award winning movies.

# In[37]:


temp2 = winners.lead_actor.value_counts().head(10).rename_axis('actor').reset_index(name='appearences')
alt.Chart(temp2, title='Lead actor with most appearnces in award winning movies').mark_bar().encode(
    alt.X('actor', axis=alt.Axis(labelAngle=-45)),
    alt.Y("appearences")
).properties(width=800)


# Great, Big B, Shri Amitabh Bachchan would have been obvious choice considering Big B’s career span over five decades.Rajesh Khanna is second with 40 award winning movies in short career span. For a brief period in bollywood Rajesh Khanna was surely the biggest star. 
# king khan is fourth right after Ajay devgan in giving award winning movies.

# ### Let's look at the award winning movies of leading bollywood actors across time and see how they were rate by public.

# In[38]:


temp3 = winners[winners['lead_actor'].isin(temp2.actor.tolist())]
alt.Chart(temp3, title='Award winning Movies of Top actors').mark_circle(opacity=0.8).transform_window(
    id='rank()',
    groupby=['year']
).encode(
    alt.X('year', scale=alt.Scale(type='linear', domain=[1950, 2020])),
    alt.Y('rating', scale=alt.Scale(type='linear', domain=[0, 10])),
    tooltip=['title', 'year', 'rating','genre'],
    color='lead_actor'
).properties(height=600, width=800)


# Interesting! Lot of bad movies have started winning awards in hollywood in recent years. Going as low as humshakals and himmatwala each of which have close to two rating and still won awards.

# ### Who is the most successful actor in bollywood?
# 
# Calculating percentage of award winning movies from total movies done by leaading bollywood actors

# In[40]:


movies_done_by_top_actors = df[df['lead_actor'].isin(temp3.lead_actor.tolist())]
success_percentage = (temp3.lead_actor.value_counts() / movies_done_by_top_actors.lead_actor.value_counts() * 100).rename_axis('actor').reset_index(name='% success')
alt.Chart(success_percentage, title='Success percentage of top actors').mark_bar().encode(
    alt.X('actor', axis=alt.Axis(labelAngle=-45)),
    alt.Y("% success")
).properties(width=600)


# Expected. Shah Rukh Khan is known as king khan for a reason with highest 80% success rate.

# ### Does movie length impact its rating?
# Is there a sweet spot of movie runtime that impact the ratings. Do the directors know that?

# In[42]:


alt.Chart(df, title='How rating varies with runtime').mark_circle().encode(
    alt.X('runtime', bin=alt.Bin(maxbins=20), scale=alt.Scale(type='linear', domain=[60, 220])),
    alt.Y('rating', bin=alt.Bin(maxbins=10)),
    size='count()'
).properties(height=300, width=800)


# It seems bollywood has consensus on movie length. Most of the movie falls between 120-140 mins.

# ### Highest rated bollywood movies since 1950 till 2019?
# 

# In[43]:


temp5 = df.groupby(['year'])[['title','genre', 'rating']].max().reset_index()
alt.Chart(temp5, title='Highest rates movies of all years').mark_circle(opacity=0.8).transform_window(
    id='rank()',
    groupby=['year']
).encode(
    alt.X('year', scale=alt.Scale(type='linear', domain=[1950, 2020])),
    alt.Y('rating', scale=alt.Scale(type='linear', domain=[0, 10])),
    tooltip=['title', 'year', 'rating','genre'],
    color='genre'
).properties(height=200, width=800)


# Wow, Although Drama dominated movies in till 1980 but highest rated movies are "ROMANTIC" for those years. I guess some really good romantic gems are hidden there. Also, action dominated bollywood since 1990s but ratings show that people love dramas more now.

# ### Does Movie genre and word cloud of summary have some realation?
# 

# In[53]:


temp6 = df.groupby(['genre'])['summary']

x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

for name, group in temp6:
    text = ' '.join(group.tolist())
    wordcloud = WordCloud(width=500, height=500, margin=0,background_color ='lightblue' ,
                          stopwords = stopwords, mask=mask)
    wordcloud = wordcloud.generate(text)
    print(name)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# ## Looks like movie summary are good descriptiors of type of genre.

# Would Explore the Dataset more and add the results ...
