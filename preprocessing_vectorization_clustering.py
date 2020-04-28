#!/usr/bin/env python
# coding: utf-8

# 
# # Required Imports
# 

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import glob
import nltk

import itertools
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import chart_studio.plotly as py
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

import gensim.models.word2vec as w2v
from sklearn.manifold import TSNE
import plotly.express as px

from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from fastcluster import linkage
from matplotlib.colors import rgb2hex, colorConverter


# 
# # Reading all csv's and creating a dataframe(for english language)
# 

# In[2]:


all_files=glob.glob("/Users/rishabhsrivastava/Downloads/CSVS/DIKSHA APP REVIEWS AND RATINGS/"+"/*.csv")
dflist = []


# In[3]:


for filename in all_files:
    # Dataframe of one file
    df_sm = pd.read_csv(filename, index_col=None, header=0)
    dflist.append(df_sm)
    
df = pd.concat(dflist, axis=0, ignore_index=True)


# In[4]:


df.dropna(subset=["Review Text"],inplace=True)


# In[5]:


eng_data = df.loc[df['Reviewer Language']=='en']
eng_df = pd.DataFrame(eng_data)
# eng_df


# In[6]:


eng_df.reset_index(inplace = True) 


# In[7]:


eng_df


# In[8]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import re

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
       # The yield statement suspends function’s execution and sends a value back to the caller.
        yield subtree.leaves()

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def get_terms(tree):

    for leaf in leaves(tree):
        term = [ w for w,t in leaf if acceptable_word(w) ]
        yield term


# In[9]:


grammar =r"""
  NP: {<DT|JJ|NN.*>+}          
  PP: {<IN><NP>}            
  VP: {<VB.*><NP|PP|CLAUSE>+$}
  CLAUSE: {<NP><VP>}          
  """
      
    


# In[10]:


def phrase_extraction(text, grammar):
    text = text.lower()
    sentence_re = r'''(?x)          
      (?:[A-Z]\.)+        
    | \w+(?:-\w+)*        
    | \$?\d+(?:\.\d+)?%?  
    | \.\.\.              
    | [][.,;"'?():_`-]    
    '''
    
    ls = [] 
    word_token_ls = text.split(" ")

    toks = nltk.regexp_tokenize(text, sentence_re)
    postoks = nltk.tag.pos_tag(toks)
    
    chunker = nltk.RegexpParser(grammar)
    
    tree = chunker.parse(postoks)
    terms = get_terms(tree)
    for term in terms:
        ls.append(" ".join(term)) 
    return list(set(ls))


# 
# # Preprocessing of Review Text
# 

# In[11]:


ls = list(eng_df["Review Text"])
print(ls)


# In[12]:


dic = dict(zip(eng_df["Review Text"],eng_df["Star Rating"]))
dic


# In[13]:


# changing review text into lowecase

review_text_lower = {}
for i ,j in dic.items():
    review_text_lower[i.lower()] = j
    
review_text_lower


# In[14]:


# changing review text into lowecase

# out = map(lambda x:x.lower(), dic)  
# review_text_lower = dict(out)
# review_text_lower


# In[83]:


# Numbers removing
import re
review_text_lower_wdoutno = {}
for i ,j in review_text_lower.items():
    review_text_lower_wdoutno[re.sub(r'\d+', '', i)] = j
    
review_text_lower_wdoutno


# In[84]:


# Numbers removing

# import re
# review_text_lower_wdoutno = list(map(lambda x: re.sub(r'\d+', '', x), review_text_lower)) 
# review_text_lower_wdoutno



# In[87]:


# Remove punctuation
import string 

def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 

review_text_wdout_punct = {}
for i,j in review_text_lower.items():
    review_text_wdout_punct[remove_punctuation(i)] = j
#     review_text_wdout_punct.append(x)
review_text_wdout_punct   


# In[88]:


# remove whitespace from text 
def remove_whitespace(text): 
    return " ".join(text.split()) 

review_text_wdout_whitespace = {}
for i,j in review_text_wdout_punct.items():
    review_text_wdout_whitespace[remove_whitespace(i)] = j
#     x = 
#     review_text_wdout_whitespace.append(x)
    
review_text_wdout_whitespace  


# In[89]:


# convert a list to string    
def listToString(s):  
    str1 = ""   
    for ele in s:  
        str1 += ele   
        str1 += ' '
    return str1  
        


# In[90]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
# remove stopwords function 
def remove_stopwords(text): 
    stop_words = list(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return filtered_text 
  
review_text_wdout_stopwords = {}
for i,j in review_text_wdout_whitespace.items():
    x = remove_stopwords(i)
    y = listToString(x)
    review_text_wdout_stopwords[y] = j
review_text_wdout_stopwords  


# In[91]:


# remove emoji
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

review_text_wdout_emojis = {}
for i,j in review_text_wdout_stopwords.items():
    x = deEmojify(i)
    review_text_wdout_emojis[x] = j
review_text_wdout_emojis  


# In[92]:


# lemmatization
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
lemmatizer = WordNetLemmatizer() 

# lemmatize string 
def lemmatize_word(text): 
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
    return lemmas 
  
review_text_lemmas = {}
for i,j in review_text_wdout_emojis.items():
    x = lemmatize_word(i)
    y = listToString(x)
    review_text_lemmas[y] = j
review_text_lemmas  


# In[93]:


#stemming

# from nltk.stem.porter import PorterStemmer 
# from nltk.tokenize import word_tokenize 
# stemmer = PorterStemmer() 

# # stem words in the list of tokenised words 
# def stem_words(text): 
#     word_tokens = word_tokenize(text) 
#     stems = [stemmer.stem(word) for word in word_tokens] 
#     return stems 

 


# In[94]:


# remove review text containing less than 3 words using regex (findall()) 

import re 

processed_review_text = {}
for i,j in review_text_lemmas.items():
    res = len(re.findall(r'\w+', i)) 
    if(res>=3):
        processed_review_text[i] = j
        
processed_review_text    


# 
# # Dataframe of filtered Review Text and their corresponding star rating 
# 

# In[95]:


# df2 = pd.DataFrame(processed_review_text.values())
# df2

initial_df = pd.DataFrame(list(zip(processed_review_text.keys(), processed_review_text.values())), 
               columns =['Review Text', 'Star Rating']) 
initial_df


# In[96]:


final_processed_review_text = []
for key in processed_review_text.keys():
    final_processed_review_text.append(key)
final_processed_review_text


# 
# # Vectorization of Review Text(Word Embeddings) 512-D
# 

# In[97]:


# Vectorization

def vectorization_of_list(input_list):
    #word embedding(vectorization)
    embed = hub.Module("/Users/rishabhsrivastava/Downloads/vectorization_trained_dataset/")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(input_list))
#         print(message_embeddings)
        lst = []
        for i in message_embeddings:
            df = pd.DataFrame([i])
            lst.append(df)
    frame = pd.concat(lst)
    return frame


# In[98]:


vectorized_frame = vectorization_of_list(final_processed_review_text)
vectorized_frame  


# In[99]:


# from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarity([vectorized_frame[0]], [vectorized_frame[1]])


# In[ ]:





# In[100]:


review_text_df = pd.DataFrame(final_processed_review_text,columns=['Review Text'])


# In[101]:


vectorized_frame.set_index(review_text_df["Review Text"], inplace = True) 
vectorized_frame


# In[102]:


vectorized_frame.to_csv ('/Users/rishabhsrivastava/Downloads/CSVS/Vectorized_processed_review_text.csv', index = False, header=True)
vectorized_frame


# 
# # Dimension Reduction from 512D to 3D and 2D (T-SNE)
# 

# In[103]:


def TSNE_3D(df):
    get_ipython().run_line_magic('pylab', 'inline')

    #Reduce Dimensinality
    X_embedded = TSNE(n_components=3).fit_transform(df)
    vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2","ft3"])
    #vec_df
    #plot 3-D graph
    fig = px.scatter_3d(vec_df,x="ft1",y="ft2",z="ft3")
    fig.show()


# In[104]:


TSNE_3D(vectorized_frame)


# In[105]:


def TSNE_2D(df):
    get_ipython().run_line_magic('pylab', 'inline')

    #Reduce Dimensinality
    X_embedded = TSNE(n_components=2).fit_transform(df)
    vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2"])
    #vec_df
    #plot 3-D graph
    fig = px.scatter(vec_df,x="ft1",y="ft2")
    fig.show()


# In[106]:


TSNE_2D(vectorized_frame)


# 
# # Dendrograms and hierarchial clustering of Review Text vectors - scatter plot
# 

# In[107]:


def dendrogram_genetator(df):
    plt.figure(figsize=(10, 7))  
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    


# In[108]:


get_ipython().run_line_magic('pylab', 'inline')
#Reduce Dimensinality
X_embedded = TSNE(n_components=2).fit_transform(vectorized_frame)
vec_df = pd.DataFrame(X_embedded, columns=["ft1","ft2"])
vec_df


# In[109]:


dendrogram_genetator(vec_df)


# In[110]:


def dendrogram_genetator_with_thresold(df,thresold):
    plt.figure(figsize=(10, 7))
#     y=800
    plt.title("Dendrograms")  
    dend = shc.dendrogram(shc.linkage(df, method='ward'))
    plt.axhline(thresold, color='r', linestyle='--')
    


# In[111]:


dendrogram_genetator_with_thresold(vec_df,2000)


# In[112]:


def hierarchial_clustering(df):
    cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
    cluster.fit_predict(df)
    
    plt.figure(figsize=(10, 7))  
    plt.scatter(df['ft1'], df['ft2'], c=cluster.labels_) 
    


# In[113]:


hierarchial_clustering(vec_df)


# In[114]:


#result.reset_index(drop=True, inplace=True)
vec_df.set_index(review_text_df["Review Text"], inplace = True) 
vec_df


# 
# # Element extraction from clusters
# 

# In[115]:


def cluster_element_extraction(vec_df):
    sns.set_palette('Set1', 10, 0.65)
    palette = (sns.color_palette())
    #set_link_color_palette(map(rgb2hex, palette))
    sns.set_style('white')
    
    np.random.seed(25)
    
    link = linkage(vec_df, metric='correlation', method='ward')

    figsize(8, 3)
    den = dendrogram(link, labels=vec_df.index)
    plt.xticks(rotation=90)
    no_spine = {'left': True, 'bottom': True, 'right': True, 'top': True}
    sns.despine(**no_spine);

    plt.tight_layout()
    plt.savefig('feb2.png');
    
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
                
    class Clusters(dict):
        def _repr_html_(self):
            html = '<table style="border: 0;">'
            for c in self:
                hx = rgb2hex(colorConverter.to_rgb(c))
                html += '<tr style="border: 0;">'                 '<td style="background-color: {0}; '                            'border: 0;">'                 '<code style="background-color: {0};">'.format(hx)
                html += c + '</code></td>'
                html += '<td style="border: 0"><code>' 
                html += repr(self[c]) + '</code>'
                html += '</td></tr>'

            html += '</table>'

            return html
    
    cluster_classes = Clusters()
    for c, l in cluster_idxs.items():
        i_l = [den['ivl'][i] for i in l]
        cluster_classes[c] = i_l
        
    return cluster_classes
    


# In[116]:


cluster_element_extraction(vec_df)


# 
# # Element extraction from clusters for each rating
# 

# In[117]:


def clustering_rating_wise(key_rate_df, rating):
    key_data = key_rate_df.loc[key_rate_df['Star Rating'] == rating]
    key_rate_df1 = pd.DataFrame(key_data)
#     return key_rate_df1
    ky_ls =key_rate_df1['Review Text'].tolist()
    
    frame = vectorization_of_list(ky_ls)
    frame.set_index(key_rate_df1["Review Text"], inplace = True) 
    return cluster_element_extraction(frame)


# In[118]:


def keyword_freq_per_cluster(cluster_elem, cluster_no):
    #changing to lower order
    out = map(lambda x:x.lower(), cluster_elem)  
    elem_lower = list(out)  
    #print(output) 
    #Creating a new Dtaframe which is required for final csv
    df_rate_c = pd.DataFrame(columns=['Review Text', 'Keywords', 'Cluster'])

    for i in elem_lower:
        if(i == ''):
            df_rate_c = df_rate_c.append({'Review Text': i, 'Keywords': '[]', 'Cluster': cluster_no}, ignore_index=True)
        else:
            x = phrase_extraction(i, grammar)
            df_rate_c = df_rate_c.append({'Review Text': i, 'Keywords': x, 'Cluster': cluster_no}, ignore_index=True)
    
    ky_ls = (df_rate_c['Keywords'].tolist())
    ky_ls_df = pd.DataFrame(ky_ls)
    return ky_ls_df[0].value_counts()[:30]


# 
# # cluster analysis for rating 5 - elements of cluster, wordcloud, frequency of most frequent words that are appearing
# 

# In[119]:


elem_rate5 = clustering_rating_wise(initial_df, 5)


# In[120]:


g5_elem = elem_rate5['g']
g5_elem


# In[121]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(g5_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[123]:


from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import re

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
       # The yield statement suspends function’s execution and sends a value back to the caller.
        yield subtree.leaves()

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
                    and word.lower() not in stopwords)
    return accepted

def get_terms(tree):

    for leaf in leaves(tree):
        term = [ w for w,t in leaf if acceptable_word(w) ]
        yield term


# In[124]:


keyword_freq_per_cluster(g5_elem,1).to_dict()


# In[125]:


import re
g5_most_frequent = [y for y in g5_elem if re.search('good app', y)]
g5_most_frequent


# In[126]:


r5_elem = elem_rate5['r']
r5_elem


# In[127]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(r5_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[128]:


keyword_freq_per_cluster(r5_elem,2).to_dict()


# In[129]:


import re
r5_most_frequent = [y for y in r5_elem if re.search('india', y)]
r5_most_frequent


# In[130]:


c5_elem = elem_rate5['c']
c5_elem


# In[131]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(c5_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[132]:


keyword_freq_per_cluster(c5_elem,3).to_dict()


# In[133]:


import re
c5_most_frequent = [y for y in c5_elem if re.search('good students', y)]
c5_most_frequent


# 
# # cluster analysis for rating 4 - elements of cluster, wordcloud, frequency of most frequent words that are appearing
# 

# In[134]:


elem_rate4 = clustering_rating_wise(initial_df, 4)


# In[135]:


g4_elem = elem_rate4['g']
g4_elem


# In[136]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(g4_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[137]:


keyword_freq_per_cluster(g4_elem,1).to_dict()


# In[138]:


import re
g4_most_frequent = [y for y in g4_elem if re.search('good app', y)]
g4_most_frequent


# In[139]:


r4_elem = elem_rate4['r']
r4_elem


# In[140]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(r4_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[141]:


keyword_freq_per_cluster(r4_elem,2).to_dict()


# In[142]:


import re
r4_most_frequent = [y for y in r4_elem if re.search('goodbut question answer available', y)]
r4_most_frequent


# In[143]:


c4_elem = elem_rate4['c']
c4_elem


# In[144]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(c4_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[145]:


keyword_freq_per_cluster(c4_elem,3).to_dict()


# In[146]:


import re
c4_most_frequent = [y for y in c4_elem if re.search('teachers', y)]
c4_most_frequent


# 
# # cluster analysis for rating 3 - elements of cluster, wordcloud, frequency of most frequent words that are appearing
# 

# In[147]:


elem_rate3 = clustering_rating_wise(initial_df, 3)


# In[148]:


g3_elem = elem_rate3['g']
g3_elem


# In[149]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(g3_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[150]:


keyword_freq_per_cluster(g3_elem,1).to_dict()


# In[151]:


import re
g3_most_frequent = [y for y in g3_elem if re.search('add class st state board maharashtra', y)]
g3_most_frequent


# In[152]:


r3_elem = elem_rate3['r']
r3_elem


# In[153]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(r3_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[154]:


keyword_freq_per_cluster(r3_elem,2).to_dict()


# In[155]:


import re
r3_most_frequent = [y for y in r3_elem if re.search('easy learn', y)]
r3_most_frequent


# In[156]:


c3_elem = elem_rate3['c']
c3_elem


# In[157]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(c3_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[158]:


keyword_freq_per_cluster(c3_elem,3).to_dict()


# In[207]:


import re
c3_most_frequent = [y for y in c3_elem if re.search('nice', y)]
c3_most_frequent


# 
# # cluster analysis for rating 2 - elements of cluster, wordcloud, frequency of most frequent words that are appearing
# 

# In[160]:


elem_rate2 = clustering_rating_wise(initial_df, 2)


# In[161]:


g2_elem = elem_rate2['g']
g2_elem


# In[162]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(g2_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[163]:


keyword_freq_per_cluster(g2_elem,1).to_dict()


# In[164]:


import re
g2_most_frequent = [y for y in g2_elem if re.search('minutes', y)]
g2_most_frequent


# In[165]:


r2_elem = elem_rate2['r']
r2_elem


# In[166]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(r2_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[167]:


keyword_freq_per_cluster(r2_elem,2).to_dict()


# In[168]:


import re
r2_most_frequent = [y for y in r2_elem if re.search('tamil', y)]
r2_most_frequent


# In[169]:


c2_elem = elem_rate2['c']
c2_elem


# In[170]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(c2_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[171]:


keyword_freq_per_cluster(c3_elem,3).to_dict()


# In[172]:


import re
c2_most_frequent = [y for y in c4_elem if re.search('app', y)]
c2_most_frequent


# 
# # cluster analysis for rating 1 - elements of cluster, wordcloud, frequency of most frequent words that are appearing
# 

# In[173]:


elem_rate1 = clustering_rating_wise(initial_df, 1)


# In[174]:


g1_elem = elem_rate1['g']
g1_elem


# In[175]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(g1_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[176]:


keyword_freq_per_cluster(g1_elem,1).to_dict()


# In[177]:


import re
g1_most_frequent = [y for y in g1_elem if re.search('board', y)]
g1_most_frequent


# In[178]:


r1_elem = elem_rate1['r']
r1_elem


# In[179]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(r1_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[180]:


keyword_freq_per_cluster(r1_elem,2).to_dict()


# In[181]:


import re
r1_most_frequent = [y for y in r1_elem if re.search('question', y)]
r1_most_frequent


# In[182]:


c1_elem = elem_rate1['c']
c1_elem


# In[183]:


from wordcloud import WordCloud

#text = list(eng_df['Review Text'])
long_string = ','.join(c1_elem)
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()


# In[184]:


keyword_freq_per_cluster(c1_elem,3).to_dict()


# In[185]:


import re
c1_most_frequent = [y for y in c1_elem if re.search('bad app', y)]
c1_most_frequent


# In[202]:


# m1_elem = elem_rate1['m']
# m1_elem


# In[ ]:


# from wordcloud import WordCloud

# #text = list(eng_df['Review Text'])
# long_string = ','.join(m1_elem)
# wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# wordcloud.generate(long_string)
# wordcloud.to_image()


# In[ ]:


# keyword_freq_per_cluster(m1_elem,4).to_dict()


# In[ ]:


# import re
# m1_most_frequent = [y for y in m1_elem if re.search('content', y)]
# m1_most_frequent


# 
# # cluster element extraction for each rating
# 

# In[187]:


clustering_rating_wise(initial_df, 1)


# In[188]:


clustering_rating_wise(initial_df, 2)


# In[189]:


clustering_rating_wise(initial_df, 3)


# In[190]:


clustering_rating_wise(initial_df, 4)


# In[191]:


clustering_rating_wise(initial_df, 5)


# 
# # LDA - Topic Modelling for each rating
# 

# In[192]:


from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


# In[193]:


def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    defaultPos = nltk.pos_tag(token_words) # get the POS tags from NLTK default tagger
    words_lemma = []
    
    #mapping to wordnet lemmatazer
    pos_to_wn = {"JJ": 'a',
             "JJR": 'a',
             "JJS": 'a',
             "NN": 'n', 
             "NNS": 'n', 
             "NNP": 'n', 
             "NNPS": 'n', 
             "VB": 'v',
             "VBD": 'v',
             "VBG": 'v',
             "VBN": 'v',
             "VBP": 'v',
             "VBZ": 'v',
             "RB": 'r',
            "RBR": 'r',
            "RBS": 'r'}
    for item in defaultPos:
        try:
            words_lemma.append(wordnet_lemmatizer.lemmatize(item[0],pos_to_wn[item[1]]))
        except:
            words_lemma.append(item[0])
        #words_lemma.append(lemmatizer.lemmatize(item[0],pos_to_wn(item[1])))
    return " ".join(words_lemma)
#stemSentence(text[0])


# In[194]:


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


# In[195]:


import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation as LDA
 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


# In[196]:


def LDA_Topic_Modeling_Plotting(text):
    count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
    count_data = count_vectorizer.fit_transform(text)
    plot_10_most_common_words(count_data, count_vectorizer)

    # Tweak the two parameters below
    number_topics = 4
    number_words = 8

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)

    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, number_words)


# In[197]:


rating1_text = list(initial_df[initial_df['Star Rating']==1]['Review Text'])
LDA_Topic_Modeling_Plotting(rating1_text)


# In[198]:


rating2_text = list(initial_df[initial_df['Star Rating']==2]['Review Text'])
LDA_Topic_Modeling_Plotting(rating2_text)


# In[199]:


rating3_text = list(initial_df[initial_df['Star Rating']==3]['Review Text'])
LDA_Topic_Modeling_Plotting(rating3_text)


# In[200]:


rating4_text = list(initial_df[initial_df['Star Rating']==4]['Review Text'])
LDA_Topic_Modeling_Plotting(rating4_text)


# In[201]:


rating5_text = list(initial_df[initial_df['Star Rating']==5]['Review Text'])
LDA_Topic_Modeling_Plotting(rating5_text)


# In[ ]:





# In[ ]:


def clustering_rating_upto_3(key_rate_df):
    key_data = key_rate_df.loc[key_rate_df['Star Rating'] <= 3]
    key_rate_df1 = pd.DataFrame(key_data)
#     return key_rate_df1
    ky_ls =key_rate_df1['Review Text'].tolist()
    
    frame = vectorization_of_list(ky_ls)
    frame.set_index(key_rate_df1["Review Text"], inplace = True) 
    return cluster_element_extraction(frame)


# In[205]:


cluster_upto_rate3 = clustering_rating_upto_3(initial_df)


# In[206]:


cluster_upto_rate3


# In[208]:


def clustering_rating_greater_than_3(key_rate_df):
    key_data = key_rate_df.loc[key_rate_df['Star Rating'] > 3]
    key_rate_df1 = pd.DataFrame(key_data)
#     return key_rate_df1
    ky_ls =key_rate_df1['Review Text'].tolist()
    
    frame = vectorization_of_list(ky_ls)
    frame.set_index(key_rate_df1["Review Text"], inplace = True) 
    return cluster_element_extraction(frame)


# In[209]:


cluster_greater_than_rate3 = clustering_rating_greater_than_3(initial_df)


# In[210]:


cluster_greater_than_rate3


# In[ ]:




