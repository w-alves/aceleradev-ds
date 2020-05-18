#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[148]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[149]:


countries = pd.read_csv("countries.csv")


# In[150]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[151]:


def format_countries(df):
    numerical_features = df.drop(columns=['Country', 'Region'])
    categorical_features = df[['Country', 'Region']]
    
    numerical_features = numerical_features.applymap(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    categorical_features = categorical_features.applymap(lambda x: x.strip())
        
    df[numerical_features.columns] = numerical_features.values
    df[categorical_features.columns] = categorical_features.values
    
    return df


# In[152]:


countries = format_countries(countries)
countries.head()


# In[153]:


countries.dtypes


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[154]:


def q1():
    return list(countries.Region.sort_values().unique())


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[155]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    bin_pop_density = discretizer.fit_transform(countries[['Pop_density']])
        
    return int(sum(bin_pop_density[:, 0] == 8))


# ##  Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[156]:


def q3():
    encoded = pd.get_dummies(countries[['Region', 'Climate']].fillna(''))
    return int(encoded.shape[1])


# ##  Questão 4
# 
# 
# Aplique o seguinte pipeline:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o pipeline descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo pipeline (ou ColumnTransformer) ao dado abaixo. Qual o valor da variável Arable após o pipeline? Responda como um único float arredondado para três casas decimais.

# In[157]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[158]:


def q4():
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("standart_scaler", StandardScaler())
    ])

    numeric_features = countries.select_dtypes(include=['float64', 'int64'])
    num_pipeline.fit(numeric_features)
    test_country_transform = num_pipeline.transform([test_country[2:]])
    arable_transform = test_country_transform[:, numeric_features.columns.get_loc("Arable")]
    
    return round(arable_transform.item(), 3)


# ##  Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[164]:


def q5():
    quant = np.quantile(countries.Net_migration.dropna(), [0.25, 0.5, 0.75])

    iqr = quant[2]-quant[0]

    lower_outliers = [x for x in countries.Net_migration if x<quant[0]-1.5*iqr]
    upper_outliers = [x for x in countries.Net_migration if x>quant[2]+1.5*iqr]
    
    remove = False #Explicação dessa escolha na célula abaixo

    return tuple([len(lower_outliers), len(upper_outliers), remove])


# ### Por que não remover os outliers aqui?
# ---
# Trecho a seguir traduzido e adaptado do artigo [Guidelines for Removing and Handling Outliers in Data](https://statisticsbyjim.com/basics/remove-outliers/) do Jim Frost.
#     
# >Se o outlier em questão:
# * For um erro de medição ou erro de input dos dados = CORRIJA O ERRO, se possível. Caso não NÃO CONSIGA, remova.
# * Não faz parte da população que você está estudando (ou seja, propriedades ou condições incomuns) = REMOVA.
# * **É uma parte natural da população que você está estudando = NÃO REMOVA.**
#     
#     
# ---
# No caso em questão, é improvável que haja erro na medição ou input dos dados, visto que é um dataset consolidado, compilado pelo governo dos Estados Unidos. Além disso, é nítido que os outliers não são externos à população que estamos estudando, pois assim como as demais observações, se tratam de países. 
# 
# Analisando sob um olhar mais crítico, há de se fazer a pergunta **"por tantos outliers?"** 
# Bom, a coluna Net_migration trata sobre a rede de migração e, como explicado pelo geografia/demografia, pessoas migram, em linhas gerais, por _fatores repulsivos_ ou _fatores atrativos_. Então, devido a combinação específica desses fatores, é natural que existam países com **alta taxa imigratória**(ou seja, taxa líquida de migração muito positiva) e países com **alta taxa emigratória**(ou seja, taxa líquida de migração muito negativa).
# Ademais, a própria taxa de migração é sensível a outliers pois tem como grandeza _saldo de migrantes por mil habitantes_ , "favorecendo" países de baixa população.
# 

# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[165]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[166]:


def q6():
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    return int(newsgroups_counts[:, count_vectorizer.vocabulary_['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[181]:


def q7():
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(newsgroups_counts)
    newsgroups_tfidf = tfidf_transformer.transform(newsgroups_counts)
    
    tdidf_matrix = pd.DataFrame(newsgroups_tfidf.toarray(), columns=np.array(count_vectorizer.get_feature_names()))

    return float(round(tdidf_matrix['phone'].sum(), 3))

