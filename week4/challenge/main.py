#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as sct
from statsmodels.distributions.empirical_distribution import ECDF


# ## Parte 1

# ### _Setup_ da parte 1

# In[2]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[3]:


# Sua análise da parte 1 começa aqui.
dataframe.head()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[15]:


def q1():
    q_norm = np.quantile(dataframe['normal'], [0.25, 0.50, 0.75])
    q_binom = np.quantile(dataframe['binomial'], [0.25, 0.50, 0.75])
    diff_quantiles = [round(q_norm-q_binom, 3) for q_norm, q_binom in zip(q_norm, q_binom)]
    
    return tuple(diff_quantiles)


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[5]:


def q2():
    norm_ecdf = ECDF(dataframe.normal)
    norm_mean = dataframe.normal.mean()
    norm_std = dataframe.normal.std()

    ecdf = norm_ecdf(norm_mean + norm_std) - norm_ecdf(norm_mean - norm_std)
    
    return round(ecdf, 3).item()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[17]:


def q3():
    binom_mean = dataframe.binomial.mean()
    binom_var = dataframe.binomial.var()

    norm_mean = dataframe.normal.mean()
    norm_var = dataframe.normal.var()

    return (round(binom_mean-norm_mean, 3), round(binom_var-norm_var, 3))


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[7]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[8]:


# Sua análise da parte 2 começa aqui.
stars.head()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[9]:


def q4():
    notpulsar_mean_profile = stars.query('target == 0').mean_profile
    notpulsar_mean_profile_scaled = (notpulsar_mean_profile - notpulsar_mean_profile.mean())/notpulsar_mean_profile.std()
    
    norm_quantile_func = sct.norm.ppf([0.8, 0.9, 0.95])
    ecdf = ECDF(notpulsar_mean_profile_scaled)
    
    probabilities = ecdf(norm_quantile_func).round(3)
    
    return (probabilities[0], probabilities[1], probabilities[2])


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[11]:


def q5():
    notpulsar_mean_profile = stars.query('target == 0').mean_profile
    notpulsar_mean_profile_scaled = (notpulsar_mean_profile - notpulsar_mean_profile.mean())/notpulsar_mean_profile.std() 
    
    scaled_quantiles = np.quantile(notpulsar_mean_profile_scaled, [0.25, 0.50, 0.75])
    norm_quantiles = sct.norm.ppf([0.25, 0.50, 0.75], loc=0, scale=1)
    
    diff_scaled_norm_quantiles = [round(scaled-norm, 3) for scaled, norm in zip(scaled_quantiles, norm_quantiles)]

    return tuple(diff_scaled_norm_quantiles)


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
