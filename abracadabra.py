# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 19:21:50 2021

@author: begui
"""

import string
import numpy as np
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import seaborn as sns
import sympy
import time

chars = [char for char in string.ascii_uppercase]


word  = 'ABC'

# =============================================================================
# assert type(word) == str 
# monkey_word  = ''
# count = 1
# keys_typed = 0
# while not monkey_word == word:
#     keys_typed +=1
#     monkey_word += np.random.choice(chars)
#     print(f'Monkey {monkey_word}')
#     if word[:count] == monkey_word:
#         count += 1
#     else:
#         count = 1 
#         monkey_word = ''
# =============================================================================


def monkey(word, chars, verbose = False):
    assert type(word) == str 
    count = 0
    keys_typed = 0
    lenght = len(word)
    while not count==lenght:
        if verbose:
            time.sleep(0.5)
        keys_typed +=1
        monkey_types = np.random.choice(chars)
        if verbose:
            print(monkey_types)
        if word[count] == monkey_types: #Gets one letter right
            count+=1
        elif word[0] == monkey_types: # Gets one letter wrong, but it's the first letter of the word
            count = 1
        else:
            count = 0
        if verbose:
            print(f'   Streak:{count}')
    return keys_typed
        
        
        
            

monkey(word, ['A', 'B', 'C'], True)
def parallel_monkey(word, its, core_num, simulations, chars):
    if core_num == 0:
        sims = []
        for it in tqdm(range(its), desc='Progress'):
            sims.append(monkey(word))
        simulations.append(sims)
    else:
        simulations.append([monkey(word) for i in range(its)])
        
    
            
#results = {word:{1000:None, 5000: None, 10000: None}}


CORES = 2

#twostring = {'AA':[], 'AB':[]}

import json

with open('abracadabra.json') as infile:
    twostring = json.load(infile)

avgtimes = []
while not len(twostring['AA']) == 300000:
    start = time.time()
    print(f"Current observations:{len(twostring['AA'])}")
    for word in ('AA', 'AB'):
        print(word)
        for N in (1000,):
            #print(f'N = {N}')
            simulations = [monkey(word, chars) for i in range(N)]
            twostring[word] += simulations
    clock = time.time() - start
    avgtimes.append(clock)
    print(f'It time:{round(clock)}\n Avg: {round(np.mean(avgtimes))}')
    print(f"AB Mean: {round(np.mean(twostring['AB']))} \n AA Mean: {round(np.mean(twostring['AA']))}  ")
    with open('abracadabra.json', 'w') as fp:
        json.dump(twostring, fp)

    
import matplotlib.pyplot as plt 
for result in results:
    if result in ('AAA','AAB','ABC'):
    #plt.title('Convergence')
        for N in (1000,5000,10000):
            print(pd.Series(results[result][N]).describe())
    if result in ('AA', 'AB'):
        print(result)
        for key in results[result]:
            print(key)
            print(pd.Series(results[result][key]).describe())
        #plt.hist(results[result][N])
    #plt.legend()
    #plt.show()
        
for result in results:
    print(result)
    print(pd.Series(results[result][10000]).describe())

# analytical solution

word = 'JULIETROMEO'
right_slices = {word[i:len(word)] for i in range(len(word))  }
left_slices  = {word[0:i+1] for i in range(0,len(word)) }

str_expr = ''
repeating = []
for substring in right_slices:
    if substring in left_slices:
        repeating.append(substring)
        str_expr+= f'+ 26**{len(substring)}'

final_expr= sympy.sympify(str_expr)
print(final_expr)



# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame(twostring)
describe = round(df.describe())
print(describe.to_latex())
sns.histplot(df)
plt.savefig('abra_hist.pdf')
df['diff'] = df['AA']-df['AB']

df['diffpos']  = df['diff'].where(df['diff']>0)
df['diffneg']  = df['diff'].where(df['diff']<0)
sns.histplot(df[['diffpos', 'diffneg']])
print(df['diff'].describe())
print(df['diff'].skew())


def monkeys(word1, word2, chars):
    assert type(word) == str 
    count = {word1:0, word2:0}
    keys_typed = 0
    lenght = {word:0 for word in [word1, word2]}
    while True:
        if verbose:
            time.sleep(0.5)
        keys_typed +=1
        monkey_types = np.random.choice(chars)
        if verbose:
            print(monkey_types)
        for word in [word1, word2]:
            if count[word] == monkey_types: #Gets one letter right
                count[word]+=1
                if count[word] == lenght[word]:
                    if word == word1:
                        return 1
                    else:
                        return 0
            elif word[0] == monkey_types: # Gets one letter wrong, but it's the first letter of the word
                count[word] = 1
            else:
                count[word] = 0
            if verbose:
                print(f'   Streak:{count[word]}')
    