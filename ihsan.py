# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:28:26 2016

@author: BILKI
"""

import pandas as pd
d = [
  (1,70399,0.988375133622),
  (1,33919,0.981573492596),
  (1,62461,0.981426807114),
  (579,1,0.983018778374),
  (745,1,0.995580488899),
  (834,1,0.980942505189)
]

df = pd.DataFrame(d, columns=['source', 'target', 'weight'])
#dfclone=df 
#df1= df.source
#dfclone = df #dfsebelumdidiff
# dfclone = pd.DataFrame(d, columns=['source', 'target', 'weight'])
dfclone = df

df.source = (df.source.diff() != 0).cumsum() -1

index=0
for k in dfclone.source: 
    index2=0
    for g in df.target:
        if k==g:
            df.target[index2]=df.source[index]
        index2+=1
    index+=1
#if k==df.target.any():    
    #else :
       # print "0"
   
#if dfclone.source.any() = df.target.any()
#df.source = df.source.cumsum()-1
#df.source = df.source.diff()!=0
#df2 = df.source

#if df2.any()!=df1.any():
 #   if df2.any()!=0:
   #     df.target = 0
    
print df