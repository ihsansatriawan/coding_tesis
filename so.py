import pandas as pd
import numpy as np
d = [
  (1,70399,0.988375133622),
  (1,33919,0.981573492596),
  (1,62461,0.981426807114),
  (579,1,0.983018778374),
  (745,1,0.995580488899),
  (834,1,0.980942505189)
]
# e = []
# for x in d:
#   e.append(x[0])
#   e.append(x[1])

# e = list(set(e))
# print e
# df_new = pd.DataFrame(e, columns=['source_target']).sort_values(['source_target'], ascending=[True]) #--> sort

# new_source_old = df_new.source_target.copy()

# df_new.source_target = (df_new.source_target.diff() != 0).cumsum() - 1
# new_ser = pd.Series(df_new.source_target.values, index=new_source_old).drop_duplicates()

df_beda = pd.DataFrame(d, columns=['source', 'target', 'weight'])

print "df_beda"
print df_beda

# print "new_ser"
# print new_ser

# print "new_ser dict"
# new_ser_dict = new_ser.to_dict()
# print new_ser_dict
# df_new_ser_dict = pd.DataFrame.from_dict(new_ser_dict, columns=['Date', 'DateValue'])
# print "df_new_ser_dict"
# print df_new_ser_dict


# df_new_ser = pd.DataFrame({'origin':new_ser.index, 'transform':new_ser.values})
# print "df_new_ser"
# print df_new_ser

# df_beda.target = df_beda.target.mask(df_beda.target.isin(new_ser), df_beda.target.map(new_ser)).astype(int)
# df_beda.source = df_beda.source.mask(df_beda.source.isin(new_ser), df_beda.source.map(new_ser)).astype(int)

uniq_vals = np.unique(df_beda[['source','target']])
map_dict = dict(zip(uniq_vals, xrange(len(uniq_vals))))
print "map_dict"
print map_dict
df_beda[['source','target']] = df_beda[['source','target']].replace(map_dict)

# df_beda = pd.merge(df_beda, df_new_ser, left_on='source', right_on='origin')

print "df_beda"
print df_beda.dtypes

inverse_map_dict = {v:k for k,v in map_dict.iteritems()}
print "inverse_new_ser_dict"
print inverse_map_dict

df_beda[['source','target']] = df_beda[['source','target']].replace(inverse_map_dict)
print "df_beda"
print df_beda

# df_map_dict = pd.DataFrame(list(map_dict.iteritems()),columns=['origin','transform'])

# print "df_map_dict"
# print df_map_dict