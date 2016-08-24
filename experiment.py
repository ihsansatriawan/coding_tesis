import pandas as pd
d = [
  (1,70399,0.988375133622),
  (1,33919,0.981573492596),
  (1,62461,0.981426807114),
  (579,1,0.983018778374),
  (745,1,0.995580488899),
  (834,1,0.980942505189)
]
e = []
for x in d:
  e.append(x[0])
  e.append(x[1])

e = list(set(e))
print e
df_new = pd.DataFrame(e, columns=['source_target'])
new_source_old = df_new.source_target.copy()

df_new.source_target = (df_new.source_target.diff() != 0).cumsum() - 1
new_ser = pd.Series(df_new.source_target.values, index=new_source_old).drop_duplicates()


df = pd.DataFrame(d, columns=['source', 'target', 'weight'])
df_beda = pd.DataFrame(d, columns=['source', 'target', 'weight'])

# print "df_new"
# print df_new
# print "df"
# print df
source_old = df.source.copy()
# target_old = df.target.copy()

# print "source_old"
# print source_old
# print "target_old"
# print target_old

# new_old = source_old.append(target_old)
# print "new_old"
# print new_old

df.source = (df.source.diff() != 0).cumsum() - 1

# #series for maping
ser = pd.Series(df.source.values, index=source_old).drop_duplicates()
print "ser"
print (ser)

df.target = df.target.mask(df.target.isin(ser), df.target.map(ser)).astype(int)

print "(df)"
print (df)

print "df_beda"
print df_beda

print "new_ser"
print new_ser

df_beda.target = df_beda.target.mask(df_beda.target.isin(new_ser), df_beda.target.map(new_ser)).astype(int)
df_beda.source = df_beda.source.mask(df_beda.source.isin(new_ser), df_beda.source.map(new_ser)).astype(int)

print "df_beda diubah"
print df_beda