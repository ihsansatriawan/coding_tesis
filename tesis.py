from collections import Counter
from itertools import combinations
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from scipy import spatial
from scipy.stats.stats import pearsonr  
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sqlalchemy import create_engine
import arff
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psycopg2
import pyfpgrowth
import sys
import traceback
import math
import time
style.use("ggplot")


conn = psycopg2.connect(database="hijub_db_2016", user="postgres", password="hijup-ihsan", host="127.0.0.1", port="5432")
cur = conn.cursor()

def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)

def sim(user1, user2):
  query = """
    select freq, total from tesis_point_user where id_pengirim=%s and id_penerima=%s
  """

  cur.execute(query, [user1, user2])
  if cur.rowcount == 0:
    vector_user_1 = [0,0]
  else:  
    vector_user_1 = cur.fetchall()
  # print "vector_user_1: ", vector_user_1

  cur.execute(query, [user2, user1])
  if cur.rowcount == 0:
    vector_user_2 = [0,0]
  else:  
    vector_user_2 = cur.fetchall()
  # print "vector_user_2: ", vector_user_2

  cosine_similarity = 1 - spatial.distance.cosine(vector_user_1, vector_user_2)
  if math.isnan(cosine_similarity):
    return 0.0
  return cosine_similarity

def user_pair():
  start_time = time.time()
  query = """
    select id_pengirim, id_penerima from tesis_point_user
  """

  query_pengirim = """
    select distinct id_pengirim from tesis_point_user
  """

  cur.execute(query_pengirim)
  pengirim = cur.fetchall()  

  query_penerima = """
    select distinct id_penerima from tesis_point_user
  """
  
  cur.execute(query_penerima)
  penerima = cur.fetchall()  

  users = list(set(pengirim+penerima))

  for x in users:
    for user in users:
      if sim(x[0], user[0]) != 0.0:
        print x[0],",",user[0],",", sim(x[0], user[0])
  # print("--- %s seconds ---" % (time.time() - start_time))
  # a = [1,2,3]
  # b = [3,4,5,6]
  # c = list(set(a+b))

  # cur.execute(query)
  # users = cur.fetchall()
  # for user in users:
  #   if sim(user[0], user[1]) != 0.0:
  #     print user[0], user[1], ": ", sim(user[0], user[1])




def getUserSimilarity():
  conn = psycopg2.connect(database="hijub_db_2016", user="postgres", password="hijup-ihsan", host="127.0.0.1", port="5432")
  cur = conn.cursor()

  cur.execute("""
  select
    id_pengirim,
    id_penerima,
    count(*) as freq,
    sum(point) as total
  from
  (
    select 
      a.from as id_pengirim,
      a.owner_id as id_penerima,
      ABS(a.point) as point
    from 
      point_receivedfrom_from_hijup_2015 a

  ) as foo
  group by id_pengirim, id_penerima
  """)

  point_from_hijup = cur.fetchall()

  cur.execute("""
  select
    id_pengirim,
    id_penerima,
    count(*) as freq,
    sum(point) as total
  from
  (
    select 
      a.owner_id as id_pengirim,
      a.to as id_penerima,
      ABS(a.point) as point
    from point_redeem_from_2015 a
  ) as foo
  group by id_pengirim, id_penerima
  order by freq desc
  """)

  point_redeem = cur.fetchall()

  cur.execute("""
  select
    id_pengirim,
    id_penerima,
    count(*) as freq,
    sum(point) as total
  from
  (
    select 
      a.owner_id as id_pengirim,
      b.owner_id as id_penerima,
      ABS(b.point) as point
    from 
      point_transferto_from_2015 a 
    inner join 
      point_receivedfrom_from_2015 b 
        on a.from = b.from and a.to = b.to and to_char(a.created_at, 'YYYY-MM-DD HH24:MI:SS') = to_char(b.created_at, 'YYYY-MM-DD HH24:MI:SS')
  ) as foo
  group by id_pengirim, id_penerima
  """)

  point_to_from = cur.fetchall()

  data_point = point_from_hijup + point_redeem + point_to_from
  # data = [
  #   ('1', '2', 8, 10),
  #   ('2', '1', 2, 7),
  #   ('4', '3', 3, 9),
  #   ('3', '4', 5, 6)
  # ]
  # print data
  # df = pd.DataFrame(data, columns=['id_pengirim', 'id_penerima', 'freq', 'total'])
  # print df
  # print len(data_point)
  df = pd.DataFrame(data_point, columns=['id_pengirim', 'id_penerima', 'freq', 'total'])

  query_delete = "DROP table tesis_point_user;"
  cur.execute(query_delete)
  conn.commit()
  engine = create_engine('postgresql://postgres@localhost:5432/hijub_db_2016')
  df.to_sql('tesis_point_user', engine)
  # print df
  # df = df.set_index(['id_pengirim', 'id_penerima'])
  
  # print df[['freq', 'total']]
  # pt = df.pivot(index='unique_id', columns='category_product', values='count').fillna(0)
  # print df.pivot(index='id_penerima')
  # combos = combinations(pt.index, 2)
  # results = [(a, b, 1 - spatial.distance.cosine(pt.ix[a].values, pt.ix[b].values)) for a, b in combos]

  # pt = pd.pivot_table(df, index=['id_pengirim','id_penerima'], aggfunc='sum')
  # print pt

  # pt = df.pivot(index='id_pengirim', columns='id_penerima', values='total').fillna(0)
  # print pt
  # combos = combinations(pt.index, 2)

  # results = [(a, b, 1 - spatial.distance.cosine(pt.ix[a].values, pt.ix[b].values)) for a, b in combos]
  # print results

  # csv.register_dialect(
  #   'mydialect',
  #   delimiter = ',',
  #   quotechar = '"',
  #   doublequote = True,
  #   skipinitialspace = True,
  #   lineterminator = '\r\n',
  #   quoting = csv.QUOTE_MINIMAL)

  # with open('userSimilarity.csv', 'w') as mycsvfile:
  #   thedatawriter = csv.writer(mycsvfile, dialect='mydialect')
  #   for row in results:
  #     if row[2] > 0.0:
  #       thedatawriter.writerow((row[0], row[1]))



def main(argv):

  # getUserSimilarity()
  # print sim("63879", "35365")
  user_pair()
  # for user in users:
  #   print user[0]

  # a = [1,2,3]
  # b = [3,4,5,6]
  # c = list(set(a+b))
  # for i in range(len(c)):
  #   if i != (len(c)-1):
  #     print c[i],
  #     print c[i+1]

if __name__ == "__main__":
  sys.exit(main(sys.argv))