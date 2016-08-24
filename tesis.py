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

def sim(user1, user2):
  # query = """
  #   select freq, total from tesis_point_user where id_pengirim=%s and id_penerima=%s
  # """

  query = """
    select freq from tesis_point_user where id_pengirim=%s and id_penerima=%s
  """
  print "user1"
  print user1

  print "user2"
  print user2
  cur.execute(query, [user1, user2])
  if cur.rowcount == 0:
    vector_user_1 = [0]
  else:  
    vector_user_1 = cur.fetchall()
  print "vector_user_1"
  print vector_user_1
  
  cur.execute(query, [user2, user1])
  if cur.rowcount == 0:
    vector_user_2 = [0]
  else:  
    vector_user_2 = cur.fetchall()
  print "vector_user_2"
  print vector_user_2
  if (vector_user_1 == [0]) & (vector_user_2 == [0]):
    return 0.0
  # cosine_similarity = 1 - spatial.distance.cosine(vector_user_1, vector_user_2)
  euclidean_distance = spatial.distance.euclidean(vector_user_1, vector_user_2)
  euclidean_similarity = 1/(1 + euclidean_distance)
  print "euclidean_similarity"
  print euclidean_similarity
  return euclidean_similarity
  # if math.isnan(cosine_similarity):
  #   return 0.0
  # return cosine_similarity

def userSimilarity():
  print "start userSimilarity"
  start_time = time.time()

  # query_pengirim = """
  #   select distinct id_pengirim from tesis_point_user
  # """

  # cur.execute(query_pengirim)
  # pengirim = cur.fetchall()  

  # query_penerima = """
  #   select distinct id_penerima from tesis_point_user
  # """

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
  user_similarity = []
  users = list(set(pengirim+penerima))
  for pair in combinations(users,2):
    if sim(pair[0],pair[1]) != 0.0:
      print pair[0][0],",",pair[1][0],",", sim(pair[0][0],pair[1][0])
      user_similarity.append((pair[0][0], pair[1][0], sim(pair[0][0],pair[1][0])))

  df = pd.DataFrame(user_similarity, columns=['source', 'target', 'weight'])
  print df
  # df.ix[df.source == "Hijup", 'source'] = 1
  # df.ix[df.target == "Hijup", 'target'] = 1

  df['source'] = df['source'].astype(int)
  df['target'] = df['target'].astype(int)

  df_without_transform = df.sort_values(['source'], ascending=[True]) #--> sort
  df_without_transform.to_csv("user_nohijup_similarity_without_transform.csv", sep=',', index=False, header=False)


  df_with_transform = df.sort_values(['source'], ascending=[True]) #--> sort
  uniq_vals = np.unique(df_with_transform[['source','target']])
  map_dict = dict(zip(uniq_vals, xrange(len(uniq_vals))))
  print "map_dict"
  print map_dict
  df_with_transform[['source','target']] = df_with_transform[['source','target']].replace(map_dict)

  
  df_with_transform.to_csv("user_nohijup_similarity_with_transform.csv", sep=',', index=False, header=False)

  df_map_dict = pd.DataFrame(list(map_dict.iteritems()),columns=['origin','transform'])
  query_delete = "DROP table map_dict;"
  cur.execute(query_delete)
  conn.commit()
  engine = create_engine('postgresql://postgres@localhost:5432/hijub_db_2016')
  df_map_dict.to_sql('map_dict', engine)

  inverse_map_dict = {v:k for k,v in map_dict.iteritems()}
  query_delete = "DROP table inverse_map_dict;"
  cur.execute(query_delete)
  conn.commit()
  df_inverse_map_dict = pd.DataFrame(list(inverse_map_dict.iteritems()),columns=['transform', 'origin'])
  df_inverse_map_dict.to_sql('inverse_map_dict', engine)


  print("--- %s seconds ---" % (time.time() - start_time))


  print "end userSimilarity"

def praProcess():
  print "start praProcess"

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

  # point_from_hijup = cur.fetchall()

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

  # point_redeem = cur.fetchall()

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

  # data_point = point_from_hijup + point_redeem + point_to_from
  
  cur.execute("""
  select
    id_pengirim,
    id_penerima,
    count(*) as freq
  from
  (
    select 
      a.id as id_pengirim, b.owner_id as id_penerima, b.point
    from 
      users a
    inner join
      point_receivedfrom_from_nohijup_2015 b
        on a.email = b.from
  ) as foo
  group by id_pengirim, id_penerima
  """)

  point_received = cur.fetchall()

  cur.execute("""
  select
    id_pengirim,
    id_penerima,
    count(*) as freq
  from
  (
    select 
      b.owner_id as id_pengirim, a.id as id_penerima, ABS(b.point)
    from 
      users a
    inner join
      point_transferto_from_2015 b
        on a.email = b.to
  ) as foo
  group by id_pengirim, id_penerima
  """)

  point_transfer = cur.fetchall()

  data_point = point_received + point_transfer
  
  # df = pd.DataFrame(data_point, columns=['id_pengirim', 'id_penerima', 'freq', 'total'])
  df = pd.DataFrame(data_point, columns=['id_pengirim', 'id_penerima', 'freq'])
  df = df.sort_values(['id_pengirim', 'id_penerima'], ascending=[True, True])
  print df

  query_delete = "DROP table tesis_point_user;"
  cur.execute(query_delete)
  conn.commit()
  engine = create_engine('postgresql://postgres@localhost:5432/hijub_db_2016')
  df.to_sql('tesis_point_user', engine)
  
  print "end praProcess"


def main(argv):

  # praProcess()
  userSimilarity()

if __name__ == "__main__":
  sys.exit(main(sys.argv))