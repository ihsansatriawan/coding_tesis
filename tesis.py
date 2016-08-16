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
style.use("ggplot")

def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)

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
  print len(point_from_hijup)

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
  print len(point_redeem)

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
  print len(point_to_from)

  data_point = point_from_hijup + point_redeem + point_to_from
  print len(data_point)
  df = pd.DataFrame(data_point, columns=['id_pengirim', 'id_penerima', 'freq', 'total'])
  print df
  # df = df.sort_values(['unique_id'], ascending=[True])

  # pt = df.pivot(index='unique_id', columns='category_product', values='count').fillna(0)
  # combos = combinations(pt.index, 2)
  # results = [(a, b, 1 - spatial.distance.cosine(pt.ix[a].values, pt.ix[b].values)) for a, b in combos]



def main(argv):

  getUserSimilarity()

if __name__ == "__main__":
  sys.exit(main(sys.argv))