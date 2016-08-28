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
import time

conn = psycopg2.connect(database="hijub_db_2016", user="postgres", password="hijup-ihsan", host="127.0.0.1", port="5432")
cur = conn.cursor()

def pct_rank_qcut(series, n):
  edges = pd.Series([float(i) / n for i in range(n + 1)])
  f = lambda x: (edges >= x).argmax()
  return series.rank(pct=1).apply(f)

def save_result_com_detection():
  tmp = []
  with open("output.csv", 'rb') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
      tmp.append([row[0], row[1]])
      print row[0],
      print row[1]

  df_id_transform = pd.DataFrame(tmp, columns=['id', 'community'])
  df_id_transform['id'] = df_id_transform['id'].astype(int)
  df_id_transform['community'] = df_id_transform['community'].astype(int)

  ############
  query = """
    select transform, origin from inverse_map_dict
  """

  cur.execute(query)
  inverse_map = cur.fetchall()
  df_inverse_map = pd.DataFrame(inverse_map, columns=['transform', 'origin']) 
  # print df_inverse_map

  print "df_id_transform"
  print df_id_transform

  dict_inverse = df_inverse_map.set_index('transform')['origin'].to_dict()
  print "dict_inverse"
  print (dict_inverse)

  # df_id_transform[['id']] = df_id_transform[['id']].replace(dict_inverse)
  df_id_origin = df_id_transform
  df_id_origin[['id']] = df_id_origin[['id']].replace(dict_inverse)
  # df_id_origin = df_id_origin.replace({'id': dict_inverse})

  print "df_id_origin"
  print df_id_origin

  query_delete = "DROP table result_community_detection;"
  cur.execute(query_delete)
  conn.commit()
  engine = create_engine('postgresql://postgres@localhost:5432/hijub_db_2016')
  df_id_origin.to_sql('result_community_detection', engine)

def get_rfm():
  start_time = time.time()
  print "masuk get_rfm"
  query="""
    select 
      user_id,
      360*date_part('year',age('2016-04-01', max(date(carts.created_at+INTERVAL'7 hour'))))+
      30*date_part('month',age('2016-04-01', max(date(carts.created_at+INTERVAL'7 hour')))) +
      date_part('day',age('2016-04-01', max(date(carts.created_at+INTERVAL'7 hour')))) as recency,
      count(*) as frequency,
      round(sum(
      CASE WHEN orders.currency='USD' AND date(orders.created_at+INTERVAL'7 hour')<='2014-09-17' THEN orders.items_price*11000
           WHEN orders.currency='USD' AND date(orders.created_at+INTERVAL'7 hour') BETWEEN '2014-09-18' AND '2015-04-25' THEN orders.items_price*12000
           WHEN orders.currency='USD' AND date(orders.created_at+INTERVAL'7 hour')>='2015-03-26' THEN orders.items_price*10000
           ELSE orders.items_price*1
      END 
      ),0) as monetary
    from 
      carts 
    INNER JOIN
          orders on orders.cart_id = carts.id
    where 
      date(carts.created_at+INTERVAL'7 hour') >= '2015-01-01' AND 
      date(carts.created_at+INTERVAL'7 hour') < '2016-04-01' AND 
      orders.state IN ('confirmed','packed','delivered') AND 
      user_id is not null AND
      user_id IN (
       select id from result_community_detection
      )
    group by 
      user_id
    order by monetary desc
  """
  cur.execute(query)
  print "done cur.execute"
  datas = cur.fetchall()
  print "done fetchall"
  print("---fetchall:  %s seconds ---" % (time.time() - start_time))
  df = pd.DataFrame(datas, columns=['userid', 'recency', 'frequency', 'monetary'])
  print df
  df['recency'] = df['recency'].astype(float)
  df['frequency'] = df['frequency'].astype(float)
  df['monetary'] = df['monetary'].astype(float)

  df['recency'] = pct_rank_qcut(df.recency, 5)
  df['frequency'] = pct_rank_qcut(df.frequency, 5)
  df['monetary'] = pct_rank_qcut(df.monetary, 5)

  query_delete = "DROP table customer_rfm;"
  cur.execute(query_delete)
  conn.commit()
  engine = create_engine('postgresql://postgres@localhost:5432/hijub_db_2016')
  df.to_sql('customer_rfm', engine)

def customer_segment():
  query = """
    select userid, recency, frequency, monetary from customer_rfm
  """

  cur.execute(query)
  datas = cur.fetchall()
  X = np.array(datas)

  df_X = pd.DataFrame(X, columns=['userid', 'recency', 'frequency', 'monetary'])
  print df_X
  range_n_clusters = [3, 4, 5, 6, 7, 8, 9]
  best_silhouette_score = 0.0
  best_n_cluster = 0
  for n in range_n_clusters:
    kmeans = KMeans(n_clusters=n, n_init=50)
    kmeans.fit(df_X[df_X.columns[1:]])

    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_X[df_X.columns[1:]], labels)
    if silhouette_avg > best_silhouette_score:
      best_silhouette_score = silhouette_avg
      best_n_cluster = n
    print "Jumlah cluster : ", n, " dengan silhouette_score : ", silhouette_avg

  print "best_silhouette_score : ", best_silhouette_score, " dengan n_clusters : ", best_n_cluster
  cluster_num = best_n_cluster

  kmeans = KMeans(n_clusters=cluster_num, n_init=50)
  kmeans.fit(df_X[df_X.columns[1:]])

  df_X['cluster'] = kmeans.fit_predict(df_X[df_X.columns[1:]])
  centroids = kmeans.cluster_centers_
  labels = kmeans.labels_

  c = Counter(labels)
  print "centroids : "
  print centroids

  for cluster_number in range(cluster_num):
    print("Cluster {} contains {} samples".format(cluster_number, c[cluster_number]))

  
  print "df_X"
  print df_X

  
  #visualize 3 dimension
  plot_df = df_X[df_X.columns[1:5]]
  color = ["b", "g", "r", "c", "m", "y", "k", "w"]
  fig = figure()
  ax = fig.gca(projection='3d')
  for i in range(len(plot_df)):
    ax.scatter(plot_df['recency'][i], plot_df['frequency'][i], plot_df['monetary'][i], c=color[plot_df['cluster'][i]], s=150)
    label = '(%d, %d, %d)' % (plot_df['recency'][i], plot_df['frequency'][i], plot_df['monetary'][i])
    ax.text(plot_df['recency'][i], plot_df['frequency'][i], plot_df['monetary'][i], label)

  ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], marker = "x", s=150, linewidths = 5, zorder = 100)
  ax.set_xlabel('Recency (R)')
  ax.set_ylabel('Frequency (F)')
  ax.set_zlabel('Monetary (M)')
  plt.show()
  
  # visualize 2 dimension
  pca_2 = PCA(2)
  plot_columns = pca_2.fit_transform(df_X[df_X.columns[1:]])

  plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=df_X["cluster"])
  plt.show()

  ##########
  query_result_community = """
    select id, community from result_community_detection
  """
  cur.execute(query_result_community)
  data_result_community = cur.fetchall()
  df_result_community = pd.DataFrame(data_result_community, columns=['id', 'community'])

  print "df_result_community"
  print df_result_community

  df_combine = pd.merge(df_X, df_result_community, left_on='userid', right_on='id')
  print df_combine
  df_X['community'] = df_combine['community']
  print df_X


  query_delete = "DROP table customer_segment;"
  cur.execute(query_delete)
  conn.commit()
  engine = create_engine('postgresql://postgres@localhost:5432/hijub_db_2016')
  df_X.to_sql('customer_segment', engine)

def main(argv):
  # save_result_com_detection()
  # get_rfm("41")
  get_rfm()
  customer_segment()

if __name__ == "__main__":
  sys.exit(main(sys.argv))