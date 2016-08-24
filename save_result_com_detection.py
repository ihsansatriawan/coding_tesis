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

def first():
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
  

def main(argv):

  first()

if __name__ == "__main__":
  sys.exit(main(sys.argv))