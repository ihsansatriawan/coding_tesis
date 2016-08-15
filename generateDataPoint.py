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


conn = psycopg2.connect(database="hijub_db_2016", user="postgres", password="hijup-ihsan", host="127.0.0.1", port="5432")
cur = conn.cursor()

############################
query = """
  select a.from, a.to, ABS(a.point) from point_redeem_from_2015 a
"""

cur.execute(query)
datas = cur.fetchall()


for data in datas:
  print data[0], ",",
  print data[1]
#############################
query = """
  select a.from, a.to, a.point from point_receivedfrom_from_2015 a where detail not like 'FREE%'
"""

cur.execute(query)
datas = cur.fetchall()


for data in datas:
  print data[0], ",",
  print data[1]
#############################
# query = """
#   select a.from, a.to, a.point from point_order_from_2015 a
# """

# cur.execute(query)
# datas = cur.fetchall()


# for data in datas:
#   print data[0], ",",
#   print data[1]
#############################