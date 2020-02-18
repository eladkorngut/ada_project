from ROOT import gROOT,TCanvas,TH1F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df=pd.read_csv('/home/elad/Advance data analysis/Redshift_table.CSV', header=None)
df.columns=['x']
