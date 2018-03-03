

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_scorebc


df = pd.read_json('./data/train.json')[['longitude','latitude','interest_level','price']]

df['response'] = 0
df.loc[df.interest_level == 'medium','response'] = 0.5
df.loc[df.interest_level == 'high', 'response'] = 1


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.
    
    """
    lon1, lat1, lon2, lat2 = map(np.radians,[lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(15,6))
print('Length before removing outliers', len(df))
ax[0].plot(df.longitude, df.latitude, '.')
ax[0].set_title('Before outlier removal')
ax[0].set_xlabel('Longitude')
ax[0].set_ylabel('Latitude')

ny_lat = 40.785091
ny_lon = -73.968285
to = 95
# simplified distance
df['distance_from_center'] = haversine_np(ny_lon,ny_lat,df['longitude'],df['latitude'])
print('Threshold {}% :{:3.5f}'.format(to,np.percentile(df.distance_from_center,to)))
df = df[df['distance_from_center'] < np.percentile(df.distance_from_center,to)]
print('Length after removing outliers', len(df))

ax[1].plot(df.longitude, df.latitude, '.')
ax[1].set_title('After outlier removal')
ax[1].set_xlabel('longitude')
ax[1].set_ylabel('Latitude')

fig, ax = plt.subplots(4,1,figsize=(9,30))
for ix, ncomp in enumerate([5, 10, 20, 40]):
    r = KMeans(ncomp,random_state = 1)
    temp = df[['longitude','latitude']].copy()
    # Z-Score Normalization of (longitude, latitude) before K-means
    temp['longitude'] = (temp['longitude'] - temp['longitude'].mean()) / temp['longitude'].std()
    temp['latitude'] = (temp['latitude'] - temp['latitude'].mean()) / temp['latitude'].std()
    # Fit K-Means and get labels
    r.fit(temp[['longitude','latitude']])
    df['labels'] = r.labels_
    # Plot results
    cols = sns.color_palette("Set2", n_colors = ncomp,desat =0.5)
    cl = [cols[i] for i in r.labels_]
    area = 12
    ax[ix].scatter(df.longitude, df.latitude, s = area, c= cl, alpha = 0.5)
    ax[ix].set_title("Number of Components: " + str(ncomp))
    ax[ix].set_xlabel('Longitude')
    ax[ix].set_ylabel('Latitude')
    # Show aggregated volume and interest at each neighborhood
    x = df.groupby('labels')[['longitude','latitude','response']].mean().sort_values(['response'])
    x = pd.concat([x, df['labels'].value_counts()], axis = 1).sort_values(['response'])
    print(x)
    

print(df.groupby('labels')['longitude','latitude'].agg(['count']))
    