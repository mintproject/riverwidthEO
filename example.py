import riverwidthEO as rwm
import numpy as np
import os
import time

#-------------------------parameters--------------------------------

# Type of roi (region of interest) input
'''

three different types of input options for ROI
'POINT': assumes each row of rois array to be a coordinate (lon,lat).
Using this option, the user can provide any set of points on the river acrosss the globe to analyze.

'REGION:' assumes each row to be a region of interest. Auotmatically selects
          pre-defined points that are within the region.
           (lon_min,lat_min,lon_max,lat_max)
Using this option, user can process a set of pre-defined points on river in any region across the globe.
The shapefile with the pre-defined points is available here: https://data.mint.isi.edu/files/remote-sensing/global_river_points.zip
The shapefile contains 3,576,396 points on the rivers across the globe.
Use any GIS software such has QGIS or AcrGIS to visualize the shapefile.

 'COUNTRY:' a list of country names.
Using this option, user can process pre-defined points for any set of countries.

 '''


job_type = 'COUNTRY' # possible options: 'POINT','REGION','COUNTRY'
# provide region or points of interest

# rois = np.array([[43,5],[43,6],[44,7]]) # an array of points when job_type='POINT'. array shape should be (NX1)
# rois= np.array([[37.480343,7.535860,41.052997,10.332098]]) # an array of bounding boxes when job_type='REGION'. array shape should be (Nx4)
rois = ['Ethiopia'] # a list of country names when job_type='COUNTRY'.
# use this file https://data.mint.isi.edu/files/remote-sensing/countries.zip to find valid country names

# folder where the intermediate files and output files will be stored
job_name = 'rwmtest3' # no special characters and no underscores
job_loc = os.getcwd() + '/' # location where job folder shoud be created

start_date = '2019-03-01'
end_date = '2020-12-31'

# how much area around the points to be analyzed. The units are in decimal degrees.
#0.006 roughly corresponds to 1.3kmx1.3km at the equator.
buf_size = 0.007 # it cannot be smaller than 0.006 because machine learning algorithms require a minimum image size


start_time = time.time()
# this convert the rois input into a shapefile and it is stored in the job folder with the name rois.shp
#rwm.get_cells(rois,job_type,job_name,job_loc,buf_size)
print('Time taken to get cell information: ' + str(time.time()-start_time) + ' seconds.')

# provide a cell id
# Use any GIS software such as QGIS or AcrGIS to visualize the rois.shp file in the job folder.
# Each cell has an id associated with it in the rois.shp file.
# provide '-1' to process all cells in the rois.shp file
cell_id = '1050883510-7366'
start_time = time.time()
rwm.classify_cell(job_name,job_loc,cell_id,start_date,end_date)
print('Time taken to process the cell: ' + str(time.time()-start_time) + ' seconds.')
