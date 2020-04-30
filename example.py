import dlab as rwm
import numpy as np
import os
import time

#-------------------------parameters--------------------------------

# what type of roi input
'''
three different types of input options for ROI
'POINT': assumes each row of rois array to be a coordinate (lon,lat).
'REGION:' assumes each row to be a region of interest. Auotmatically selects
          pre-defined points that are within the region.
           (lon_min,lat_min,lon_max,lat_max)
 'COUNTRY:' a list of country names.
 '''
job_type = 'COUNTRY' # other options are REGION and COUNTRY
# rois = np.array([[43,5],[43,6],[44,7]]) # POINT array shape should be (NX1)
# rois= np.array([[37.480343,7.535860,41.052997,10.332098]]) # REGION array shape should be (Nx4)
rois = ['Ethiopia'] # COUNTRY

# folder where the intermediate files and output files will be stored
job_name = 'rwmtest3' # no special characters and no underscores
job_loc = os.getcwd() + '/' # location where job folder shoud be created

start_date = '2020-03-01'
end_date = '2020-12-31'

# how much area around the points to be analyzed. The units are in decimal degrees.
#0.006 roughly corresponds to 1.3kmx1.3km at the equator.
buf_size = 0.007 # it cannot be smaller than 0.006 because machine learning algorithms require a minimum image size


start_time = time.time()
# this convert the rois input into a shapefile and it is stored in the job folder with the name rois.shp
rwm.get_cells(rois,job_type,job_name,job_loc,buf_size)
print('Time taken to get cell information: ' + str(time.time()-start_time) + ' seconds.')

cell_id = '1050883510-7366'  # provide '-1' to loop through all cells in the rois.shp file
start_time = time.time()
rwm.classify_cell(job_name,job_loc,cell_id,start_date,end_date)
print('Time taken to process the cell: ' + str(time.time()-start_time) + ' seconds.')
#NOTE: run this code in parallel to process multiple cells simulatenously
