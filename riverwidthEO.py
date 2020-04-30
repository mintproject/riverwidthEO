from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
import os
import sys
import numpy as np
import glob
import pandas
import shapely
import uuid
import geopandas
import gdal,osr,ogr
import math
import time
import progressbar
import matplotlib.pyplot as plt
from keras.models import load_model

def extract_epsg(tile_name):
    '''
    calculates the EPSG number of the sentinel2 tile.

    tile_name: name of the sentinel tile. It must contain 6 characters starting with T.

    '''
    num = int(tile_name[1:3])
    z = tile_name[3]
    if ord(z)>=ord('C') and ord(z)<=ord('M'):
        epsg_base = 32700
    if ord(z)>=ord('N') and ord(z)<=ord('X'):
        epsg_base = 32600
    return int(epsg_base + num)

def read_txt_file(filename):
    with open(filename) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def rois2polygon(rois,roi_file,job_type='POINT',buf_size=0.007):

    '''
    reads the rois and creates a shapefile that contains the corresponding cells
    to be processed

    rois: a numpy array when job_type is 'POINT' or 'REGION'
          a string when the job type is 'COUNTRY'

    roi_file: absolute path of the shapefile that will contain the cells.

    job_type: 3 options
        'POINT': assumes each row of rois array to be a coordinate(lon,lat).
        'REGION:' assumes each row to be a region of interest. Selects the pre-defined
                   cells that intersect with a region.
                   (lon_min,lat_min,lon_max,lat_max)
         'COUNTRY:' a list of country names.
    buf_size: buffer distance around the points in decimal degrees.
    '''

    if job_type=='POINT':

        job_name = roi_file.split('/')[-2]
        N = rois.shape[0]
        for i in range(N):
            lon,lat = rois[i,:]
            polygon_geom = shapely.geometry.Point((lon,lat))
            # crs = {'init': 'epsg:4326'}
            if i==0:
                pf = geopandas.GeoDataFrame(index=[i], crs="EPSG:4326", geometry=[polygon_geom])
                pf['RPOINT_ID'] = job_name + '-' + str(i)
            else:
                cf = geopandas.GeoDataFrame(index=[i], crs="EPSG:4326", geometry=[polygon_geom])
                cf['RPOINT_ID'] = job_name + '-' + str(i)
                pf = pandas.concat([pf,cf])

        pf = pf.drop_duplicates()
        pf['geometry'] = pf.geometry.buffer(buf_size)
        print('Total number of cells: ' + str(pf.shape[0]))
        pf.to_file(filename=roi_file, driver="ESRI Shapefile")
        os.system('rm -f ' + roi_file[0:-4] + '.zip')
        os.system('zip -jq ' +  roi_file[0:-4] + '.zip ' +  roi_file[0:-4] + '.*')
        return
    if job_type=='REGION':
        N = rois.shape[0]
        for i in range(N):
            lon_min,lat_min,lon_max,lat_max = rois[i,:]
            if i==0:
                pf = geopandas.read_file('http://umnlcc.cs.umn.edu/tmp/global_river_points.zip',bbox=tuple([lon_min,lat_min,lon_max,lat_max]))
            else:
                cf = geopandas.read_file('http://umnlcc.cs.umn.edu/tmp/global_river_points.zip',bbox=tuple([lon_min,lat_min,lon_max,lat_max]))
                pf = pandas.concat([pf,cf])
        # print(pf.shape)
        pf['geometry'] = pf.geometry.buffer(buf_size)
        pf = pf.drop_duplicates()
        print('Total number of pre-defined cells: ' + str(pf.shape[0]))
        pf.to_file(filename=roi_file, driver="ESRI Shapefile")
        os.system('rm -f ' + roi_file[0:-4] + '.zip')
        os.system('zip -jq ' +  roi_file[0:-4] + '.zip ' +  roi_file[0:-4] + '.*')
        return


    if job_type=='COUNTRY':
        N = len(rois)
        for i in range(N):
            print('extracing pre-defined cells for: ' + rois[i])
            cf = geopandas.read_file('http://umnlcc.cs.umn.edu/tmp/countries.zip')
            cf = cf[cf['NAME_0']==rois[i]]
            bounds = cf.bounds
            # print(bounds)
            lon_min = bounds.iloc[0]['minx']
            lat_min = bounds.iloc[0]['miny']
            lon_max = bounds.iloc[0]['maxx']
            lat_max = bounds.iloc[0]['maxy']
            tf = geopandas.read_file('http://umnlcc.cs.umn.edu/tmp/global_river_points.zip',bbox=tuple([lon_min,lat_min,lon_max,lat_max]))

            if i==0:
                pf = geopandas.tools.sjoin(tf,cf,op='intersects',how='left')
                pf = pf[np.isnan(pf['index_right'].values)==0]

            sf = geopandas.tools.sjoin(tf,cf,op='intersects',how='left')
            sf = sf[np.isnan(sf['index_right'].values)==0]
            pf = pandas.concat([pf,sf])

            # print(pf.shape)
        # TODO: make the buffer size a parameter
        pf['geometry'] = pf.geometry.buffer(buf_size)
        pf = pf.drop_duplicates()
        print('Total number of pre-defined cells: ' + str(pf.shape[0]))
        pf.to_file(filename=roi_file, driver="ESRI Shapefile")
        os.system('rm -f ' + roi_file[0:-4] + '.zip')
        os.system('zip -jq ' +  roi_file[0:-4] + '.zip ' +  roi_file[0:-4] + '.*')
        return


def get_cells(rois,job_type,job_name,job_loc,buf_size):

    '''
    rois: a Nx4 numpy array, where N is the number of cells or regions.
          the 4 columns correspond to lon_min,lat_min,lon_max,lat_max

    job_type: It has two options
                - 'CELL': In this mode, each row in rois is considered as a cell.
                        The module does not use pre-defined cells.
                - 'REGION': In this mode, each row in rois is considered a region.
                          The module uses pre-defined cells with each region.
    job_name: a string that will be used to create a folder where results will be stored.


    job_loc: a writeable location where results will be stored.

    buf_size: buffer distance around the points in decimal degrees.


    '''

    job_dir = job_loc + job_name + '/'
    if os.path.isdir(job_dir)==False:
        os.mkdir(job_dir)
    else:
        print('ERROR: Folder ' + job_dir + ' already exists. Please delete the folder to avoid overwriting. Exiting.')
        sys.exit()
    roi_file = job_dir + 'rois.shp'
    rois2polygon(rois,roi_file,job_type=job_type,buf_size=buf_size)
    os.system('wget --quiet http://umnlcc.cs.umn.edu/tmp/rwmbase.zip -O ' + job_dir + 'rwmbase.zip')
    os.system('unzip -jq ' + job_dir + 'rwmbase.zip -d ' + job_dir)
    # os.system('gsutil -q cp ' + 'gs://river-box-data/CountryDatasets/ETH/init/' + '*clipped_i4* ' + job_dir)
    # os.system('gsutil -q cp ' + 'gs://river-box-data/CountryDatasets/ETH/init/' + '*min_max_i4* ' + job_dir)
    # os.system('gsutil -q cp ' + 'gs://river-box-data/CountryDatasets/ETH/init/' + 'summary_stats*.txt ' + job_dir)
    return


def CreateGeoTiffsSingleBand(data,outfile,basefile):

    '''
    converts the input data into a single band geotiff using a basefile
    '''

    rasterFormat = 'GTiff' # for now assuming output format is going to GTiff
    rasterDriver = gdal.GetDriverByName(rasterFormat)
    ds = gdal.Open(basefile)
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()

    # read the first band of the white image to extract rows and columns
    band = ds.GetRasterBand(1)
    full_xsize = band.XSize
    full_ysize = band.YSize

    mds = rasterDriver.Create(outfile,full_xsize,full_ysize,1,gdal.GDT_Byte)
    mds.SetGeoTransform(geotransform)
    mds.SetProjection(projection)

    # initializing data array and putting zero filled bands in the prediction raster
    mds.GetRasterBand(1).WriteArray(data)

    # closing datasets
    mds = None
    ds = None

def CreateGeoTiffs(data,outfile,basefile,odtype):

    '''
    converts the input data into a multi band geotiff using a basefile
    odtype: gdal type for the output file
    '''

    rasterFormat = 'GTiff' # for now assuming output format is going to GTiff
    rasterDriver = gdal.GetDriverByName(rasterFormat)
    ds = gdal.Open(basefile)
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()


    band = ds.GetRasterBand(1)
    full_xsize = band.XSize
    full_ysize = band.YSize
    ds = None
    mds = rasterDriver.Create(outfile,full_xsize,full_ysize,data.shape[2],odtype)
    mds.SetGeoTransform(geotransform)
    mds.SetProjection(projection)
    # initializing data array and putting zero filled bands in the prediction raster
    for i in range(data.shape[2]):
        mds.GetRasterBand(i+1).WriteArray(data[:,:,i])
    # closing datasets
    mds = None


def download_cell(job_name,job_loc,cell_id,start_date,end_date):

    '''
    downloads the data for a given cell.

    job_name: name of the job folder where rois file is present.
    job_loc: location of the job folder
    cell_id: id of the cell to be processed. pass '-1' to process all

    '''
    def cloud_mask(fpath):
        '''
        internal function. creates the cloud mask using s2cloudless package
        '''

        fname = fpath.split('/')[-1]

        aband_list = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12','BQA','BQA','BQA','BQA']
        cband_list = ['B01','B02','B04','B05','B08','B8A','B09','B10','B11','B12']
        ds = gdal.Open(fpath,0)
        ctr = 0
        for cband in cband_list:
            cind = aband_list.index(cband)
            cdata = ds.GetRasterBand(cind+1).ReadAsArray()
            if ctr==0:
                ndata = np.zeros((cdata.shape[0],cdata.shape[1],len(cband_list)))
            ndata[:,:,ctr] = cdata
            ctr = ctr + 1
        ndata = ndata.astype(float)
        ndata = ndata*1.0/10000
        ndata = np.expand_dims(ndata,0)
        # print(time.time()-start_time)
        cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
        cloud_matrix = cloud_detector.get_cloud_probability_maps(np.array(ndata))
        cloud_matrix = np.squeeze(cloud_matrix)
        cloud_matrix = np.round(cloud_matrix*100).astype(np.uint16)

        ctr = 0
        for cband in aband_list:
            cind = aband_list.index(cband)
            cdata = ds.GetRasterBand(cind+1).ReadAsArray()
            if ctr==0:
                odata = np.zeros((cdata.shape[0],cdata.shape[1],len(aband_list)+1))
            odata[:,:,ctr] = cdata
            ctr = ctr + 1

        odata[:,:,-1] = cloud_matrix
        odata = odata.astype(np.uint16)
        CreateGeoTiffs(odata,fpath,fpath,gdal.GDT_UInt16)

    import descarteslabs as dl
    raster_client = dl.Raster()
    driver = ogr.GetDriverByName("ESRI Shapefile")

    job_dir = job_loc + job_name + '/'
    roi_file = job_dir + 'rois.shp'

    cds = driver.Open(roi_file, 0)
    cdl = cds.GetLayer()
    if cell_id!='-1':
        cdl.SetAttributeFilter("RPOINT_ID='" + cell_id + "'")

    for cfeature in cdl:
        cgeom = cfeature.GetGeometryRef()
        minX, maxX, minY, maxY = cgeom.GetEnvelope() #bounding box of the box
        lat_min = minY
        lat_max = maxY
        lon_min = minX
        lon_max = maxX
        cur_id = cfeature.GetField('RPOINT_ID')

        # create a separate directory for each cell
        out_path = job_dir + 'Cells/' #prefix + '-' + curID + '/'
        if os.path.isdir(out_path)==False:
            os.mkdir(out_path)
        out_path = out_path + 'data-' + cur_id + '/'
        if os.path.isdir(out_path)==False:
            os.mkdir(out_path)

        # converting cell boundaries to geometry
        aoi_geometry = {'type': 'Polygon',
                    'coordinates': (((lon_min, lat_min),
                                     (lon_max, lat_min),
                                     (lon_max, lat_max),
                                     (lon_min, lat_max),
                                     (lon_min, lat_min)),)}


#["coastal-aerosol", "blue", "green", "red", "red-edge","red-edge-2","red-edge-3","nir","red-edge-4","water-vapor","cirrus","swir1","swir2","bright-mask","cirrus-cloud-mask","cloud-mask","opaque-cloud-mask"]
#bands = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12','BQA','BQA','BQA','BQA','BCM']
        scenes, geoctx = dl.scenes.search(aoi_geometry, products=["sentinel-2:L1C"],start_datetime=start_date,end_datetime=end_date,limit=None)
        print('downloading ' + str(len(scenes)) + ' images for ' + cur_id)
        bar = progressbar.ProgressBar(maxval=len(scenes), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(0,len(scenes)):
            curid = scenes[i].properties.id
            curname = scenes[i].properties.identifier
            # print(curname)
            if os.path.isfile(out_path + curname + '.tif')==True:
                continue
            raster_file=raster_client.raster(inputs=curid,bands=["coastal-aerosol", "blue", "green", "red", "red-edge","red-edge-2","red-edge-3","nir","red-edge-4","water-vapor","cirrus","swir1","swir2","bright-mask","cirrus-cloud-mask","cloud-mask","opaque-cloud-mask"],data_type='UInt16',align_pixels=True,cutline=aoi_geometry,save=True,outfile_basename=out_path + curname,output_format='GTiff',resolution=10)
            #add the cloud mask as an additional band in the end
            cloud_mask(out_path + curname + '.tif')
            bar.update(i+1)
        bar.finish()


def classify_image(tpath,init_file,s2model,ntype,suffix):

    '''
    internal function.
    tpath: absolute path of the input file.
    init_file: name of the file that contains normalization constants.
    s2model: loaded keras model image classification.
    ntype: normalization type (1: min/max scaling. 4: z-normalization).
    suffix: string to add at the end of the prediction file name.
    '''


    def read_image(imgpath,rows,cols,barr,idn,init_file):
        # print(idn)
        if imgpath[0:2]=='gs':
            imgpath = ModifyPath(imgpath)

        ds = gdal.Open(imgpath)
        delv = np.array(ds.GetRasterBand(1).ReadAsArray())
        brows = delv.shape[0]
        bcols = delv.shape[1]
        matrix_rp=np.zeros((brows,bcols,barr.shape[0]),float)
        for b in range(matrix_rp.shape[2]):
            matrix_rp[:,:,b]= np.array(ds.GetRasterBand(int(barr[b]+1)).ReadAsArray())

        pad_row = int((brows-rows)/2)
        pad_col = int((bcols-cols)/2)
        matrix_rp_center=matrix_rp[pad_row:pad_row+rows,pad_col:pad_col+cols,:]
        matrix_rp_norm=matrix_rp_center.copy()
        if idn==0: #constant based scaling
            #do noting
            a = 1
        if idn==1: # local min, max based normalization
            for b in range(matrix_rp_norm.shape[2]):
                b_max=np.max(matrix_rp_center[:,:,b])
                b_min=np.min(matrix_rp_center[:,:,b])
                # print(b,b_min,b_max)
                matrix_rp_norm[:,:,b]=(matrix_rp_center[:,:,b]-b_min)/(b_max-b_min)

        if idn==2: # local mean based normalization

            for b in range(matrix_rp_norm.shape[2]):
                b_mean = np.mean(matrix_rp_center[:,:,b])
                matrix_rp_norm[:,:,b]=matrix_rp_center[:,:,b]*1.0/b_mean

        if idn==3: # global mean based normalization
            names = read_txt_file(init_file)
            init_dir = init_file[0:init_file.rfind('/') + 1]
            dmean = np.load(init_dir + names[0])
            for b in range(matrix_rp_norm.shape[2]):
                # b_mean = np.mean(matrix_rp_center[:,:,b])
                matrix_rp_norm[:,:,b]=matrix_rp_center[:,:,b]*1.0/dmean[int(barr[b])]

        if idn==4: # global mean and std dev clip
            names = read_txt_file(init_file)
            init_dir = init_file[0:init_file.rfind('/') + 1]
            dmean = np.load(init_dir + names[0])
            dstd = np.load(init_dir + names[1])
            # print(dmean)
            # print(dstd)
            for b in range(matrix_rp_norm.shape[2]):
                # b_mean = np.mean(matrix_rp_center[:,:,b])
                temp=(matrix_rp_center[:,:,b]*1.0-dmean[int(barr[b])])*1.0/np.sqrt(dstd[int(barr[b])])
                temp = temp + 3
                temp[temp<0] = 0
                matrix_rp_norm[:,:,b] = temp

        if idn==5: #constant based scaling
            for b in range(matrix_rp_norm.shape[2]):
                temp=matrix_rp_center[:,:,b]*1.0/10000
                matrix_rp_center[:,:,b] = temp

        if idn==6: # local mean and std dev clipping based normalization

            for b in range(matrix_rp_norm.shape[2]):
                b_mean = np.mean(matrix_rp_center[:,:,b])
                b_std = np.mean(matrix_rp_center[:,:,b])

                matrix_rp_norm[:,:,b]=matrix_rp_center[:,:,b]*1.0/b_mean

        ds = None
        return matrix_rp_norm,idn



    b_arr = np.array([1,2,3,4,5,6,7,11,12])
    ds = gdal.Open(tpath,0)
    delv = ds.GetRasterBand(1).ReadAsArray()
    num_bands = ds.RasterCount
    brows = delv.shape[0]
    bcols = delv.shape[1]

    obrows = delv.shape[0]
    obcols = delv.shape[1]

    # loading the selected bands used in classification
    pred_bands_org = read_image(tpath,brows,bcols,b_arr,ntype,init_file)[0]
    pred_bands_org_no_norm = read_image(tpath,brows,bcols,b_arr,0,init_file)[0]

    # loading the cloud mask stored in the last band
    cloud_matrix = ds.GetRasterBand(num_bands).ReadAsArray()>40
    cloud_matrix = cloud_matrix.astype(int)


    org_data_mask = np.ones((brows,bcols))
    # minimum size of the image required is 96x96
    # padding the image with zeros if dimensions are smaller
    if brows<96:
        temp = pred_bands_org.copy()
        ctemp = cloud_matrix.copy()
        nof = pred_bands_org.shape[2]

        pred_bands_org = np.zeros((96,bcols,nof))
        pad_ind = int((96-brows)/2)
        for i in range(nof):
            pred_bands_org[pad_ind:pad_ind+brows,:,i] = temp[:,:,i]


        temp = pred_bands_org_no_norm.copy()
        nof = pred_bands_org_no_norm.shape[2]

        pred_bands_org_no_norm = np.zeros((96,bcols,nof))
        pad_ind = int((96-brows)/2)
        for i in range(nof):
            pred_bands_org_no_norm[pad_ind:pad_ind+brows,:,i] = temp[:,:,i]


        cloud_matrix = np.zeros((96,bcols))+2
        pad_ind = int((96-brows)/2)
        cloud_matrix[pad_ind:pad_ind+brows,:] = ctemp[:,:]

        temp = org_data_mask.copy()
        org_data_mask = np.zeros((96,bcols))
        pad_ind = int((96-brows)/2)
        org_data_mask[pad_ind:pad_ind+brows,:] = temp[:,:]

        brows = 96

    if bcols<96:

        temp = pred_bands_org.copy()
        ctemp = cloud_matrix.copy()
        nof = pred_bands_org.shape[2]

        pred_bands_org = np.zeros((brows,96,nof))
        pad_ind = int((96-bcols)/2)
        for i in range(nof):
            pred_bands_org[:,pad_ind:pad_ind+bcols,i] = temp[:,:,i]


        temp = pred_bands_org_no_norm.copy()
        # ctemp = cloud_matrix.copy()
        nof = pred_bands_org_no_norm.shape[2]

        pred_bands_org_no_norm = np.zeros((brows,96,nof))
        pad_ind = int((96-bcols)/2)
        for i in range(nof):
            pred_bands_org_no_norm[:,pad_ind:pad_ind+bcols,i] = temp[:,:,i]

        cloud_matrix = np.zeros((brows,96))+2
        pad_ind = int((96-bcols)/2)
        cloud_matrix[:,pad_ind:pad_ind+bcols] = ctemp[:,:]

        temp = org_data_mask.copy()
        org_data_mask = np.zeros((brows,96))
        pad_ind = int((96-bcols)/2)
        org_data_mask[:,pad_ind:pad_ind+bcols] = temp[:,:]

        bcols = 96


    # creating starting indices for different 96x96 blocks in the data
    step = 48 # overlapping pixels between blocks
    rarr = np.arange(0,brows,step)
    rarr = rarr[0:-1*int(96/step)]
    rarr = np.append(rarr,brows-96)
    carr = np.arange(0,bcols,step)
    carr = carr[0:-int(96/step)]
    carr = np.append(carr,bcols-96)

    pred_labels=np.zeros((brows,bcols),float)
    pred_deno=np.zeros((brows,bcols),float)
    ppad = 0
    # reshaping all blocks into a 4 dimensional array
    X = np.zeros((len(rarr)*len(carr),96,96,9))
    ctr = 0
    for r in rarr:
        for c in carr:
            pred_bands=pred_bands_org[r:r+96,c:c+96,:].copy()
            X[ctr,:,:,:] = pred_bands
            ctr = ctr + 1
    Y=s2model.predict_on_batch(X)
    # print(Y.shape)
    ctr = 0
    for r in rarr:
        for c in carr:
            CNN_pred_matrix = np.squeeze(Y[ctr,:,:,:])
            pred_labels[r+ppad:r+96-ppad,c+ppad:c+96-ppad]=pred_labels[r+ppad:r+96-ppad,c+ppad:c+96-ppad]+CNN_pred_matrix[ppad:96-ppad,ppad:96-ppad]
            pred_deno[r+ppad:r+96-ppad,c+ppad:c+96-ppad]=pred_deno[r+ppad:r+96-ppad,c+ppad:c+96-ppad] + 1
            ctr = ctr + 1

    pad_inds = np.logical_or(pred_deno==0,np.sum(pred_bands_org_no_norm>10,axis=2)==0)
    # print(8)
    pred_deno[pred_deno==0] = 1
    pred_labels = pred_labels*1.0/pred_deno
    pred_labels = np.round(pred_labels*100.0).astype(np.uint8)
    cloud_matrix[pad_inds] = 2


    sx = np.where(np.sum(org_data_mask,axis=1)>0)[0][0]
    ex = np.where(np.sum(org_data_mask,axis=1)>0)[0][-1]
    sy = np.where(np.sum(org_data_mask,axis=0)>0)[0][0]
    ey = np.where(np.sum(org_data_mask,axis=0)>0)[0][-1]


    output = np.zeros((obrows,obcols,2)).astype(np.uint8)
    output[:,:,0] = pred_labels[sx:ex+1,sy:ey+1]
    output[:,:,1] = cloud_matrix[sx:ex+1,sy:ey+1]

    CreateGeoTiffs(output,tpath[0:-4] + suffix ,tpath,gdal.GDT_Byte)
    return


def prepare_cell_movie(lpath,out_dir,flist,info_arr):

    '''
    prepares a movie that shows raw imagery with labels

    lpath: absolute path data for the cell is stored.
    out_dir: location where the movie will be stored
    flist: list of image names
    info_arr: numpy array generated by label_correction function that contains the area information

    '''

    ID = lpath.split('/')[-2]

    png_dir = lpath + 'pngs/'
    if os.path.isdir(png_dir)==False:
        os.mkdir(png_dir)
    else:
        os.system('rm -rf ' + png_dir + '/*.png')

    # removing timesteps that were skipped due to clouds or empty data
    bad_inds = np.sum(info_arr,axis=0)==-3
    info_arr = info_arr[:,bad_inds==0]
    T = np.sum(bad_inds==0)
    ctr = 0
    # preparing png for each image
    for fpath in flist:
        iname = fpath[0:-9] + '.tif'
        cname = fpath[0:-8] + 'corr.tif'
        cdate = int(fpath[-39:-31])

        pds = gdal.Open(fpath,0)
        clouds = pds.GetRasterBand(2).ReadAsArray().astype(float)
        rows,cols = clouds.shape

        rgb = np.zeros((rows,cols,3))+1
        rds = gdal.Open(iname,0)
        rgb[:,:,0] = rds.GetRasterBand(8).ReadAsArray()
        rgb[:,:,1] = rds.GetRasterBand(4).ReadAsArray()
        rgb[:,:,2] = rds.GetRasterBand(3).ReadAsArray()

        rgb_map_scaled = rgb.copy().astype(float)
        for j in range(3):
            rgb_map_scaled[:,:,j] = (rgb_map_scaled[:,:,j] - np.min(rgb_map_scaled[:,:,j]))*1.0/(np.max(rgb_map_scaled[:,:,j]) - np.min(rgb_map_scaled[:,:,j]))

        for j in range(3):
            temp = rgb_map_scaled[:,:,j]
            temp[clouds>0] = 1
            rgb_map_scaled[:,:,j] = temp
        cds = gdal.Open(cname,0)
        corr_labels = cds.GetRasterBand(1).ReadAsArray()
        corr_labels[0,0] = 0
        corr_labels[0,1] = 1


        f = plt.figure(figsize=(14, 8))
        ax1= f.add_subplot(2,2,1)
        ax2= f.add_subplot(2,2,2)
        ax3= f.add_subplot(2,1,2)

        ax1.imshow(rgb_map_scaled)
        ax1.title.set_text('Image Composite')

        ax2.imshow(corr_labels,cmap='Blues')
        ax2.title.set_text('Land/Water Mask')

        ax3.plot(np.arange(T),info_arr[2,:],'-ob',markersize=2)
        ax3.xaxis.grid(True)
        ax3.plot([ctr,ctr],[0,np.nanmax(info_arr[2,:])],'--k')
        ax3.title.set_text(str(int(cdate)))

        plt.tight_layout()
        f.savefig(png_dir + str(ctr).zfill(5) + '.png')
        plt.close()
        ctr = ctr + 1
    # converting pngs into a movie
    os.system('ffmpeg -hide_banner -loglevel warning -r 1 -i ' + png_dir + '%05d.png -vcodec libx264 -y -pix_fmt yuv420p ' + out_dir + ID + '.mp4')

def label_correction(lpath):

    '''

    removes cloudy images and corrects labeling errors using physical principles

    lpath: absolute path data for the cell is stored.

    '''

    # sorting images dy date
    tlist = glob.glob(lpath + 'S2*_zprd.tif')
    dates = []
    for tpath in tlist:
        dates.append(int(tpath[-39:-31]))
    N = len(dates)
    dargs = np.argsort(np.array(dates))
    nlist = []
    for j in range(N):
        nlist.append(tlist[dargs[j]])


    #estimating relative elevation using multi-temporal classification maps
    # elevation of a pixel is inversely proportional to # of times it is labelled as water
    for j in range(N):
        ds = gdal.Open(nlist[j],0)
        pred = ds.GetRasterBand(1).ReadAsArray().astype(float)
        cloud = ds.GetRasterBand(2).ReadAsArray().astype(float)
        pred[cloud==1] = np.nan
        pred[cloud==2] = np.nan
        if j ==0:
            rows,cols = pred.shape
            maps = np.zeros((rows,cols,N))
        maps[:,:,j] = pred

    rank_labels = np.nanmean(maps,axis=2)
    rank_labels[np.isnan(rank_labels)] = 0

    # correcting labels using relative elevation ordering
    ix = np.argsort(rank_labels.flatten())
    flist = []
    info_arr = np.zeros((3,N))-1
    for j in range(N):

        # reading labels
        ds = gdal.Open(nlist[j],0)
        pred_labels = ds.GetRasterBand(1).ReadAsArray().astype(float)*1.0/100.0
        lds = gdal.Open(nlist[j][0:-8] + 'lprd.tif',0)
        lprd = lds.GetRasterBand(1).ReadAsArray().astype(float)*1.0/100.0
        pred_labels[pred_labels>0.2] = lprd[pred_labels>0.2]/0.6

        cloud = ds.GetRasterBand(2).ReadAsArray().astype(float)
        pred_labels[cloud==1] = 0.5
        pred_labels[cloud==2] = 0.5

        rows,cols = pred_labels.shape
        # skipping image if more than 30 % pixels have no data or
        # more than 10 % of the data is flagged as cloudy
        if np.sum(cloud==2)*100.0/(rows*cols)>30:
            continue
        if np.sum(cloud==1)*100.0/(rows*cols)>10:
            continue

        flist.append(nlist[j]) # storing valid image names
        pred_labels = pred_labels.flatten()
        ord_labels = pred_labels[ix]
        ord_labels = np.append(0,ord_labels)
        sum1 = np.cumsum(ord_labels)
        sum2 = np.cumsum(1 - ord_labels)
        score = sum1 + np.sum(sum2) - sum2
        ind = np.argmin(score)

        new_labels = ord_labels.copy()
        new_labels[:] = 0
        new_labels[0:ind] = 0
        new_labels[ind:] = 1
        new_labels = new_labels[1:]
        flabels = pred_labels.copy()
        flabels[ix] = new_labels
        flabels = np.reshape(flabels,(rows,cols)).astype(np.uint8)
        # storing corrected labels
        CreateGeoTiffsSingleBand(flabels,nlist[j][0:-8] + 'corr.tif',nlist[j])

        # binarzing labels for area calculation
        pred_labels = np.reshape(pred_labels,(rows,cols))
        pred_labels[pred_labels>0.5] = 1
        pred_labels[pred_labels<=0.5] = 0
        pred_labels[cloud==1] = 2
        pred_labels[cloud==2] = 2

        # saving area information
        info_arr[0,j] = np.sum(pred_labels==2)*100.0/(rows*cols) # percentage of empty pixels
        info_arr[2,j] = np.sum(flabels) # area (number of water pixels)
        info_arr[1,j] = np.sum(pred_labels!=flabels)*100.0/(rows*cols) # percentage of corrected labels


    return info_arr.astype(int),nlist,flist

def prepare_cell_timeseries(job_name,job_loc,cell_id):

    '''
    creates river area timeseries csv and a movie to show river surface changes

    job_name: name of the job folder.
    job_loc: location where job folder will be created.
    cell_id: id of the cell as string.

    '''

    job_dir = job_loc + job_name + '/'

    out_dir = job_dir + 'outputs/' # sub-folder where the csv and movie will be stored
    if os.path.isdir(out_dir)==False:
        os.mkdir(out_dir)

    cpath = job_dir + '/Cells/data-' + cell_id + '/'
    cname = cpath.split('/')[-2]

    # removing cloudy images and applying physics based correction to fix errors
    info_arr,tlist,slist = label_correction(cpath)
    print('Using ' + str(len(slist)) + ' images out of ' + str(len(tlist)) + ' due to clouds...')

    prepare_cell_movie(cpath,out_dir,slist,info_arr.copy()) # preparing the movie

    # preparing the csv
    fid = open(out_dir + 'data-' + cell_id + '.csv','w')
    fid.write('date,area\n')
    for i in range(len(tlist)):
        cpath = tlist[i]
        cname = cpath.split('/')[-1]
        pieces = cname.split('_')
        for p in pieces:
            if len(p)==15:
                cdate = p
                break
        cyear = cdate[0:4]
        cmonth = cdate[4:6]
        cday = cdate[6:8]
        chour = cdate[9:11]
        cminute = cdate[11:13]
        csecond = cdate[13:15]
        cdate = cyear + '-' + cmonth + '-' + cday + 'T' + chour + ':' + cminute + ':' + csecond
        carea = info_arr[2,i]
        if carea>0:
            carea = np.round(carea*0.0001,4)
        fid.write(cdate + ',' + str(carea) + '\n')
    fid.close()

def classify_cell(job_name,job_loc,cell_id,start_date,end_date):

    '''
    create land/water mask for each image of a  given cell.

    job_name: name of the job folder where rois file is present.
    job_loc: location of the job folder
    cell_id: id of the cell to be processed. pass '-1' to process all

    '''

    job_dir = job_loc + job_name + '/'
    driver = ogr.GetDriverByName("ESRI Shapefile")
    s2model_znorm=load_model(job_dir + 'random_s2img_s2lab_exp22_global_znorm_clipped_i4.hdf')
    s2model_lminmax=load_model(job_dir + 'random_s2img_s2lab_exp22_local_min_max_i4.hdf')
    roi_file = job_dir + 'rois.shp'


    cds = driver.Open(roi_file, 0)
    cdl = cds.GetLayer()

    # subsetting to the specific cell requested
    if cell_id!='-1':
        cdl.SetAttributeFilter("RPOINT_ID='" + cell_id + "'")

    for cfeature in cdl:
        cur_id = cfeature.GetField('RPOINT_ID')
        download_cell(job_name,job_loc,cur_id,start_date,end_date)
        cpath = job_dir + '/Cells/data-' + cur_id + '/'
        tlist = glob.glob(cpath + '*.tif')
        print('starting classification process for ' + cur_id)
        bar = progressbar.ProgressBar(maxval=len(tlist), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(len(tlist)):
            tpath = tlist[i]
            if 'prd.tif' in tpath or 'corr.tif' in tpath:
                continue
            classify_image(tpath,job_dir + 'summary_stats_znorm_model.txt',s2model_znorm,4,'_zprd.tif')
            classify_image(tpath,job_dir + 'summary_stats_lminmax_model.txt',s2model_lminmax,1,'_lprd.tif')
            bar.update(i+1)
        bar.finish()
        print('preparing timeseries csv and movie for ' + cur_id)
        prepare_cell_timeseries(job_name,job_loc,cur_id)
    return
