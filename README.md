# riverwidthEO
A python package that processes river segments using satellite imagery and machine learning to create surface area estimates (delivered in a csv file).

The package enables the user to process any set of user-defined points on rivers or process any of the pre-defined 3,576,396 points on rivers across the globe. Please see the [example](https://github.com/mintproject/riverwidthEO/blob/master/example.py) script for information on how to use the package.

### Dataset
River Segment Surface Area Dataset provides pre-computed surface area variations for 8,710 river segments in Ethiopia using Sentinel2 imagery from 2015 till 2010. This dataset is availabe through the MINT DataCatalog, and can be downloaded using the jupyter [notebook](https://github.com/mintproject/riverwidthEO/blob/master/MINT_DataCatalog_riverwidthEO.ipynb) (available with the package) as well.

### Background
<div style="text-align: justify">In many earth science applications, calibration of physical models is a key challenge because ground observations are very scarce or completely absent in most regions. For example, hydrological models simulate the flow of water in a basin using physical principles, but necessarily contain numerous parameters (e.g., soil conductivity at different grid points) whose values need to be calibrated for each study region with the help of observations. The most commonly used observation is discharge (volume per second) estimates that are available through ground stations. These stations are costly to install and maintain, and thus are limited in number.  This paucity (or complete absence) of observation data often leads to poorly calibrated models that provide incorrect predictions or have high uncertainty in practice.

Our approach is to provide this much needed calibration data using novel machine learning techniques and multi-temporal satellite imagery that is available freely from Earth Observing satellite based sensors such as Sentinel and Landsat. The latest version (version 1.0) of the package uses descarteslabs API to download Sentinel-2 imagery of any given river segment. The multi-spectral imagery is then converted into land/water maps using CNN based deep learning techniques. The area variations thus obtained can be used to constraint hydrological models. Watch this [<ins>video</ins>](http://umnlcc.cs.umn.edu/tmp/data-1050883510-7366.mp4) to see surface area variations of a river segment in Ethiopia.</div>




### Installation

##### Docker
Use the Dockerfile to setup the docker image -
```
docker build -f Dockerfile -t <image_tag> .
```
Use the following command to use the docker to run the script -
```
sudo docker run -v <path_of_local_directory>:<docker_mount_path> -it <image_tag>
```
Set the descarteslabs API client and secret when you are in the docker image -
```
export DESCARTESLABS_CLIENT_ID=...
DESCARTESLABS_CLIENT_SECRET=...
```

##### Anaconda
Install anaconda if it is currently not installed -
```
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
sha256sum Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
source /home/khand035/.bashrc
```
set up the conda environment -
```
conda create --yes -n rweo numpy pandas tensorflow keras gdal shapely scikit-image fiona geopandas
source activate rweo
pip install s2cloudless
pip install progressbar
pip install descarteslabs
```

##### Descarteslabs API
setup the client id and secret -

```
export DESCARTESLABS_CLIENT_ID=...
export DESCARTESLABS_CLIENT_SECRET=...
```

### Methodology
<div style="text-align: justify">The package uses semantic segmentation based deep neural network architecture to convert a satellite image into a land/water mask. First, a CNN based auto-encoder network was pre-trained using 11,000 unlabeled images taken from rivers around the globe. The pre-trained network weights were then used to fine-tune a semantic segmentation network using 2,900 labeled images. To make the algorithms more robust to atmospheric disturbances, land/water masks were updated using physical principles. Specifically, pixels of a river segment do not change independently but are related to each other through hydraulic and bathymetric constraints. These constraints can be used to identify and correct physical inconsistencies in land/water labels obtained from machine learning algorithms. </div>

##### Limitations
The current version of the algorithm has three limitations that we intend to address in the next version:
1. In certain cases clouds filter has omission errors which leads to underestimation of surface area by the algorithm.
2. Similarly, in certain cases cloud shadows get incorrectly classified as water which leads to overestimation of surface area by the algorithm.
3. The current version does not mask out water bodies adjacent to the river segment in the area calculation which could lead to overestimation of surface area for river segments.
