# riverwidthEO
A python package to process river segments using satellite imagery and machine learning.

### Background
In many earth science applications, calibration of physical models is a key challenge because ground observations are very scarce or completely absent in most regions. For example, hydrological models simulate the flow of water in a basin using physical principles, but necessarily contain numerous parameters (e.g., soil conductivity at different grid points) whose values need to be calibrated for each study region with the help of observations. The most commonly used observation is discharge (volume per second) estimates that are available through ground stations. These stations are costly to install and maintain, and thus are limited in number.  This paucity (or complete absence) of observation data often leads to poorly calibrated models that provide incorrect predictions or have high uncertainty in practice.

Our approach is to provide this much needed calibration data using novel machine learning techniques and multi-temporal satellite imagery that is available freely from Earth Observing satellite based sensors such as Sentinel and Landsat. The latest version (version 1.0) of the package uses descarteslabs API to download Sentinel-2 imagery of any given river segment. The multi-spectral imagery is then converted into land/water maps using CNN based deep learning techniques. The area variations thus obtained can be used to constraint hydrological models. Click on the [<ins>video</ins>](http://umnlcc.cs.umn.edu/tmp/data-1050883510-7366.mp4) to see surface area variations of a river segment in Ethiopia.

An example python script (examples.py) provides information on how to use the package.  


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

### Notes
