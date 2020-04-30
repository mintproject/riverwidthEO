# riverwidthEO
process river segments using satellite imagery and machine learning.

### Background
In many earth science applications, calibration of physical models is a key challenge because ground observations are very scarce or completely absent in most regions. For example, hydrological models simulate the flow of water in a basin using physical principles, but necessarily contain numerous parameters (e.g., soil conductivity at different grid points) whose values need to be calibrated for each study region with the help of observations. The most commonly used observation is discharge (volume per second) estimates that are available through ground stations. These stations are costly to install and maintain, and thus are limited in number.  This paucity (or complete absence) of observation data often leads to poorly calibrated models that provide incorrect predictions or have high uncertainty in practice.

Our approach is to provide this much needed calibration data using novel machine learning techniques and multi-temporal satellite imagery that is available freely from Earth Observing satellite based sensors such as Sentinel and Landsat. The latest version (version 1.0) of the packages uses descarteslabs API to download Sentinel-2 imagery of any given river segment. The multi-spectral imagery is then converted into land/water maps using CNN based deep learning techniques. The area variations thus obtained can be used to constraint hydrological models. Click on the video below to see surface area variations of a river segment in Ethiopia.

[![](http://umnlcc.cs.umn.edu/tmp/method_example.png)](http://umnlcc.cs.umn.edu/tmp/data-1050883510-7366.mp4)
