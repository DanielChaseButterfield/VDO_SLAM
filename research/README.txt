# Development of VDO-SLAM+IRTE

This readme is temporary, and used for the development process of VDO-SLAM+IRTE. VDO-SLAM was originally developed on Ubuntu 16.04, meaning that we need a simulated Docker environment to continue development.

## Installation & Setup

Run the following command to build the Docker image

```
sudo build_docker.sh
```

Run the following command to open a Docker container from the image to develop within.

```
sudo open_docker.sh
```

You are now in an environment with all of the proper dependencies.

## Building and Running VDO-SLAM

Build VDO-SLAM and the test cases with the following command:

``` 
sudo build.sh
```

Run VDO-SLAM on the KITTI Dataset with the following command. You'll need to follow the instructions in the main readme for information on installing it:

```
./example/vdo_slam example/kitti-0000-0013.yaml PATH_TO_KITTI_SEQUENCE_DATA_FOLDER
```

## Test cases

The test cases can be found in "/tests/", and can be run with the following command:

```
./example/test_vdo_slam
```
