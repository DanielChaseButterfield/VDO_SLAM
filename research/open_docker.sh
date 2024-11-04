VDO_SLAM_ROOT="/home/dbutterfield3/Desktop/VDO_SLAM"

# Lines 5 & 6 for X-Forwarding - Make sure to run "xhost +local:root" on host computer for this to work
# Lines 7 to the VDO_SLAM repository so changes update to local computer.
docker run -it --rm --net=host \
    -e "DISPLAY=$DISPLAY" \
    -v "${HOME}/.Xauthority:/root/.Xauthority:ro" \
    -v ${VDO_SLAM_ROOT}:/root/VDO_SLAM \
    vdo_slam