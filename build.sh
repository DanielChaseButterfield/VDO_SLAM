# Detect if we're within docker
if [ -f /.dockerenv ]; then
    # If we are, build files outside of mounted VDO-SLAM repo
    echo "Configuring and building g2o ..."

    cd ../

    mkdir build_g2o
    cd build_g2o
    cmake ../VDO_SLAM/dependencies/g2o -DCMAKE_BUILD_TYPE=Release
    make -j

    cd ../

    echo "Configuring and building VDO-SLAM ..."

    mkdir build
    cd build
    cmake ../VDO_SLAM/ -DCMAKE_BUILD_TYPE=Release
    make -j

else
    # If we aren't, build as normal
    echo "Configuring and building g2o ..."

    cd dependencies/g2o

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j

    cd ../../../

    echo "Configuring and building VDO-SLAM ..."

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j
fi
