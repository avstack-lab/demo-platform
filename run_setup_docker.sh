#!/usr/bin/env bash

set -e 

DATAFOLDER=${1:-/data}
MODELFOLDER=${2:-/models}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
MODELFOLDER=${MODELFOLDER%/}  # remove trailing slash

# (optional) Add symbolic links to data in api folder
# ./submodules/lib-avstack-api/data/add_custom_symlinks.sh $DATAFOLDER

# Add symbolic links to data here
# mkdir ./data
ln -sf "${DATAFOLDER}/KITTI" "./data/KITTI"
ln -sf "${DATAFOLDER}/nuScenes" "./data/nuScenes"
ln -sf "${DATAFOLDER}/MOT15" "./data/MOT15"
ln -sf "${DATAFOLDER}/nuImages" "./data/nuImages"
ln -sf "${DATAFOLDER}/CARLA" "./data/CARLA"

# Add symbolic links to perception models
./third_party/lib-avstack-core/models/download_mmdet_models.sh $MODELFOLDER
./third_party/lib-avstack-core/models/download_mmdet3d_models.sh $MODELFOLDER