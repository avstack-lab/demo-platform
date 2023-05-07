# ICCPS Demo Platform

Created by: Spencer Hallyburton


## Installation

### Test Data

To test networking without real sensors, you can download image data to playback. The [Makefile][makefile] is by default configured to use ADL-Rundle-6 from the [MOT15 challenge][mot15]. To download the data, run:
```
./download_tracking.sh
```
from inside the `data` folder. **NOTE:** if you system does not allow you to make a folder at /data/tracking, then pass an argument to the call above with your custom download folder, e.g., `./download_tracking.sh ./data/tracking` aka locally. You can use local downloads of any `AVstack`-compatible dataset such as KITTI or nuScenes.

Ensure the symbolic links attached appropriately to the location that the data was downloaded.

### Installing Dependencies


#### Third-Party

First, ensure that the submodules (in the folder `third_party`) are initialized. You can do this with `git submodule update --recursive`. If doing this for the first time, add an `--init` flag as well.

The only configuration of third party libraries you'll need is to download perception models. You will need to do this in the `third_party/lib-avstack-core/models` folder. Follow the installation instructions for `AVstack` for more clarity.

#### Environment With Poetry

After the third party liaries are initialized, we'll use [`poetry`][poetry] to manage a python environment. To get started, ensure `poetry` is installed on your machine by following the installation instructions [here][poetry-docs].

NOTE: if the installation is giving you troubles (e.g., in my experience, with `pycocotools` in `aarch64`-based chips), try the following:
```
poetry config experimental.new-installer false
```
to get through the problem packages, then reactivation with `true`.

Once poetry is installed, you can set up the poetry environment with assistance from the [Makefile][makefile]. Specifically, try running:
```
make install
```
If all goes well, you will have a working python environment!

#### Installing on NVIDIA Jetson

After we build the `jetson-inference` library, we need to link the site packages by running:
```
./link_site_packages.sh
```

### Running Scripts

#### Replaying Sensor Data

You will just need two terminal windows to run a replay system. The first will run our controller, the second will replay the data. Specifically, run:

```
make controller CCONF=camera
```
in the first window (waiting for the perception models to say they are initialized) and
```
make mot15_replay  # or make kitti_replay, e.g.
```
in the second window. If all goes well, you will see the sensor data replayed on the Qt window with tracks and labels.

#### Physical Sensors

This demo platform can also be run with physical sensors over an ethernet or serial connection. We demonstated this capability at ICCPS 2023. We plan to provide more details on how to accomplish this in the future. If particularly interested, please reach out to us, and we can schedule a consultation.


## Other Things

[Creating virtual network interfaces][virtual-network], run `sudo ip addr add 192.168.1.11/24 dev eth0`

[makefile]: https://github.com/percep-tech/jumpstreet/blob/main/Makefile
[poetry]: https://github.com/python-poetry/poetry
[poetry-docs]: https://python-poetry.org/docs/
[grafana]: https://grafana.com/
[mot15]: https://motchallenge.net/data/MOT15.zip
[tmux]: https://github.com/tmux/tmux/wiki
[virutal-network]: https://ostechnix.com/how-to-assign-multiple-ip-addresses-to-single-network-card-in-linux/