# Jump Street

Created by: Spencer Hallyburton



## Installation

### Test Data

To test networking without real sensors, you can download image data to playback. The [Makefile][makefile] is by default configured to use ADL-Rundle-6 from the [MOT15 challenge][mot15]. To download the data, run:
```
./download_tracking.sh
```
from inside the `data` folder. Ensure the symbolic links attached appropriately to the location that the data was downloaded.

### Installing Dependencies

#### Third-Party

First, ensure that the submodules (in the folder `third_party`) are initialized. You can do this with `git submodule update --recursive`. If doing this for the first time, add an `--init` flag as well.

#### Environment With Poetry

After the third party liaries are initialized, we'll use [`poetry`][poetry] to manage a python environment. To get started, ensure `poetry` is installed on your machine by following the installation instructions [here][poetry-docs].

Once poetry is installed, you can set up the poetry environment with assistance from the [Makefile][makefile]. Specifically, try running:
```
make install
```
If all goes well, you will have a working python environment!

### Running Scripts

#### Data Broker
You will need some number of terminal windows, terminal tabs, or terminal subwindows (recommended, using [tmux][tmux])

In terminal 1, to start the data broker, run:
```
make data_broker
```

In terminal 2, to run the image-based detection, run:
```
make detection_workers
```

In terminal 3, to start the display process, run:
```
make frontend
```

In terminal 4, to start replaying sensor data, run:
```
make replay
```

You should see at the least image data being played back over the front-end display. Detection may not be set up yet to actually produce anything meaningful.

### Display (TODO)

TBD...maybe use [Grafana][grafana]?


[makefile]: https://github.com/percep-tech/jumpstreet/blob/main/Makefile
[poetry]: https://github.com/python-poetry/poetry
[poetry-docs]: https://python-poetry.org/docs/
[grafana]: https://grafana.com/
[mot15]: https://motchallenge.net/data/MOT15.zip
[tmux]: https://github.com/tmux/tmux/wiki
