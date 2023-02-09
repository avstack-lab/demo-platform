# ICCPS Demo Platform

Created by: Spencer Hallyburton


## Installation

### Test Data

To test networking without real sensors, you can download image data to playback. The [Makefile][makefile] is by default configured to use ADL-Rundle-6 from the [MOT15 challenge][mot15]. To download the data, run:
```
./download_tracking.sh
```
from inside the `data` folder. **NOTE:** if you system does not allow you to make a folder at /data/tracking, then pass an argument to the call above with your custom download folder, e.g., `./download_tracking.sh ./data/tracking` aka locally.

Ensure the symbolic links attached appropriately to the location that the data was downloaded.

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

#### Bugs 

##### Display Must Be Running

Currently, there is a bug where the data broker will not pass on data to the detection workers if the frontend display is not running. I think this is because the XSUB in the data broker is connected to an XPUB in the data broker which is connected to a SUB in the display process. If there is no display process, then XPUB will not need to send anything meaning XSUB *thinks* it does not need to receive anything when in reality it still needs to receive something to send to the ROUTER. It's possible that changing the XSUB to a regular SUB and changing the XPUB to a regular PUB is the better solution. This needs to be fixed so the detection can work even if display is not working.

##### Always-copy

When data is passed over a socket, it currently must be copied in order to retain the byte-stream structure. Ideally, we would NOT copy the data if we didn't have to, however, this would return a ZMQ.Frame object which is not currently handled in the code. This is a low-priority issue.


### Display (TODO)

TBD...maybe use [Grafana][grafana]?


[makefile]: https://github.com/percep-tech/jumpstreet/blob/main/Makefile
[poetry]: https://github.com/python-poetry/poetry
[poetry-docs]: https://python-poetry.org/docs/
[grafana]: https://grafana.com/
[mot15]: https://motchallenge.net/data/MOT15.zip
[tmux]: https://github.com/tmux/tmux/wiki
