import logging
import threading
import time
from bisect import bisect

import cv2
import numpy as np
from avstack.datastructs import DataContainer
from avstack.modules.tracking import tracks as track_types
from avstack.utils.decorators import profileit

from jumpstreet.utils import BaseClass


class VideoTrackMuxer(BaseClass):
    """Muxes together images and tracks"""

    NAME = "muxer-video-track"

    def __init__(
        self,
        video_buffer,
        track_buffer,
        identifier,
        verbose=False,
        debug=False,
    ) -> None:
        super().__init__(self.NAME, identifier, verbose=verbose, debug=debug)
        self.video_buffer = video_buffer
        self.track_buffer = track_buffer
        self.ready = False
        self.DataContainer = DataContainer
        self.t_last_muxed = -np.inf

    def start_continuous_process_thread(self, execute_rate=100, t_max_delta=0.05):
        """start up a thread for the processing function"""
        self.thread = threading.Thread(
            target=self.continuous_process,
            args=(execute_rate, t_max_delta),
            daemon=True,
        )
        self.thread.start()

    def continuous_process(self, execute_rate=100, t_max_delta=0.05):
        self.t_last_execute = None
        execute_dt = 1.0 / execute_rate
        while True:
            t_now = time.time()
            # -- check if we're ready for a trigger
            if (self.t_last_execute is None) or (
                t_now - self.t_last_execute >= execute_dt - 1e-5
            ):
                self.t_last_execute = t_now
                self.process()

            # -- sleep approximately until the next trigger time
            time.sleep(max(0, execute_dt - (t_now - self.t_last_execute) - 1e-4))

    def emit_one(self):
        return self.video_buffer.emit_one()

    @profileit(f'profile_muxer.prof', folder='profiles')
    def process(self):
        """Check the data buffer and add any muxed frames that we can

        Step 1: mux anything that we can
            -- pop tracks and apply to image
            -- keep that image on the buffer
            -- tell that image we have already applied that sensor
        Step 2: emit on a real-time schedule
            -- Emit a frame once it hits the delay timing
        """
        if self.debug:
            self.print(self.video_buffer, end="\n")
            self.print(self.track_buffer, end="\n")
        # self.print(len(self.video_buffer), end="\n")
        # -- if we have tracks, let's mux if possible
        for video_id, video_bucket in self.video_buffer.data.items():
            if len(video_bucket) > 0:
                if self.track_buffer.has_data(video_id):
                    track_bucket = self.track_buffer.data[video_id]
                    # _ = track_bucket.pop_all_below(video_bucket.top()[0] - 1e-2)
                    if len(track_bucket) > 0:
                        # -- mux if we are out of date
                        sorted_images = video_bucket.n_smallest(n=len(video_bucket))
                        prior_images = [si[0] for si in sorted_images]
                        data_images = [si[1] for si in sorted_images]
                        sorted_tracks = track_bucket.n_smallest(n=len(track_bucket))
                        prior_tracks = [st[0] for st in sorted_tracks]
                        data_tracks = [st[1] for st in sorted_tracks]
                        for i in range(len(prior_images)):  # going in order
                            for j in range(len(prior_tracks)):  # going in order
                                if prior_tracks[j] >= prior_images[i]:
                                    self.mux(data_images[i], data_tracks[j])
                                    self.t_last_muxed = max(
                                        self.t_last_muxed, prior_images[i]
                                    )
                                    break  # only mux an image once

    def mux(self, image, tracks):
        """Mux together an image with track data using opencv"""
        color = (36, 255, 12)
        thickness = 2
        img = image.data
        if image.calibration.channel_order == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.debug:
            self.print(
                f"Frame: {image.frame:3d}: muxing {len(tracks):3d} tracks onto image",
                end="\n",
            )
        for track in tracks:
            if isinstance(track, track_types.BasicBoxTrack2D):
                box = track.box
                # -- draw rectangle
                img = cv2.rectangle(
                    img,
                    (int(box.xmin), int(box.ymin)),
                    (int(box.xmax), int(box.ymax)),
                    color,
                    thickness,
                )
                # -- draw text information
                txt = f"{track.obj_type}, ID: {track.ID:3d}"
                fontscale = 0.9
                img = cv2.putText(
                    img,
                    txt,
                    (int(box.xmin), int(box.ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontscale,
                    color,
                    thickness,
                )
            elif isinstance(track, track_types.XyFromRazTrack):
                # -- project to image center
                # TODO: incorporate the radar extrinsics as origin from rad calib
                xyzh_cam = np.array(
                    [-0.5 - track.position[1], 0, track.position[0], 1]
                )  # HACK to account for shift
                xy_img = image.calibration.P @ xyzh_cam
                xy_img /= xy_img[2]

                # -- draw circle for point
                img = cv2.circle(
                    img,
                    (int(xy_img[0]), int(xy_img[1])),
                    radius=6,
                    color=(0, 255, 0),
                    thickness=-1,
                )

                # -- draw text information
                # TODO
            else:
                raise NotImplementedError(type(track))
        image.data = img

    def __getattr__(self, attr):
        """Try to use muxed_buffer's attributes"""
        try:
            return getattr(self.muxed_buffer, attr)
        except AttributeError as e:
            raise AttributeError("Could not find {} attribute".format(attr))
