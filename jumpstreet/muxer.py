import threading
import time

from jumpstreet.utils import BaseClass


class VideoTrackMuxer(BaseClass):
    """Muxes together images and tracks"""

    NAME = "muxer-video-track"

    def __init__(self, video_buffer, track_buffer, identifier, verbose=False) -> None:
        super().__init__(self.NAME, identifier, verbose=verbose)
        self.video_buffer = video_buffer
        self.track_buffer = track_buffer
        self.ready = False

        # need to defer import due to Qt import error
        from avstack.datastructs import BasicDataBuffer, DataContainer

        self.DataContainer = DataContainer
        self.muxed_buffer = BasicDataBuffer(max_size=100)

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
                self.process(t_max_delta)

            # -- sleep approximately until the next trigger time
            time.sleep(max(0, execute_dt - (t_now - self.t_last_execute) - 1e-4))

    def process(self, t_max_delta=0.05):
        """Check the data buffer and add any muxed frames that we can"""
        for video_id, video_bucket in self.video_buffer.data.items():
            if self.track_buffer.has_data(video_id):
                # -- select either the above or below track
                track_bucket = self.track_buffer.data[video_id]
                t_target = video_bucket.top()[0]

                # -- get below tracks (pop since we don't need anymore afterwards...maybe)
                track_below = track_bucket.pop_all_below(t_target, with_priority=True)
                if len(track_below) > 0:
                    track_below = track_below[-1]
                    dt_below = abs(t_target - track_below[0])
                else:
                    track_below = None

                # -- get above track
                if not track_bucket.empty():
                    track_above = track_bucket.top()
                    dt_above = abs(t_target - track_above[0])
                else:
                    track_above = None

                # -- make a track selection
                if track_below is None:
                    track_select = track_above[1]
                    dt_select = dt_above
                elif track_above is None:
                    track_select = track_below[1]
                    dt_select = dt_below
                else:
                    track_select = (
                        track_below[1] if dt_below <= dt_above else track_above[1]
                    )
                    dt_select = min(dt_below, dt_below)

                # -- if they are within a threshold, mux!
                if dt_select <= t_max_delta:
                    image = video_bucket.pop()
                    frame = image.frame
                    timestamp = image.timestamp
                    image_mux = self.mux(image, track_select)
                    mux_data_container = self.DataContainer(
                        frame, timestamp, image_mux, video_id
                    )
                    self.muxed_buffer.push(mux_data_container)

    def mux(self, image, tracks):
        """Mux together an image with track data using opencv"""
        import cv2  # defer import due to conflict with qt

        color = (0, 255, 0)
        img = image.data
        if self.verbose:
            self.print(f'Frame: {image.frame:3d}: muxing {len(tracks):3d} tracks onto image', end='\n')
        for track in tracks:
            box = track.box
            img = cv2.rectangle(
                img,
                (int(box.xmin), int(box.ymin)),
                (int(box.xmax), int(box.ymax)),
                color,
                2,
            )
        return img

    def __getattr__(self, attr):
        """Try to use muxed_buffer's attributes"""
        try:
            return getattr(self.muxed_buffer, attr)
        except AttributeError as e:
            raise AttributeError("Could not find {} attribute".format(attr))
