from jumpstreet.utils import BaseClass
from jumpstreet.buffer import BasicDataBuffer


class VideoTrackMuxer(BaseClass):
    """Muxes together images and tracks"""
    NAME = "muxer-video-track"

    def __init__(self, video_buffer, track_buffer, identifier) -> None:
        super().__init__(self.NAME, identifier)
        self.video_buffer = video_buffer
        self.track_buffer = track_buffer
        self.muxed_buffer = BasicDataBuffer(identifier=0, max_size=100)
        self.ready = False

    def init(self):
        self.muxed_buffer.init()  # need to explicitly call initialize due to Qt import error

    def process(self, t_max_delta=0.05):
        """Check the data buffer and add any muxed frames that we can"""
        from avstack.datastructs import DataContainer

        for video_id, video_bucket in self.video_buffer.data.items():
            if self.track_buffer.has_data(video_id):
                # -- select either the above or below track
                track_bucket = self.track_buffer.data[video_id]
                t_target = video_bucket.top()[0]

                # -- get below tracks (pop since we don't need anymore afterwards...maybe)
                track_below = track_bucket.pop_all_below(t_target, with_priority=True)
                if len(track_below) > 0:
                    track_below = track_below[-1]
                    dt_below =  abs(t_target - track_below[0])
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
                    track_select = track_below[1] if dt_below <= dt_above else track_above[1]
                    dt_select = min(dt_below, dt_below)

                # -- if they are within a threshold, mux!
                if dt_select <= t_max_delta:
                    image = video_bucket.pop()
                    frame = image.frame
                    timestamp = image.timestamp
                    image_mux = self.mux(image, track_select)
                    mux_data_container = DataContainer(frame, timestamp, image_mux, video_id)
                    self.muxed_buffer.push(mux_data_container)

    def mux(self, image, tracks):
        """Mux together an image with track data using opencv"""
        import cv2  # defer import due to conflict with qt
        color = (0, 255, 0)
        img = image.data
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
            raise AttributeError('Could not find {} attribute'.format(attr))
