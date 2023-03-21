from jumpstreet.utils import BaseClass
from jumpstreet.buffer import BasicDataBuffer


class VideoTrackMuxer(BaseClass):
    """Muxes together images and tracks"""
    NAME = "muxer-video-track"

    def __init__(self, video_buffer, track_buffer, identifier, dt_delay=0.10) -> None:
        super().__init__(self.NAME, identifier)
        self.video_buffer = video_buffer
        self.track_buffer = track_buffer
        self.muxed_buffer = BasicDataBuffer(identifier=0, max_size=100)
        self.dt_delay = dt_delay
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
                track_below = track_bucket.pop_all_below(t_target)
                dt_below =  abs(t_target - track_below[0])
                if not track_bucket.empty():
                    track_above = track_bucket.top()
                    dt_above = abs(t_target - track_above[0])
                    track_select = track_below[1] if dt_below <= dt_above else track_above[1]
                    dt_select = min(dt_below, dt_below)
                else:
                    track_select = track_below[1]
                    dt_select = dt_below
                
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
        # TODO: actually implement the muxing
        return image
    
    def __getattr__(self, attr):
        """Try to use muxed_buffer's attributes"""
        try:
            return getattr(self.muxed_buffer, attr)
        except AttributeError as e:
            raise AttributeError('Could not find {} attribute'.format(attr))


    # def emit(self):
    #     """Emit a muxed image frame with track bounding boxes and data
        
    #     Note that once the system thinks it is ready to go, it will continue
    #     to emit images at a fixed rate, even if the track data is coming in variably
    #     this may lead to frames without track data. This characteristic underscores
    #     the importance of the dt_delay parameter. A higher delay allows more wiggle
    #     room for the track data to come in late."""

    #     # Start when we have video and track data that overlap by some measure
    #     if not self.ready:
    #         # we want to start at the earliest videos
    #         t_latest_track = self.track_buffer.get_lowest_latest_priority()
    #         if t_latest_track > t_latest_video + self.dt_delay:
    #             self.ready = True

    #     # Now we can start
    #     if self.ready:
    #         # check if we have a frame to mux and emit
    #         image = None
    #         tracks = None
    #         t_synch = None
    #         out = self.mux(image, tracks)
        
    #         # TODO: adaptively tune the dt_delay for optimal performance?
    #         self.clean(t_synch)
    #     else:
    #         out = None
    #     return out

    # def mux(self, image, tracks):
    #     pass

    # def clean(self, t):
    #     """Clear buffers up until the time point"""