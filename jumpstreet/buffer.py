from dataclasses import dataclass, field
from typing import Any
from jumpstreet.utils import BaseClass
# from avstack.datastructs import PriorityQueue


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


class VideoBuffer(BaseClass):
    """Storing videos and processing triggers"""

    NAME = "video-buffer"

    def __init__(self, identifier, max_size=100) -> None:
        super().__init__(self.NAME, identifier)
        self.buffers = {}
        self._t_start = None
        self._t_end = None
        self.max_size = max_size

    def store(self, t, cam_id, image):
        if cam_id not in self.buffers:
            self.buffers[cam_id] = PriorityQueue(max_size=self.max_size, max_heap=False)
        self.buffers[cam_id].push(priority=t, item=image)

    def trigger(self, t_start=None, t_end=None):
        """Process a trigger

        Must be paired t_start and t_end
        """
        images_out = None
        if (t_start is not None) and (
            (self._t_start is None) or (self._t_start < t_start)
        ):
            self._t_start = t_start
        if (t_end is not None) and (
            (self._t_end is None) or (self._t_end < self._t_end)
        ):
            self._t_end = t_end

        if (
            (self._t_start is not None)
            and (self._t_end is not None)
            and (self._t_start <= self._t_end)
        ):
            images_out = self._trigger(self._t_start, self._t_end)
            self._t_start = None
            self._t_end = None
        return images_out

    def _trigger(self, t_start, t_end):
        images_out = {}
        for cam, buffer in self.buffers.values():
            # -- remove all items below t_start
            buffer.pop_all_below(t_start, save=False)
            # -- get all items up to t_end
            images_out[cam] = buffer.pop_all_below(t_end, save=True)
        return images_out