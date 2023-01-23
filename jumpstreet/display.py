from jumpstreet.utils import BaseClass


class Display(BaseClass):
    NAME = "display"

    def __init__(self, identifier) -> None:
        super().__init__(self.NAME, identifier)

    def start(self):
        """Start up display process"""
        raise NotImplementedError

    def update(self):
        """Update the display with image batch"""
        raise NotImplementedError


class ConfirmationDisplay(Display):
    def __init__(self, identifier) -> None:
        super().__init__(identifier)

    def start(self):
        self.print("started display process")

    def update(self, images):
        ks = images.keys()
        lvs = [len(imgs) for imgs in images.values()]
        self.print(f"received image batch with keys {ks}" + \
                   f"and value lens {lvs} images")


class StreamThrough(Display):
    def __init__(self, identifier) -> None:
        super().__init__(identifier)

    def start(self):
        return super().start()

    def update(self):
        return super().start()