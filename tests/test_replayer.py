from jumpstreet.context import SerializingContext
from jumpstreet.sensor_replay import SensorDataReplayer


test_img_dir = "data/TUD-Campus/img1"


def test_replayer_load_image():
    HOST = "localhost"
    PORT = 5555
    context = SerializingContext()
    replayer = SensorDataReplayer(
        context, HOST=HOST, PORT=PORT, identifier=1, send_dir=test_img_dir
    )
    assert len(replayer.images) > 0
    assert replayer.backend is not None
