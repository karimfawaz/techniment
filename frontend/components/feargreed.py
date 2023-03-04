import dash_daq as daq
import fear_and_greed


def feargreed():
    return daq.Gauge(
        color={"gradient": True, "ranges": {
            "red": [0, 33], "yellow": [33, 66], "green": [66, 100]}},
        value=fear_and_greed.get().value,
        label=fear_and_greed.get().description,
        max=100,
        min=0,
    )
