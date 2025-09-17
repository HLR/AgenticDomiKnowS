"""Sensors that expose dataset fields to the graph."""
from domiknows.sensor.pytorch.sensors import ReaderSensor


class ImageSensor(ReaderSensor):
    def __init__(self, *pres, keyword: str, **kwargs):
        super().__init__(*pres, keyword=keyword, label=False, **kwargs)


class DigitLabelSensor(ReaderSensor):
    def __init__(self, *pres, keyword: str, **kwargs):
        super().__init__(*pres, keyword=keyword, label=True, **kwargs)


class SumLabelSensor(ReaderSensor):
    def __init__(self, *pres, keyword: str = "sum_label", **kwargs):
        super().__init__(*pres, keyword=keyword, label=True, **kwargs)


__all__ = ["ImageSensor", "DigitLabelSensor", "SumLabelSensor"]
