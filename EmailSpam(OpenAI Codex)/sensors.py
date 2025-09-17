"""Sensors mapping raw email records onto the knowledge graph."""
from domiknows.sensor.pytorch.sensors import ReaderSensor


class HeaderSensor(ReaderSensor):
    def __init__(self, *pres, keyword: str = "header", **kwargs):
        super().__init__(*pres, keyword=keyword, label=False, **kwargs)


class BodySensor(ReaderSensor):
    def __init__(self, *pres, keyword: str = "body", **kwargs):
        super().__init__(*pres, keyword=keyword, label=False, **kwargs)


class SpamLabelSensor(ReaderSensor):
    def __init__(self, *pres, keyword: str = "label", **kwargs):
        super().__init__(*pres, keyword=keyword, label=True, **kwargs)
