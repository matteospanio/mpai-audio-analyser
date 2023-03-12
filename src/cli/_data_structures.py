import uuid
from dataclasses import dataclass
from enum import Enum

from audiohandler import EqualizationStandard, SpeedStandard
from ml.classification import ClassificationResult


class IrregularityType(Enum):
    SPEED = "ssv"
    EQUALIZATION = "esv"
    BOTH = "ssv"


class Source(Enum):
    AUDIO = "a"
    VIDEO = "v"
    BOTH = "b"


@dataclass
class IrregularityProperties:
    reading_speed: SpeedStandard
    reading_equalisation: EqualizationStandard
    writing_speed: SpeedStandard
    writing_equalisation: EqualizationStandard

    @staticmethod
    def from_classification(classification_result: ClassificationResult):
        return IrregularityProperties(
            reading_speed=classification_result.reading_speed,
            reading_equalisation=classification_result.reading_equalization,
            writing_speed=classification_result.writing_speed,
            writing_equalisation=classification_result.writing_equalization,
        )

    def to_json(self):
        return {
            "ReadingSpeedStandard": self.reading_speed.value,
            "ReadingEqualisationStandard": self.reading_equalisation.value,
            "WritingSpeedStandard": self.writing_speed.value,
            "WritingEqualisationStandard": self.writing_equalisation.value,
        }


@dataclass
class Irregularity:
    irregularity_ID: uuid.UUID
    source: Source
    time_label: str
    irregularity_type: IrregularityType = None
    irregularity_properties: IrregularityProperties | None = None
    image_URI: str | None = None
    audio_block_URI: str | None = None

    def to_json(self):
        dictionary = {
            "IrregularityID": str(self.irregularity_ID),
            "Source": self.source.value,
            "TimeLabel": self.time_label,
        }

        if self.irregularity_type:
            dictionary["IrregularityType"] = self.irregularity_type.value

        if self.image_URI:
            dictionary["ImageURI"] = self.image_URI

        if self.audio_block_URI:
            dictionary["AudioBlockURI"] = self.audio_block_URI

        if self.irregularity_properties:
            dictionary[
                "IrregularityProperties"] = self.irregularity_properties.to_json(
                )

        return dictionary


class IrregularityFile:
    # TODO: the offset calculation is not implemented yet, so it is set to None
    irregularities: list[Irregularity]
    offset: int | None

    def __init__(self,
                 irregularities: list[Irregularity] = [],
                 offset: int | None = None):
        self.irregularities = irregularities
        self.offset = offset

    def to_json(self):
        dictionary = {
            "Irregularities":
            [irregularity.to_json() for irregularity in self.irregularities],
        }

        if self.offset:
            dictionary["Offset"] = self.offset

        return dictionary

    def add(self, irregularity: Irregularity):
        """Add an irregularity to the list of irregularities.

        Parameters
        ----------
        irregularity : Irregularity
            the irregularity to add

        Raises
        ------
        TypeError
            if the irregularity is not a py:class:`Irregularity` object
        """
        if not isinstance(irregularity, Irregularity):
            raise TypeError(
                "IrregularityFile.add() expects an Irregularity object")
        self.irregularities.append(irregularity)
