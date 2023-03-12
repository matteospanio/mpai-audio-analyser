import uuid
import pytest
from audiohandler import EqualizationStandard, SpeedStandard
from ml.classification._data_structures import ClassificationResult
from cli._data_structures import IrregularityFile, Irregularity, IrregularityType, IrregularityProperties, Source


class TestIrregularityFile:

    property = IrregularityProperties(
        SpeedStandard.II,
        SpeedStandard.III,
        EqualizationStandard.CCIR,
        EqualizationStandard.IEC,
    )

    irregularity = Irregularity(uuid.uuid4(), Source.AUDIO, "00:00:00",
                                IrregularityType.BOTH, property)

    def test_init(self):
        irf = IrregularityFile([self.irregularity])
        irf2 = IrregularityFile()

        assert irf2.irregularities == []
        assert irf.irregularities == [self.irregularity]
        assert irf.offset is None

    def test_failing_add(self):
        irf = IrregularityFile()

        with pytest.raises(TypeError):
            irf.add("irregularity")

    def test_to_json(self):
        irf = IrregularityFile([self.irregularity])
        irf2 = IrregularityFile()

        assert irf.to_json() == {
            "Irregularities": [self.irregularity.to_json()]
        }
        assert irf2.to_json() == {"Irregularities": []}


class TestIrregularityProperties:

    prop = IrregularityProperties(
        writing_speed=SpeedStandard.II,
        reading_speed=SpeedStandard.III,
        writing_equalisation=EqualizationStandard.CCIR,
        reading_equalisation=EqualizationStandard.IEC,
    )

    def test_init(self):

        assert self.prop.writing_speed == SpeedStandard.II
        assert self.prop.reading_speed == SpeedStandard.III
        assert self.prop.writing_equalisation == EqualizationStandard.CCIR
        assert self.prop.reading_equalisation == EqualizationStandard.IEC

    def test_to_json(self):

        assert self.prop.to_json() == {
            "ReadingEqualisationStandard": "IEC",
            "ReadingSpeedStandard": 3.75,
            "WritingEqualisationStandard": "IEC1",
            "WritingSpeedStandard": 1.875,
        }

    def test_from_classification(self):

        result = IrregularityProperties.from_classification(
            ClassificationResult(
                SpeedStandard.II,
                SpeedStandard.III,
                EqualizationStandard.CCIR,
                EqualizationStandard.IEC,
            ))

        assert result == self.prop


class TestIrregularity:

    irreg = Irregularity(uuid.uuid4(), Source.AUDIO, "00:00:00")

    def test_init(self):
        assert self.irreg.irregularity_ID
        assert self.irreg.source == Source.AUDIO
        assert self.irreg.time_label == "00:00:00"
        assert self.irreg.irregularity_type is None
        assert self.irreg.irregularity_properties is None
        assert self.irreg.image_URI is None
        assert self.irreg.audio_block_URI is None

    def test_to_json(self):

        assert self.irreg.to_json() == {
            "IrregularityID": str(self.irreg.irregularity_ID),
            "Source": "a",
            "TimeLabel": "00:00:00",
        }
