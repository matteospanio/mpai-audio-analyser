import uuid
from pandas import DataFrame

from audiohandler import AudioWave
from ml.classification import load_model

from .._data_structures import Source
from .._data_structures import Irregularity
from .._data_structures import IrregularityFile
from .._data_structures import IrregularityProperties
from .._data_structures import IrregularityType
from ..files import save_as_json
from ..time import seconds_to_string


def classify(audio_blocks: DataFrame) -> list[IrregularityProperties]:

    audio_blocks_classification = []

    # classify the audioBlocks
    eq_classifier = load_model("pretto_and_berio_nono_classifier")
    prediction = eq_classifier.predict(audio_blocks)

    for i in range(len(prediction)):
        classification = IrregularityProperties.from_classification(
            prediction.iloc[i].classification)
        audio_blocks_classification.append(classification)

    return audio_blocks_classification


def extract_features(audio_blocks: list[dict]) -> DataFrame:

    features = {f'mfcc{i}': [] for i in range(1, 14)}

    for audio_block in audio_blocks:
        audio = AudioWave.from_file(audio_block["Path"])
        audio_mfcc = audio.get_mfcc()
        for i, key in enumerate(features.keys()):
            features[key].append(audio_mfcc[i])

    return DataFrame(features)


def write_irregularity_file(
        audio_blocks: list,
        classification_results: list[IrregularityProperties],
        destination_dir: str):

    irreg_file = IrregularityFile()

    for audio_block, properties in zip(audio_blocks, classification_results):
        audio_block["Classification"] = properties
        irreg_type = get_irregularity_type(properties)
        irreg_file.add(
            Irregularity(irregularity_ID=uuid.uuid4(),
                         source=Source.AUDIO,
                         time_label=seconds_to_string(audio_block["StartTime"]),
                         irregularity_type=irreg_type,
                         irregularity_properties=properties
                         if irreg_type is not None else None,
                         audio_block_URI=audio_block["Path"]))

    save_as_json(
        irreg_file.to_json(),
        f'{destination_dir}/AudioAnalyser_IrregularityFileOutput.json')


def get_irregularity_type(
        classification: IrregularityProperties) -> IrregularityType:
    if classification.reading_equalisation != classification.writing_equalisation:
        if classification.reading_speed != classification.writing_speed:
            return IrregularityType.BOTH
        return IrregularityType.EQUALIZATION
    if classification.reading_speed != classification.writing_speed:
        return IrregularityType.SPEED
    return None
