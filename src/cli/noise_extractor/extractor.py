import os
import logging
import concurrent.futures as cf
from audiohandler import AudioWave, Noise


def extract_noise(filepath: str, noise_list: list[Noise], silence_len: int,
                  filename: str, destination_dir: str) -> tuple:
    """
    Extract silence indexes from an AudioWave object and save the file chunks to disk.

    Parameters
    ----------
    audio: AudioWave
        the audio file to analyze
    noise_list: list[Noise]
        a list of Noise objects
    silence_len: int
        the duration in milliseconds of silence to detect
    filename: str
        the name of the audio file

    Returns
    -------
    tuple
        a tuple containing the destination of the json file and a json representation of the results and saves the file chunks to disk

    .. note::
        It analyzes the audio file in parallel, extracting silence indexes from each channel.
    """

    log = logging.getLogger()

    log.info("Loading %s", filename)
    audio = AudioWave.from_file(filepath, bufferize=True)
    destination_dir = os.path.join(destination_dir, filename)

    # extract an AudioWave object for each channel
    tracks: list[AudioWave] = []
    for channel in range(audio.channels):
        tracks.append(audio.get_channel(channel))

    log.info("Extracting noise indexes from %s", filename)
    # spawn a thread to extract silence indexes from each channel
    with cf.ThreadPoolExecutor(max_workers=len(tracks)) as executor:
        futures = [
            executor.submit(
                lambda track, i:
                (str(i), track.get_silence_slices(noise_list, silence_len)),
                track, i) for i, track in enumerate(tracks)
        ]
        results: list[tuple[str, dict]] = [f.result() for f in futures]

    # convert list of tuples to dict in the form {channel: {noise_type: [(start, end), ...]}}
    chindex = dict(results)

    # save file chunks to disk
    log.info("Splitting %s into silence fragments", filename)
    save_chunks(tracks, chindex, destination_dir)

    # create a json representation of the results
    json_result = create_indexes_representation(chindex, tracks,
                                                destination_dir)

    return destination_dir, json_result


def save_chunks(tracks: list[AudioWave], results: dict[str, dict],
                destination_dir: str):
    """
    Save file chunks in folder ``destination_dir/AudioBlocks``
    """
    # create a directory for the file
    directory = os.path.join(destination_dir, "AudioBlocks")
    if not os.path.isdir(directory):
        os.makedirs(directory)
    else:
        # clean the content of the destination directory
        for file in os.listdir(directory):
            os.remove(os.path.join(directory, file))

    for channel in sorted(results.keys()):
        for key, value in results[channel].items():
            for start, end in value:
                tracks[int(channel)][start:end].save(
                    f"{directory}/{key}_{channel}_{start}_{end}.wav")


def create_indexes_representation(results: dict[str,
                                                dict], tracks: list[AudioWave],
                                  destination_dir: str) -> list:
    json_result = []

    for ch in sorted(results.keys()):
        for key, value in results[ch].items():
            for start, end in value:
                resulting_slice = tracks[int(ch)][start:end]
                json_result.append({
                    "NoiseType": key,
                    "StartTime": (start) / resulting_slice.samplerate,
                    "EndTime": (end) / resulting_slice.samplerate,
                    "Channel": int(ch),
                    "Path":
                    f"{destination_dir}/AudioBlocks/{key}_{ch}_{start}_{end}.wav",
                    "RMSdB": resulting_slice.db_rms(),
                })

    return json_result
