Usage
=====================

Installation
-------------------------------

In order to make the software working it is necessary to install the libraries listed in the ``pyproject.toml`` file, poetry will manage for you the versions of the libraries.

Build from source
++++++++++++++++++

If you want just to build the library from source without development libraries run:

.. code-block:: console

    poetry build
    cd dist
    pip install audiohandler-<version>.whl

it will build the library and install it in your os default python environment.

Development setup
++++++++++++++++++

For development and testing purposes it is better to install the library in a separate environment, poetry creates it for you! Just run ``poetry install --only main`` to have just the bare metal library experience, otherwise run ``poetry install`` to install also development packages (there's also the shortcut ``make install`` to install all dependencies except the notebooks ones).

For a development setup i suggest to run ``make install`` and then ``poetry shell`` to enter the poetry environment. This will allow you to run the software from the command line and to import the module in your IDE.

After any of this commands you will have ``noise-extractor``, ``noise-classifier``, ``audio-analyser`` executables and the module available for import in the poetry environment.

Docker
++++++

To avoid any issue with setup and installations a docker container is provided launch the command ``docker build -t mpai .`` to create a local container called mpai, it will install all the dependencies and the library itself. To run the container use ``docker run -it mpai``.

noise-extractor
-------------------

The software exposes a CLI interface called ``noise-extractor`` that makes it easy to integrate in more complex pipelines. Running ``noise-extractor --help`` we obtain the help message:

.. code-block:: console
    :caption: The CLI help message

    usage: noise-extractor [-h] -i INPUT -d DESTINATION_DIR -l SILENCE_LEN [-q]

    A tool to extract noise from sound files

    options:
    -h, --help            show this help message and exit
    -i INPUT, --input INPUT
                        Specify a single file or a folder containing audio files as input
    -d DESTINATION_DIR, --destination-dir DESTINATION_DIR
                            Where to put file chunks
    -l SILENCE_LEN, --silence-len SILENCE_LEN
                            The duration in milliseconds of silence to detect
    -q, --quiet           Display less output (errors only)

To execute the application it is necessary to specify the input file(s) using the ``--input`` argument, the destination folder through the ``--destination-dir`` argument and the ``--silence-len``.

.. warning::

    * The ``--input`` argument can be a single file or a folder containing audio files. If a folder is specified, all the files in the folder will be processed, but if there are subfolders in the input folder they will be ignored (future versions might include a recursive option to scan subfolders for files). The files are assumed to be in the .wav format, other formats are actually not supported neither tested (so weird inputs could have unexpected behaviour).
    * If the ``--destination-dir`` folder does not exist it will be created. If it exists, the software **will overwrite** existing files giving a warning message.

The following example executes the script on the content of the ``AudioSamples`` folder, saving the silence portions obtained in the ``Results`` folder, which must have a minimum length of 500ms in order to be extracted from the input file:

Here is an example of a standard execution:

.. code-block:: console
    :caption: CLI example

    noise-extractor --input AudioSamples --destination-dir Results --silence-len 500

we asked ``noise-extractor`` to extract silence portions long exactly 500 milliseconds from the files in the ``AudioSamples`` folder, saving the results in the ``Results`` folder.
Assuming that the ``AudioSamples`` folder contains the file ``W3N_R3N.wav`` the console output will be the following:

.. code-block::
    :caption: CLI output

    [INFO] Reading files
    [INFO] Loading W3N_R3N
    [INFO] Extracting noise indexes from W3N_R3N
    [INFO] Splitting W3N_R3N into silence fragments
    [INFO] Creating json logfile

.. note::

    The ``--quiet`` option can be used to suppress the output of the software, only errors will be displayed.

once the execution is completed the ``Results`` folder will have the following structure:

.. code-block:: console
    :caption: The output folder structure

    Results
    └── W3N_R3N
        ├── log.json
        └── AudioBlocks
            ├── A_0_10694304_10885248.wav
            ├── ...
            └── A_1_9352032_9541248.wav

for each file in input a folder with the same name is created, containing a folder ``AudioBlocks`` with the silence portions extracted from the input file. File names are composed by the noise type (``A``, ``B`` or ``C``), the channel number, the start sample and the end sample. To generate more data multi-track files are split into single-track files, i. e. a stereo file will be split into two mono files, one for each channel. This explains why the channel index is present in the file name. Along with the splitted audio files a ``log.json`` file is provided, it contains the information about the extracted silence portions.

.. seealso::

    :class:`audiohandler.Noise` to know more about the noise types ``A``, ``B`` and ``C``.

The .json file has the following structure:

.. code-block:: javascript
    :caption: Log file structure

    [
        {
            "NoiseType": "A", // The noise type
            "StartTime": 47.032614583333334, // The start time of the silence portion
            "EndTime": 47.532614583333334, // The end time of the silence portion
            "Channel": 0, // The channel number
            "Path": "W3N_R3N/AudioBlocks/A_0_10694304_10885248.wav", // The path of the file containing the silence portion
            "RMSdB": -61.75726363813678
        },
        {
            ...
        }
    ]

it contains a list of objects, each object represents a silence found, storing all main informations about the audio block.

How does it work?
+++++++++++++++++

Before analyzing in depth how the software works, it is important to specify what it is supposed to do. Summarising in few lines the software requirements are:

1. Given an audio file (or a folder containing audio files) in input extract the silence portions from the file(s)
2. Multi-channel files must be split into single-channel files, one for each channel
3. Save the silence portions in a folder with an opportune naming convention
4. Save a log file containing information about the extracted silence portions in a .json file

Given the requirements, it's trivial to define the main steps of the software:

.. code-block::
    :caption: the main steps of the software in pseudo-code
    :linenos:

    function extractNoiseSingleThreaded:
        Read the input files
        for each file in input:
            for each channel in the file:
                extract the silence portions from the channel
                save the silence portions in a folder with an opportune name
        save a json file containing information about the extracted silence portions 

since input files are independent from each other, the software can be easily parallelized. The following pseudo code spawns a thread for each input file:

.. code-block::
    :caption: the main steps of the software in pseudo-code (parallelized)
    :linenos:

    function extractNoiseParallelized:
        Read the input files
        for each file in input:
            spawn a thread calling extractNoise on the file
        wait for all threads to finish
        save a json file containing information about the extracted silence portions

    function extractNoise:
        for each channel in the file:
            extract the silence portions from the channel
            save the silence portions in a folder with an opportune name

.. note::

    the ``extractNoise`` function can be parallelized as well in an analogous way.

as programmers we are already really happy with the result obtained: from a collection of specifications we have defined two simple routines, and the major part of these routines is trivial to implement (in fact a lot of steps involve simple file handling operations). The only part that requires some effort is the extraction of the silence portions from the audio files which relies on the :meth:`audiohandler.AudioWave.get_silence_slices` method. This method is the core of the software and it is the one that requires the most effort to implement.

noise-classifier
-------------------

The software exposes a CLI interface called ``noise-classifier`` that makes it easy to integrate in more complex pipelines. Running ``noise-classifier --help`` we obtain the help message:

.. code-block:: console
    :caption: The CLI help message

    usage: noise-classifier [-h] -i INPUT -d DESTINATION_DIR [-q]

    A tool to classify audio files equalization and speed based on MFCCs

    options:
    -h, --help            show this help message and exit
    -i INPUT, --input INPUT
                            Specify a single file or a folder containing audio files as input
    -d DESTINATION_DIR, --destination-dir DESTINATION_DIR
                            Where to put classification results
    -q, --quiet           Display less output (errors only)

To execute the application it is necessary to specify the input file(s) using the ``--input`` argument and the destination folder through the ``--destination-dir`` argument.

.. code-block::
    :caption: CLI output

    [INFO] Reading files
    [INFO] Extracting features for W3N_R3N
    [INFO] Classifying W3N_R3N
    [INFO] Writing irregularity file for W3N_R3N

once the execution is completed the ``Results`` folder will have the following structure:

.. code-block:: console
    :caption: The output folder structure

    Results
    └── W3N_R3N
        ├── AudioAnalyser_IrregularityFileOutput.json
        └── AudioBlocks
            ├── A_0_10694304_10885248.wav
            ├── ...
            └── A_1_9352032_9541248.wav

The .json file has the following structure:

.. code-block:: javascript
    :caption: Irregularity file structure

    {
        "Irregularities": [
            {
                "IrregularityID": "6f4e409c-7e55-4a1d-83e9-aecdb6463fff",
                "Source": "a",
                "TimeLabel": "00:00:09.078",
                "AudioBlockURI": "Results/BERIO074/AudioBlocks/A_0_871571_919571.wav"
            },
            {
            "IrregularityID": "a55a5c59-f876-4cf9-bf4c-d868300278a3",
            "Source": "a",
            "TimeLabel": "00:03:02.462",
            "IrregularityType": "ssv",
            "AudioBlockURI": "Results/BERIO074/AudioBlocks/B_0_17516362_17564362.wav",
            "IrregularityProperties": {
                "ReadingSpeedStandard": 7.5,
                "ReadingEqualisationStandard": "IEC1",
                "WritingSpeedStandard": 3.75,
                "WritingEqualisationStandard": "IEC2"
            },
            ...
        ]
    }

How does it work?
+++++++++++++++++

Summarising in few lines the software requirements are:

1. Extracts the features from the input files (MFCCs)
2. Classifies the input files based on the extracted features
3. Saves the classification results in an Irregularity file

audio-analyser
----------------

The last CLI tool is called ``audio-analyser`` and it is used to launch the noise-extractor and the noise-classifier tools togheter. Running ``audio-analyser --help`` we obtain the help message:

.. code-block:: console
    :caption: The CLI help message

    usage: audio-analyser [-h] -i INPUT -d DESTINATION_DIR [-q]

    A tool to generate irregularity files from audio files

    options:
    -h, --help            show this help message and exit
    -i INPUT, --input INPUT
                            Specify a single file or a folder containing audio files as input
    -d DESTINATION_DIR, --destination-dir DESTINATION_DIR
                            Where to put irregularity files
    -q, --quiet           Display less output (errors only)