.. _audio_analyser:

An Audio Analyser implementation
===================================

.. epigraph::

   The purpose of software engineering is to control complexity, not to create it.

   -- Pamela Zave

Even if the equalization curves classification didn't give excellent results in the experiments illustrated at :ref:`data-analysis`, the Audio Analyser AIM is still a very important component of the preservation system. The Audio Analyser AIM is the first step of the preservation process and it is responsible for the detection of irregularities in the audio signal, and the modularity across the MPAI-AIF model allows us to implement the Audio Analyser even without a properly working classifier.

Starting from the implementation of the Audio Analyser, based on the technical specifications provided in :cite:`mpai-cae`, the module must be able to [#f1]_ :

1. calculate the temporal offset between the audio signal and the video signal [#f2]_ ;
2. detect Irregularities in the audio signal;
3. assign a unique ID to each irregularity;
4. receive an Irregularity File from the Video Analyser AIM and send the identified irregularities to the Video Analyser;
5. extract an Audio File corresponding to each irregularity (both those found in point 1 and those received in point 3);
6. send the Audio Files and the Irregularity File to the Tape Irregularity Classifier AIM.

:numref:`pipeline` provides a general overview of the Audio Analyser.

The operation that this research focuses on is, however, the detection of irregularities in the audio signal (verification of the equalization curve and the playback speed of the tape), the extraction of the corresponding Audio File, and the creation of the Irregularity File for the Video Analyzer. The modules developed for this research don't take in consideration the calculation of the offset between audio and video signals and don't receive the Irregularity File from the Video Analyser.

.. _pipeline:
.. graphviz::
   :alt: Data acquisition pipeline
   :align: center
   :caption: The Audio Analyser AIM pipeline.

   digraph {

      graph [fontsize=15, compound=true, ranksep=0.5, rankdir=LR];
      node [shape=box, fontsize=15, height=1, style=filled];

      subgraph cluster_a {
         node [style=filled];
         label="Audio Analyser AIM";
         color=lightgreen;
         style=filled;

         offset_calculator [label="offset\ncalculator"];
         irregularities [label="irregularities\ndetection"];
         extractor [label="audio\nirregularity\nextractor", height=2];
      }

      input [label="Preservation\nAudio File", shape=oval];
      input2 [label="Preservation\nAudio-Visual File", shape=oval];

      video_analyser [label="Video Analyser AIM", color=lightgreen];
      tape_aim [label="Tape Irregularity\nClassifier AIM", color=lightgreen, height=3];

      irregularities -> extractor;
      video_analyser -> extractor [label="Irreg. File"];
      irregularities -> video_analyser [label="Irreg. File"];
      extractor -> tape_aim [label="Irreg. File"];
      extractor -> tape_aim [label="Audio Blocks"];

      video_analyser -> tape_aim [label="Irreg. File"];
      video_analyser -> tape_aim [label="Irreg. Images"];

      input -> irregularities [lhead=cluster_a];
      input2 -> offset_calculator;
      input2 -> video_analyser;

   }

In this case, the implementation of the Audio Analyser has been divided into two parts: the first part identifies the silence portions in the signal and extracts them into files, while the second part classifies the pre/post equalization curves and the tape playback speed for each extracted file. The results of the classification are then saved in an IrregularityFile.

Find the Irregularities
-----------------------

Before diving into the technical analysis of the software, it is necessary to establish its fundamental requirements. The software is expected to take an audio file or a folder of audio files as input and perform the following tasks: 1) extract silence portions from the input file(s), 2) split multi-channel files into single-channel files, 3) save the extracted silence portions in a folder with an appropriate naming convention, and 4) create a log file in .json format containing information about the extracted silence portions.

Given the requirements, it's trivial to define the main steps of the software:

.. _code-block:

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

The audio portions of interest are the usual A, B, and C class silences (see :ref:`data-analysis`), which are identified as audio signal portions with a noise level below a certain threshold value. The duration of these portions is fixed at 500 milliseconds.

To identify signal portions with power below the established threshold, a linear scan of the input file is performed with a window of 500 milliseconds. If the maximum power of the signal contained in the window is lower than the threshold, then the window is considered a portion of audio signal of interest, the average power of the signal is calculated, and the silence class to which it belongs is determined. Otherwise, the window is discarded and the scan continues, moving the seek point to the next sample after the last peak above the set threshold has been identified.

Classify the Irregularities
---------------------------

The second part of the software is responsible for the classification of the extracted silence portions. The classification is performed by the classifier obtained by training with both Pretto and Berio-Nono datasets to have a better coverage of the features space.

.. code-block::
    :caption: the main steps of the irregularity classification in pseudo-code
    :linenos:

    function classifyIrregularities:
        for each AudioBlock:
            extract its first 13 MFCCs
            classify the AudioBlock with the pre-trained classifier
            map the class to the corresponding IrregularityType

For efficiency reasons, the MFCCs are saved to a DataFrame and then the classification is performed only once on the entire DataFrame, which is much faster than classifying each AudioBlock separately.

The result of the entire pipeline is an IrregularityFile, which contains the classification of each individual block of silence extracted from the input audio file. The file is structured as follows:

.. code-block::
    :caption: the structure of the IrregularityFile

    {
        "Irregularities": [
            {
                "IrregularityID": "00786a08-9020-4a3a-a4ce-6feaec768f3d",
                "Source": "a",
                "TimeLabel": "00:03:05.462",
                "AudioBlockURI": "./AudioBlocks/C_0_17804362_17852362.wav"
            },
            {
                "IrregularityID": "332fdeea-c545-4bb5-afe1-fbbe08fd8207",
                "Source": "a",
                "TimeLabel": "00:03:05.962",
                "AudioBlockURI": "./AudioBlocks/C_0_17852362_17900362.wav"
            },
            {
                "IrregularityID": "ae159ddb-116d-49bb-a9d9-5a1aa3a13c91",
                "Source": "a",
                "TimeLabel": "00:03:06.462",
                "IrregularityType": "ssv",
                "AudioBlockURI": "./AudioBlocks/C_0_17900362_17948362.wav",
                "IrregularityProperties": {
                    "ReadingSpeedStandard": 7.5,
                    "ReadingEqualisationStandard": "IEC1",
                    "WritingSpeedStandard": 3.75,
                    "WritingEqualisationStandard": "IEC2"
                }
            },
            ...
        ]
    }

as can be clearly seen from the structure of the JSON file, each portion of silence is classified, and various types of irregularities can occur even within a single audio file. The following modules after the audio analyzer are responsible for handling this information and making decisions based on it. For example, if the type of irregularity changes constantly from a certain point onwards, it can be inferred that the tape contains multiple recordings. On the other hand, if irregularities occur sporadically, it can be inferred that the tape contains only one recording and that the rare differences are due to classification errors [#f3]_ .

.. rubric:: Footnotes

.. [#f1] The following steps have not to be strictly followed in this order, neither they have to respect this separation in the implementation.

.. [#f2] Since the operation of starting and stopping the playback of the tape and the video recording is subject to latencies due to the hardware used and is not always engineered in the same way, the time offset between the audio signal and the video signal can be highly variable.

.. [#f3] Clearly, the following modules do not only perform this task, but the one exemplified is the most evident that can be appreciated from the output of the audio analyzer.