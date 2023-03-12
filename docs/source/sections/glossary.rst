Glossary
=================

The following is a list of terms extracted from the MPAI-CAE technical specifications, its purpose is to provide a quick reference for the terms used across the document and in the source code as bindings to mantain consistency across a standard ideal interface for the MPAI-CAE projects.

.. glossary::

    Audio
        Digital representation of an analogue audio signal sampled at a frequency between 8-192 kHz with a number of bits/sample between 8 and 32.

    Audio Block
        A set of consecutive audio samples.
    
    Audio Channel
        A sequence of Audio Blocks.

    Audio File
        A .wav file.

    Audio Object
        Direct audio source which is in the audible frequency band.

    Audio Segment
        An Audio Block with Start Time and an Ent Time Labels corresponding to the time of the first and last sample of the Audio Segment, respectively.

    Editing list
        The description of the speed, equalisation and reading backwards corrections occurred during the restoration process.

    Interleaved Multichannel Audio
        A data structure containing at least 2 time-aligned interleaved Audio Channels.

    Irregularity
        An event of interest to preservation from in Table 21 and Table 22 of the MPAI-CAE technical specifications.

    Irregularity File
        A JSON file containing information about Irregularities of the ARP input.

    JSON
        JavaScrit Object Notation.

    Multichannel Audio
        A set of multiple time-aligned Audio Channels.

    Preservation Audio File
        The input Audio File resulting from the digitisation of an audio open-reel tape to be preserved and, in case, restored.

    Preservation Audio-Visual File
        The input Audio-Visual File produced by a camera pointed to the playback head of the magnetic tape recorder and the synchronised Audio resulting from the tape digitisation process.

    Preservation Image
        A Video frame extracted from Preservation Audio-Visual File.

    Preservation Master Files
        Set of files providing the information stored in an audio tape recording without any restoration. As soon as the original analogue recordings is no more accessible, it becomes the new item for long-term preservation.

    Restored Audio Files
        Set of Audio Files derived from the Preservation Audio File, where potential speed, equalisation or reading backwards errors that occurred in the digitisation process have been corrected.

    Restored Audio Segment
        An Audio Segment in which the entire segment has been replaced by a synthetic speech segment, or in which each Damaged Segment has been replaced by a synthetic speech segment