.. _datasets:

Datasets
--------

In :cite:`Micheloni2017AST` the main dataset was artificially generated in laboratory. It was composed by samples that covered all the combinations of correct and incorrect filter chains that could occur during the digitization of audio tapes (including only 7.5 and 15 ips speed tapes).

In the current study the starting dataset is the same but includes also 3.75 ips speed. So all possible combinations for both reading and writing operations are: 3.75 ips with NAB equalization, 7.5 ips with CCIR or NAB equalization and 15 ips with CCIR or NAB equalization (it is not possible to record or reproduce an audio tape at 3.75 ips with CCIR equalization).

The research so has been done in two phases: first, recreate (and confirm) the study in :cite:`Micheloni2017AST` on the extended dataset (referred as *Pretto* or *Pretto dataset* in the rest of the thesis), and on second instance, test the trained models on a new dataset (referred as *Berio-Nono*) generated from real world data.

Once the audio tapes were defined, the data were collected from the digital files using the method described in the section above [#f1]_ (500 ms of silence samples), and then their first 13 MFCCs were computed. :numref:`features` shows the structure of the datasets. All kinds of noises were extracted from the audio files to see if any of them carried more information than the others. Of course, a greater amount of data allows for more coverage of the feature space, but it also increases the computational cost of the training process.

.. _features:
.. csv-table:: Dataset features
   :header: "Feature", "type", "description"
   :widths: 15, 10, 75

   "label", "string", "A string that specifies the equalization curve and speed used in the recording and reproduction processes [#f2]_ (e.g. 3C_7N)"
   "noise_type", "string", "A, B or C"
   "MFCC1-13", "float", "The first 13 MFCCs of the sample"

.. seealso::

  The feature extraction process for both dataset can be recreated executing the notebook ``data_extraction.ipynb`` at the `thesis' repository <https://github.com/matteospanio/mpai-audio-analyser>`_ , and can be summarized as follows:

  1. execute the script ``noise-extractor`` on the audio files to generate a set of samples of 500ms each labeled with the type of noise contained in the sample (A, B or C);
  2. extract the first 13 MFCCs from each sample;
  3. save the results as a csv file containing the MFCCs, the noise type and a label that specifies the equalization curve and speed used in the recording and reproduction processes. 

.. _pretto:

Pretto
++++++

As already mentioned, this dataset consists of a collection of features extracted from a tape created ad hoc at different speeds and with different pre/post-emphasis curves (CCIR or NAB) for both the writing and reading processes. As explained in :ref:`audio`, a tape can be recorded and reproduced with different equalization curves and speeds (3.75, 7.5, 15 ips). That's why this dataset was generated, as it contains all possible main combinations of speeds and equalization curves for the writing and reading processes of the tape. In particular, the tape is composed of a set of audio extracts, each lasting 10 seconds, and the interval between two extracts is composed of a 1-second beep sound, 2 seconds of silence, and another 1-second beep. The audio extracts have been taken from different sources and have been combined as shown in :numref:`audio-tape-composition`.

.. _audio-tape-composition:
.. list-table:: Audio tape composition
   :header-rows: 1

   * - Author
     - Title
     - Duration (s)
   * - Taylor Swift
     - *Shake It Off*
     - 10
   * - The Weeknd
     - *Save Your Tears*
     - 10
   * - Richard Wagner
     - *Ride of the Valkyries*
     - 10
   * - Carl Orff
     - *Carmina Burana - O Fortuna*
     - 10
   * - Queen
     - *We Will Rock You*
     - 10
   * - Eagles
     - *Hotel California*
     - 10
   * - Bruno Maderna
     - *Continuo*
     - 10
   * - Bruno Maderna
     - *Syntaxis*
     - 10
   * - Luciano Berio
     - *Différences*
     - 10
   * - Bruno Maderna
     - *Musica su Due Dimensioni*
     - 10
   * - `CLIPS project <http://www.clips.unina.it/en/index.jsp>`_
     - *LP1f19bZ*
     - 10
   * - `CLIPS project <http://www.clips.unina.it/en/index.jsp>`_
     - *LP4m19bZ*
     - 10
   * - `CLIPS project <http://www.clips.unina.it/en/index.jsp>`_
     - *LP1f20bZ*
     - 10
   * - `CLIPS project <http://www.clips.unina.it/en/index.jsp>`_
     - *LP4m20bZ*
     - 10
   * - `CLIPS project <http://www.clips.unina.it/en/index.jsp>`_
     - *LP4m18bZ*
     - 10

The eclectic intent is evident: different kinds of music have really different MFCCs (in fact, they are often used for automatic music genre recognition), so the dataset comprises classical, pop, and modern music and also speech recordings (CLIPS project has a public corpus of speech recordings).

:numref:`pretto_distribution_matrix` shows the distribution of the dataset taking into consideration different subsets based on noise type. It is noticeable that the dataset is not balanced: tapes recorded at a higher speed tend to have more noise samples when reproduced at a lower speed [#f3]_ . This imbalance could influence the training process. In fact, we will have a heterogeneous distribution of coverage of the feature space. For example, there are only 6 instances of noise :math:`A` recorded at 7.5 ips with CCIR equalization curve and reproduced at 15 ips with NAB equalization curve, while there are 181 instances with the opposite label (7.5 ips CCIR recorded and 15 ips NAB reproduced). Unless the class distribution tends to stick to the same point of the feature space, this could lead to poor generalization of the model.

.. plot:: pyplots/pretto_distribution_matrix.py
    :caption: The dataset distribution by noise type, :math:`y` labels regard the recording processes, :math:`x` labels regard the reproduction processes.

:numref:`dataset-summary` shows a summary of the dataset with some statistics: labels are 25 because each speed can be recorded and reproduced with two different equalization curves except for 3.75 ips, so each tape can be recorded or played with one of 5 configurations: 3.75 ips NAB, 7.5 ips NAB, 7.5 ips CCIR, 15 ips NAB, 15 ips CCIR.

.. _dataset-summary:
.. csv-table:: Pretto Dataset summary
   :header: "Feature", "value"

   "Samples", "9058"
   "Noise types", "3 (A, B, C)"
   "Equalization curves", "2 (NAB, CCIR)"
   "Writing speeds", "3 (3.75, 7.5, 15 ips)"
   "Reading speeds", "3 (3.75, 7.5, 15 ips)"
   "Number of labels", "25"
   "Samples of noise type A", "2075"
   "Samples of noise type B", "5050"
   "Samples of noise type C", "1933"

.. _berio-nono:

Berio-Nono
+++++++++++

This dataset is composed by the digitization of real case scenarios tapes of unpublished works by Luciano Berio and Luigi Nono, in which the tape has been recorded and reproduced with the same equalization curve and speed. Once obtained the digital files, they have been used as input for the pipeline described above. The samples extraction procedure generated 12202 samples, and the features extraction process generated 12202 samples of 13 MFCCs each. The features in the dataframe are the same as in :numref:`features`.

:numref:`berio_distribution_histogram` shows the distribution of the dataset, which is unbalanced, but no evident pattern can be inferred from the distribution since the files were generated from a variable number of different equalized tapes: 4 tapes were recorded at 7.5 ips in NAB format, 6 tapes were recorded at 7.5 ips in CCIR format, 5 tapes were recorded at 15 ips in NAB format, and 3 tapes were recorded at 15 ips in CCIR format. The tapes considered for this dataset had a much longer duration (about half an hour each) compared to the Pretto dataset, which had audio tracks of only 10 seconds.

.. plot:: pyplots/berio_distribution_histogram.py
    :width: 65%
    :caption: The dataset distribution by noise type, each column represents a different equalization curve and speed used in the recording and reproduction processes.

:numref:`berio-dataset-summary` shows a summary of the dataset with some statistics. The labels are 4 because only the correct equalization curves and speeds have been applied in the acquisition step, so each tape can be recorded or played with one of the following 4 configurations: 7.5 ips NAB, 7.5 ips CCIR, 15 ips NAB, 15 ips CCIR. In this case, the extraction software found a noticeable quantity of :math:`C` silence. It is not uncommon to find a great amount of unused tape; however, in this situation, it emerges that the characterization of the noise is missing a 4th kind of noise: when the acquisition process starts, there are a few seconds where the tape is not moving, and neither is the pristine tape. Therefore, the recorder is capturing noise from the environment, usually lower than :math:`-72 dB`. In this case, this noise from the 4th kind was included in the :math:`C` noise type, omitting the lower limit for :math:`C` noise.

.. _berio-dataset-summary:
.. csv-table:: Berio-Nono Dataset summary
   :header: "Feature", "value"

   "Samples", "12202"
   "Noise types", "3 (A, B, C)"
   "Equalization curves", "2 (CCIR or NAB)"
   "Writing speeds", "2 (7.5, 15 ips)"
   "Reading speeds", "2 (7.5, 15 ips)"
   "Number of labels", "4"
   "Samples of noise type A", "1231"
   "Samples of noise type B", "1796"
   "Samples of noise type C", "9175"

Comparision
++++++++++++

The main goal of the entire analysis is to be able to recognize the pre/post-emphasis equalization curves applied to the tape. On the basis of the data we have, the most interesting experiment is therefore to create a classifier trained on the Pretto dataset, i.e. the one built ad hoc and which includes all possible cases, to recognize the examples taken from real cases of the Berio-Nono dataset. To do this, it makes sense to visualize how the data is distributed in space. It is clear that, given the premises, the Berio-Nono dataset should cover (approximately) a subset of the space of the other dataset, but this is not the case. Different factors could the cause of this results:

- the different nature of the tapes in the two datasets, while in Pretto there are short music and speech samples (10 seconds), in Berio-Nono the musical events last for minutes
- the machine used to acquire the tapes has to be recalibrated periodically, it could be that the conditions of the machine were not the same during the acquisiotion of the samples 

Anyway it is interesting to observe that the noise classes are well recognizable along the :math:`x` axis, this means that, even if the Berio-Nono dataset is shifted along the :math:`y` axis in respect to the Pretto dataset, there are evident analogies between the two datasets, while there are groups of samples really far from the others which are probably outliers (a further analysis is needed to confirm this hypothesis, but it is not the purpose of this work).

.. plot:: pyplots/features_space.py
    :caption: The features space of the two datasets, each point represents a sample, the color represents the noise type and the shape represents the dataset (Pretto or Berio-Nono).

.. topic:: Principal component analysis

    The plots have been generated transforming the 13 MFCCs into 2D using the principal component analysis (PCA) algorithm. The PCA algorithm is a linear dimensionality reduction technique that uses singular value decomposition of the data to project it to a lower dimensional space :cite:`scikit-learn`. At each step, the algorithm choise the axis that maximizes the variance of the projected data, and then it projects the data onto that axis. The algorithm is repeated until the desired number of dimensions is reached. This method gives the possibility to visualize complex distributions in a 2D space, anyway this kind of visualization could lead to wrong considerations due to the fact that we are cutting part of the informations.

.. rubric:: Footnotes

.. [#f1] A detailed description of how the software to extract silence can be found at :ref:`audio_analyser`.

.. [#f2] The labels have been composed by a number that specifies the writing speed (3 for 3.75 ips, 7 for 7.5 ips and 15 for 15 ips), a letter that specifies the writing equalization standard used (C for CCIR and N for NAB), an underscore character, a number for the reading speed (analogue as writing) and a letter for the reading equalization standard used. For example the label ``3N_7N`` means that the sample has been recorded at 3.75 ips with NAB equalization curve and reproduced at 7.5 ips with NAB equalization curve.

.. [#f3] Of course this isn't so inespected: reproducing a tape with a lower speed than the one used for recording it, the tape will play slower resulting to have longer noise section of the same intensity as the original ones. It is like stretching the tape: the speed of the tape is slower, the pitch is lower (the frequency has been altered), but the intensity of the sound cannot be alterated since it is interested only by the :math:`y` axis.
