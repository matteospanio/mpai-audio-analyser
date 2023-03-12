.. eq_detection documentation master file, created by
   sphinx-quickstart on Wed Dec 21 00:02:22 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Equalization Detection
======================

.. epigraph::

   I want to make one minor quibble about the idea that there's Mathematics behind anything: there is no math behind anything. Math is one of the languages that we use to describe the world, computational languages are also languages that we use to describe the world.

   -- Allen Downey, Scipy Conference, 2015

.. only:: html

   **eq-detection** is a Python package for the implementation of a part of the equalization detection in MPAI-CAE ARP (Audio Recording Preservation) Standard. This package is composed of 5 main parts:

   - **audiohandler**: a module for handling audio files.  It is based on the :py:mod:`wave` and :py:mod:`numpy` modules. It has been written to have a fine grained control over the saving and loading of audio files, specifing the sample rate, the number of channels and the bit depth. It is not intended to be a full-featured audio library, but rather a tool for quickly and easily manipulating audio files without dealing with many features that are not needed, basing instead on just essential dependencies.
   - **ml**: a module to simplify machine learning workflow and studies. It is based on the standard Python data analysis stack for classical machine learning: :py:mod:`sklearn`, :py:mod:`matplotlib`, :py:mod:`seaborn`, :py:mod:`pandas`. This module facilitates the use of machine learning algorithms (especially to automatize the parameters tuning process), the visualization of the results and the loading of specific datasets. This module has been specificly designed to handle the datasets and analysis workflows specified in :ref:`data-analysis`.
   - **noise-extractor** : a CLI tool for automatic noise detection and extraction from audio files. It is based on :py:mod:`argparse` and :py:mod:`audiohandler` modules. It has been designed to be easily integrable in the MPAI-CAE ARP standard workflow.
   - **noise-classifier** : a CLI tool for automatic recognition of audio irregularities from audio files. It is based on :py:mod:`argparse` and :py:mod:`audiohandler` modules. It has been designed to be easily integrable in the MPAI-CAE ARP standard workflow.
   - **audio-analyser** : a CLI tool for the analysis of audio files. It is based on :py:mod:`argparse` and :py:mod:`audiohandler` modules. It has been designed to be easily integrable in the MPAI-CAE ARP standard workflow.

.. only:: not html


   This thesis is written as a narrative that aims to investigate the possibility of recognizing specific anomalies of magnetic tapes that are created during the reproduction phase through machine learning, and to study how to integrate the models into a software workflow. The need for a standard that brings order in the current scenario of software that makes use of AI is presented in Chapter 1, while Chapter 2 describes the possibilities of representing a signal in a digital way by presenting the particular case of the preservation of magnetic tapes. In addition to representing the tapes digitally, metadata linked to the specific medium must also be included. Chapters 3 and 4 build on the knowledge presented in the previous chapters to conduct experiments using machine learning algorithms for clustering and classification.

   The analysis section contains the results of the experiments carried out to test the models. The results are presented in the form of tables and graphs, which are explained in the text. The python notebooks used to generate the results are available in the project repository on GitHub, where also a usage guide is available and the API documentation. An html version of the thesis with integrated notebooks and API guide is hosted at `https://mpai-audio.static.app <https://mpai-audio.static.app>`_.

   .. topic:: Conventions used

      Bibliographic references are indicated by a specific number in square brackets like :cite:`downey2014`, which is then mapped in the bibliography. If the reference is inside a paragraph and is the first time it appears in the text, the number in square brackets is preceded by the author name as this :cite:t:`downey2014`.

      The references to figures, tables and code listings are indicated by the number of the object:

      - this is a reference to a table, :numref:`svm-classification-results`,
      - this is a reference to a figure, :numref:`classification_25_pretto`,
      - this is a reference to a code section, :numref:`code-block`

      Inter-chapter references are composed by the title of the section referred and the page number specification between round braces, like this: :ref:`mpai`.

   The target reader of this thesis is expected to have prior knowledge of machine learning algorithms for clustering and classification, while aspects of signal processing and audio analysis are briefly introduced in the text to provide context.

.. toctree::
   :numbered:
   :maxdepth: 3
   :caption: Table of contents:

   .. sections/usage
   .. sections/api/index
   .. auto_examples/index
   sections/mpai
   sections/audio
   sections/data_analysis/index
   sections/audio_analyser
   sections/conclusion
   .. sections/glossary
   sections/bibliography
