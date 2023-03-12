.. _data-analysis:

Data Analysis
================

.. epigraph::

   Without data, you're just another person with an opinion.

   -- W. Edwards Deming

Given the abstractions that allow representing the physical reality of signals in the digital world, it is now possible to analyze how they are integrated into the data analysis process and the implementation of AI modules within the AIF ARP. As previously mentioned, the central points of investigation in this thesis are:

- the project for implementing the Audio Analyser AIM within the AIF ARP;
- the study, through clustering and classification, of the possibilities of automating the recognition of errors resulting from incorrect application of speed or equalization filters during the acquisition of an audio tape.

In this chapter, we are going to evaluate the possibilities and capabilities of automating the detection of such errors based on studies that have been previously conducted on the topic in :cite:t:`Micheloni2017AST` and :cite:t:`10.1162/comj_a_00487`.

One of the fundamental issues is the type of data that needs to be processed. So, before explaining in detail the procedures used and the results, it is necessary to study a methodology for sound analysis and understand which representation is most suitable for our purpose. In :cite:t:`depoli_audio`, a scheme is provided, whose main parts are shown in :numref:`sound-analysis`, which illustrates the different phases of sound analysis. A characteristic aspect of this type of data is *segmentation*, which involves breaking the original signal into multiple parts. In fact, considering audio files with a duration of 10 minutes rather than fragments of 10 milliseconds can strongly influence the results of the investigation [#f1]_ , especially if the signal being considered is highly variable over time. Therefore, in addition to the sampling frequency, it is necessary to establish the frequency at which the signal is analyzed or to establish terms by which to divide the signal into parts in order to obtain homogeneous events. In any case, the final result is an approximation of the signal, as each segment must be considered as a "snapshot" at a specific moment of the input, which is assumed to be constant [#f2]_.

.. _sound-analysis:
.. graphviz::
   :alt: Scheme for sound analysis
   :align: center
   :caption: Scheme for sound analysis.

   digraph {

      graph [fontsize=15,rankdir=LR];
      node [shape=box, fontsize=15, height=1, style=filled];

      data [label="Data Collection"];
      features [label="Features Extraction"];
      classification [label="Classification\nor Clustering"];
      post [label="Post Processing"];

      data -> features [label="Segmentation"];
      features -> classification [label="Features Vector"];
      classification -> post [label="Results", style=dashed];

   }

The second really peculiar aspect of audio data are their features, there are many and each one has its own characteristics. In :cite:`depoli_audio` can be found a detailed description of the most common features used in audio analysis, but in this thesis we are going to focus only on *Mel Frequency Cepstral Coefficients* (MFCCs) which have been used in the studies mentioned above. Just for general knowledge it is important to say that there exist two main types of features: *spectral* and *temporal*. The first type is based on the frequency spectrum of the signal, while the second is based on the time domain. In the case of the spectral features, the most common are the MFCCs.

The MFCCs relies on a chain of transformations of the input signal: first, the signal is transformed into the frequency domain, then the frequency spectrum is transformed into a Mel scale, and finally, the Mel spectrum is transformed into a cepstral domain. Initially, the signal is converted in the frequency domain via DFT (see :ref:`fft`), then the frequency spectrum is converted into the Mel scale, which is a logarithmic scale of frequency correspondant to the response of humans' ears to frequency [#f3]_ :

.. math::
   :label: mel

   \text{mel}(f) =
   \begin{cases}
      f &\quad \text{if }f\leqslant 1000 \text{ Hz}\\
      2595 \log_{10}\left(1+\frac{f}{700}\right) &\quad \text{if }f>1000 \text{ Hz}
   \end{cases}

this operation changes the unit of measure of the frequency by applying a filter-bank to the signal [#f4]_ , where each filter is centered on a specific frequency and has a width that is proportional to the distance from the center frequency. So from a smooth spectrum, the signal is transformed into a series of peaks, each one corresponding to a specific frequency. At this point would be interesting to separate the Mel spectrum into multiple components: that's the concept of *cepstrum* [#f5]_. This is a general method to divide the oscillatory-harmonic components (the base signal) of a sound from its timbric components (the filter-bank). If we think of a signal in the frequency domain as the multiplication of this two components it can be mathematically expressed as :math:`y(n) = x(n) \cdot h(n)` where the spectrum :math:`y` is the resultant from :math:`x`, the oscillatory-harmonic component, and :math:`h`, the timbric component. :cite:`depoli_audio` A further step can be done due to the logarithmic representation of the spectrum, which, thanks to the property of logarithms :math:`\log(a \cdot b) = \log(a) + \log(b)`, consents to pass from a multiplication to a sum, then the discrete time cosine transform (DCT) is applied to the result [#f6]_ :

.. math::
   :label: partial-cepstrum

   \text{DCT}(\log(|Y[k]|)) = \text{DCT}(\log(|X[k]|)) + \text{DCT}(\log(|H[k]|))

where :math:`k` is the :math:`k`-th component of the filter-bank.

As result of this operations the direct formula to extract the :math:`i`-th MFCC is:

.. math::
   :label: mfcc

   C_i = \sum_{k=1}^N Y_k \cos \left[ i \left( k - \frac{1}{2} \right) \frac{\pi}{N} \right]\quad i=1,2,\dots,M

where :math:`Y_1, \dots, Y_N` are the log-energy outputs of a mel-spaced filter-bank and :math:`N` is the total number of channels in the filter-bank.

In :cite:`Micheloni2017AST`, it has been shown that certain portions of audio are more significant than others. In particular, it has been noted that different types of silence (i.e., with different signal intensity) have a good descriptive capability of the considered signal to identify the correct playback speed and equalization curve. :numref:`noise-types` illustrates the three classes of noise that have been identified and their intensity in decibels [#f7]_.

.. _noise-types:
.. csv-table:: Noise classes
   :header: "Class", "Intensity (dB)", "Description"

   "A", "-50 to -63", "Noise in the middle of a recording, i.e., silence between spoken words"
   "B", "-63 to -69", "Noise of the recording head without any specific input signal"
   "C", "-69 to -72", "Noise coming from sections of pristine tape"

By extrapolating sections of silence with a length of 500 milliseconds [#f8]_ , it was proven that MFCCs contain sufficient information to train classifiers with accuracy very close to 100%. While in the study by :cite:`10.1162/comj_a_00487` manually extracting sections of silence was carried out, in this case, a software was created that is capable of automatically identifying sections of silence and extracting MFCCs (with the idea of then integrating it directly into the AIF ARP Audio Analyser).

.. rubric:: Footnotes

.. [#f1] The first option is called *long-time* analysis, often used to analyze the structure of an entire piece of music or for signal that don't vary much over time. In contrast *short-time* analysis is use where the signal varies quite often. In addition considering a 10 minute audio file as a whole it would be interpreted as a single sample, while the same file splitted in 10ms chunks would generate 60000 samples; since in data analysis the sample size matters a lot, it is important to consider the right size of the sample in relation to the quantity of data available.

.. [#f2] For example, if one considers voice recording, the assumption that the signal is constant is justified by the time it takes for the body to move the larynx, mouth, and all the other muscles and organs involved in the process, whose changes are not faster than a few hundred milliseconds, so the signal can be considered constant for a period of time of about 100-200 milliseconds.

.. [#f3] Long story short: in the occidental music theory system the distance between two octaves is always the double in frequency of the lower note (:math:`A_3=220` Hz, :math:`A_4=440` Hz, :math:`A_5=880` Hz), but the space between each octave is always divided linearly in twelve semitones and the human ear perceives these distances as equally distributed. The Mel scale is a logarithmic scale that tries to reproduce the same perception of the human ear, it applies a filter-bank to the signal and then it applies a logarithmic scale to the result. The Mel scale is used in audio analysis because it is more similar to the human perception of sound than the linear scale.

.. [#f4] A filter-bank is a traditional method for analyzing a signal's spectrum. It involves using a set of band-pass filters placed at regular intervals along the frequency axis to obtain the log-energies at the output of each filter. This provides a general idea of the signal's spectral shape and helps to minimize any harmonic structure that may exist. :cite:p:`doi:10.1080/09298219708570724`

.. [#f5] This strange name is derived from inverting the first four letter of *spectrum*: this name reflects that it is the application of a transformation (usually the inverse FFT) to the spectrum, also the unit measure passes from frequency (Hz) to *quefrency* (seconds).

.. [#f6] The DCT is specific for the MFCCs calculation, the general method to evaluate the cepstrum is based on the inverse FFT instead.

.. [#f7] Actually, the intensity value is not exactly absolute, but it could slightly vary depending on the bit depth of the audio file, which can introduce quantization errors. However, the difference is negligible and the values reported in the table are correct for the majority of the cases except for the class :math:`C`, in fact higher bit representaition of the signal have a higher dynamic range and can reach values under :math:`-72` dB (the minimum noise intensity at 16 bit is :math:`-96` dB, while at 24 bit is :math:`-144` dB).

.. [#f8] The length of the silence sections was chosen based on the results of the study in :cite:`Micheloni2017AST` and :cite:`10.1162/comj_a_00487`. In those studies different lengths of silence sections were tested and it was found that longer sections did not enrich the results.

.. toctree::
   :hidden:

   datasets
   clustering
   classification
