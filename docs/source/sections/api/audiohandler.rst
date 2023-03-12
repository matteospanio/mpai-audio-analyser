audiohandler
------------
.. currentmodule:: audiohandler

As explained in the :ref:`audio` section, a digital signal is represented as a flow of samples over the time and a series of informations: the sampling rate, the number of channels, the bit depth, the encoding, etc. All the latest informations are stored in the headers of the binary audio file and are used to properly decode the signal data. A natural consequence of this representation is the class :py:class:`AudioWave` which is used to store the signal data and the header informations.
Essentially it contains the most important information about a digitalized audio wave, which are:

* bit depth: the number of bits used in the quantization process (8, 16, 24, 32);
* number of channels: the number of channels of the audio wave, usually 1 for mono and 2 for stereo;
* sample rate: the number of samples per second measured in Hertz (Hz);
* data: the array containing the sampled and quantized values of the signal.

.. note::

    One of the features of the :py:class:`AudioWave` class is the possibility to extract portions of the signal data based on its intensity. This is done by the ``get_silence_slices`` method which returns a dictionary containing the start and end indexes of a portion divided by type of noise. The extraction is based on the :py:class:`Noise` class which is used to store the information about a class of signals which amplitute lies within a certain range, as described in :cite:`10.1162/comj_a_00487`.

.. automodule:: audiohandler
    :inherited-members:

Submodules
++++++++++

Here is a list of all the submodules of the :py:mod:`audiohandler` module:

.. autosummary::
   :toctree: gen_modules/
   :template: module.rst

    audiohandler.utils
