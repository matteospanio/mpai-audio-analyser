The research future
====================

.. epigraph::

    The only true wisdom is in knowing you know nothing.

    -- Socrates

As often happens in research, this work is part of a wider investigation. In the early stages, efforts were made to confirm previous research findings, but then, the ability of the starting data to predict similar but unobserved data was examined. The Pretto dataset was used due to its composition, with the hypothesis that it could cover all possible cases to be tested. Unfortunately, this initial hypothesis was incorrect.

The analysis revealed that the Pretto dataset was inadequate to cover all possible cases, prompting exploration of additional avenues to gather more data. One direction that could be taken is to investigate whether data augmentation techniques could be used to expand the existing dataset. Techniques such as audio stretching, pitch shifting, and noise injection are commonly employed to increase the size of datasets in computer vision and speech recognition applications. These techniques generate additional audio samples that can be used to train machine learning models, leading to a more diverse audio dataset that better represents the range of audio signals in the real world.

Another potential factor identified as a possible cause of this insuccess was the analog recorder used to acquire the tapes. The data was acquired at different times from the same machine, and it is possible that it was not in the same calibration condition, leading to variations in the recorded data. These variations could explain why the distribution of the data was dissimilar and why the data was unable to cover all possible cases.

In light of these findings, it has been realized the need to go back to the drawing board and rethink the analysis approach. It may be that the Mel Frequency Cepstrum Coefficients used to describe the audio signals are not sufficient to capture all the nuances of the audio data, other audio features could be used to better describe the audio signals.

To address these issues, further research was recommended into alternative audio features that could be used to describe the audio signals, and other machine learning models could be tested to better predict the audio data. Additional audio data should also be collected to supplement the existing dataset.

In conclusion, while the initial hypothesis of this work did not hold true, several avenues for future research were revealed that could lead to better predictions of audio data. By exploring data augmentation techniques and alternative audio features, and by using a different machine learning model, the accuracy of predictions may be improved. Collecting more audio data and ensuring that the recorder is well calibrated could help to generate a more comprehensive and representative dataset of real-world audio signals.