================================================
Benchmarks
================================================

Image Captioning on MSCOCO (Cross-Entropy Loss)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `LSTM-A3 <https://drive.google.com/file/d/1kdKHsdOB53AsbsM1aEnAOZEvdkWBzhzA/view?usp=sharing>`_, 75.1, 58.9,	45.3, 34.8, 26.5, 55.4, 107.3, 19.7
   `Up-Down <https://drive.google.com/file/d/1giOJ5llaNjXz2JClN3Mqe93VIy1Fu5pq/view?usp=sharing>`_, 76.3, 60.3, 46.6, 36.0, 27.6, 56.6, 113.1, 20.7
   `GCN-LSTM <https://drive.google.com/file/d/1eLZqt2xS32lUOQibxEDclwANMtska4L9/view?usp=sharing>`_, 76.8, 61.1, 47.6, 36.9, 28.2, 57.2, 116.3, 21.2
   `Transformer <https://drive.google.com/file/d/1Q6Tt2z_NKmnr0ai0uRRNyap2-DxxM7Wy/view?usp=sharing>`_, 76.4, 60.3, 46.5, 35.8, 28.2, 56.7, 116.6, 21.3
   `Meshed-Memory Transformer <https://drive.google.com/file/d/1n_ytLQmR4Cg-SK9T116Wlp3xM4gbcOCt/view?usp=sharing>`_, 76.1, 59.9, 46.2, 35.7, 27.8, 56.3, 114.5, 20.7
   `X-LAN <https://drive.google.com/file/d/1zgUWEDD7EiRyih8G_DyE6unshjKjeKjV/view?usp=sharing>`_, 77.5, 61.9, 48.3, 37.5, 28.6, 57.6, 120.7, 21.9
   `TDEN <https://drive.google.com/file/d/19alfPj-gIudoL5CHsS4VwhfnU-FhTXW3/view?usp=sharing>`_, 75.5, 59.4, 45.7, 34.9, 28.7, 56.7, 116.3, 22.0


Image Captioning on MSCOCO (CIDEr Score Optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `LSTM-A3 <https://drive.google.com/file/d/1MqHUBWt20qPfM7T6IczZMLviL1d7nbbW/view?usp=sharing>`_, 78.0, 61.7, 46.9, 35.2, 26.9, 56.3, 116.5, 20.3
   `Up-Down <https://drive.google.com/file/d/1tHM06k413ANuAr7a5jCAtKeN_lQ-ieBk/view?usp=sharing>`_, 80.1, 64.3, 49.7, 37.7, 28.0, 58.0, 124.7, 21.5
   `GCN-LSTM <https://drive.google.com/file/d/1qwilTeK2WQCZEDXcJAmmteLZfLOEhg7P/view?usp=sharing>`_, 80.2, 64.7, 50.3, 38.5, 28.5, 58.4, 127.2, 22.1
   `Transformer <https://drive.google.com/file/d/1y3E4t5pQUuvN_gB_tgBVX9HvzM5QSex5/view?usp=sharing>`_, 80.5, 65.4, 51.1, 39.2, 29.1, 58.7, 130.0, 23.0
   `Meshed-Memory Transformer <https://drive.google.com/file/d/1cPyWLPoq81XQaC9KLPefFb1vUg0dnDHB/view?usp=sharing>`_, 80.7, 65.1, 50.7, 38.8, 28.8, 58.0, 130.0, 22.3
   `X-LAN <https://drive.google.com/file/d/13b6nhbnq4h8JKbS0oQB_F2tnRUiUt5g-/view?usp=sharing>`_, 80.4, 65.2, 51.0, 39.2, 29.4, 59.0, 131.0, 23.2
   `TDEN <https://drive.google.com/file/d/1GTbbwfbJHIu6uDmcLY-pedCiuWHyR7nK/view?usp=sharing>`_, 81.3, 66.3, 52.0, 40.1, 29.6, 59.8, 132.6, 23.4

Video Captioning on MSVD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `MP-LSTM <https://drive.google.com/file/d/1NDjaCyBntQZI3ehQ8QyUMTMrb1e6Dgsp/view?usp=sharing>`_, 77.0, 65.6, 56.9, 48.1, 32.4, 68.1, 73.1, 4.8
   `TA <https://drive.google.com/file/d/1SqvugATqHU3Le1jtTQKnL3FADJ7kbJK0/view?usp=sharing>`_, 80.4, 68.9, 60.1, 51.0, 33.5, 70.0, 77.2, 4.9
   `Transformer <https://drive.google.com/file/d/1NlwZrAhGE9RPbWdypVz-Tkirt4u8E1t0/view?usp=sharing>`_, 79.0, 67.6, 58.5, 49.4, 33.3, 68.7, 80.3, 4.9
   `TDConvED <https://drive.google.com/file/d/1Th9FJe8o_4bMULuoCKqDHP_4Faa0RabZ/view?usp=sharing>`_, 81.6, 70.4, 61.3, 51.7, 34.1, 70.4, 77.8, 5.0
   
Video Captioning on MSR-VTT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `MP-LSTM <https://drive.google.com/file/d/1OBhtruTexuYV_MbiUL4obfUoNKZbEiUd/view?usp=sharing>`_, 73.6, 60.8, 49.0, 38.6, 26.0, 58.3, 41.1, 5.6
   `TA <https://drive.google.com/file/d/126nPL9lC6_Qa6_hMs32V1zSsJSDxpR9-/view?usp=sharing>`_, 74.3, 61.8, 50.3, 39.9, 26.4, 59.4, 42.9, 5.8
   `Transformer <https://drive.google.com/file/d/1u6mh13eKd93Y_OoSnxk_d9BI5yi_3Vd-/view?usp=sharing>`_, 77.1, 61.6, 47.9, 36.4, 26.7, 57.7, 43.1, 6.3
   `TDConvED <https://drive.google.com/file/d/1A3OGvjCpXUI6p1vy1qbNTVGLy5a0b3Dc/view?usp=sharing>`_, 81.6, 70.4, 61.3, 51.7, 34.1, 70.4, 77.8, 5.0

Visual Question Answering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table::
   :header: Model, Overall, Yes/No, Number, Other
   :widths: auto

   `Uniter <https://drive.google.com/file/d/1cjBAeYSuSEN_IlQCnqtIoalkATMSQs87/view?usp=sharing>`_, 70.1, 86.8, 53.7, 59.6
   TDEN,  71.1, 87.4, 53.3, 61.2

Caption-based image retrieval on Flickr30k
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table::
   :header: Model, R1, R5, R10
   :widths: auto

   `Uniter <https://drive.google.com/file/d/1hvoWMmHjSvxp3zqW10L7PoBQGbxM9MiF/view?usp=sharing>`_, 61.6, 87.7, 92.8
   `TDEN <https://drive.google.com/file/d/1SqYscN6UCbifxhMJ-ScpiLgWepMSx7uq/view?usp=sharing>`_, 62.0, 86.6, 92.4 

Visual commonsense reasoning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table::
   :header: Model, Q -> A, QA -> R, Q -> AR
   :widths: auto

   `Uniter <https://drive.google.com/file/d/1eYTa6HlndaRkJa3LHFTpnRRwKJZniBhZ/view?usp=sharing>`_, 71.7, 73.1, 52.9
   TDEN, 75.2, 76.7, 58.1

