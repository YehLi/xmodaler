================================================
Benchmarks
================================================

Image Captioning on MSCOCO (Cross-Entropy Loss)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `Up-Down <https://drive.google.com/file/d/1giOJ5llaNjXz2JClN3Mqe93VIy1Fu5pq/view?usp=sharing>`_, 76.3, 60.3, 46.6, 36.0, 27.6, 56.6, 113.1, 20.7
   `Transformer <https://drive.google.com/file/d/1Q6Tt2z_NKmnr0ai0uRRNyap2-DxxM7Wy/view?usp=sharing>`_, 76.4, 60.3, 46.5, 35.8, 28.2, 56.7, 116.6, 21.3
   `X-LAN <https://drive.google.com/file/d/1zgUWEDD7EiRyih8G_DyE6unshjKjeKjV/view?usp=sharing>`_, 77.5, 61.9, 48.3, 37.5, 28.6, 57.6, 120.7, 21.9
   TDEN, 75.5, 59.4, 45.7, 34.9, 28.7, 56.7, 116.3, 22

Image Captioning on MSCOCO (CIDEr Score Optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `Up-Down <https://drive.google.com/file/d/1tHM06k413ANuAr7a5jCAtKeN_lQ-ieBk/view?usp=sharing>`_, 80.1, 64.3, 49.7, 37.7, 28.0, 58.0, 124.7, 21.5
   `Transformer <https://drive.google.com/file/d/1y3E4t5pQUuvN_gB_tgBVX9HvzM5QSex5/view?usp=sharing>`_, 80.5, 65.4, 51.1, 39.2, 29.1, 58.7, 130.0, 23.0
   `X-LAN <https://drive.google.com/file/d/13b6nhbnq4h8JKbS0oQB_F2tnRUiUt5g-/view?usp=sharing>`_, 80.4, 65.2, 51.0, 39.2, 29.4, 59.0, 131, 23.2
   TDEN, 81.3, 66.3, 52.0, 40.1, 29.6, 59.8, 132.6, 23.4

Video Captioning on MSVD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `MP-LSTM <https://drive.google.com/file/d/1EsVTLlRcviUsz9RpHCQijyZk5l_Wy0B0/view?usp=sharing>`_, 76.1, 64.2, 55.1, 45.9, 31.6, 67.0, 70.8, 4.6
   `TA <https://drive.google.com/file/d/1sZT7bOG9qa6Ho2ptpe_B1CNw6GfdHt_4/view?usp=sharing>`_, 79.0, 66.9, 57.6, 48.2, 32.0, 68.0, 74.4, 4.6
   
Video Captioning on MSR-VTT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `MP-LSTM <https://drive.google.com/file/d/1ZU6Fv2aJddSphpSc9pac3cK_IcSh5o_J/view?usp=sharing>`_, 72.8, 60.2, 48.8, 38.6, 25.8, 58.3, 40.1, 5.6
   `TA <https://drive.google.com/file/d/1Dm7CToj71RawjKd1fqSe_yCE0CGPktvK/view?usp=sharing>`_, 72.3, 60.3, 49.3, 39.3, 25.8, 58.8, 41.5, 5.6

Visual Question Answering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table::
   :header: Model, Overall, Yes/No, Number, Other
   :widths: auto

   TDEN, 72.5, 88.5, 54.7, 63.0

Caption-based image retrieval on Flickr30k
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table::
   :header: Model, R1, R5, R10
   :widths: auto

   TDEN, 63.6, 88.2, 92.9


