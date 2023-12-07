## Sign language recognition using CNN

# Dataset

Word level American Sign Langauge dataset is a public dataset that contains 2000 glosses from the English language. The glosses are words such as how, where, which etc. 

The advantage of using words as glosses is that signs can directly be converted into words. It does not need the user to spell each word alphabetically. 

Each gloss has nearly 7000 videos of about 2-3 seconds. The labels, i.e., glosses are marked using json file which maps the videos with the glosses. Additionally, the json file also contains frames to which the glosses belong and bounding boxes to represent the location of where the sign was generated in the video.

The dataset is extracted from Kaggle and is of 5 GB.


#Methodology

3D CNNs are used to extract features from videos which are first divided into frames using opencv.

A model trained on WLASL detects the sign and predicts a class.

The predicted class is shown with the video as a caption.

Using Seq2Seq translation technique, words in the form of glosses are translated into sentences.

The generated sentences are stored in a text file which is downloadable using UI.

Django is used for backend. Whereas, HTML and CSS cover the frontend. 
