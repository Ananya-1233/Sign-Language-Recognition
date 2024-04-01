# Sign language recognition using CNN

# Dataset

Word level American Sign Langauge dataset is a public dataset that contains 2000 glosses from the English language. The glosses are words such as how, where, which etc. 
The advantage of using words as glosses is that signs can directly be converted into words, not needing the user to spell each word alphabetically. This saves time and improves the efficiency of the model. On the other hand, time and space requirements are increased, making the process of training tedious and heavy. 
For each word in the dataset, nearly 7 videos of about 2-3 seconds exist. The labels, i.e., glosses are marked using json file. Duration of the video, label, number of frames etc., are some of the features mentioned in the json file. It also contains frames to which the glosses belong and bounding boxes to represent the location of where the sign was generated in the video.

Source: Kaggle
Size: 5 GB


# Methodology

3D CNNs are used to extract features from videos which are first divided into frames using opencv.

A model trained on WLASL detects the sign and predicts a class.

The predicted class is shown with the video as a caption.

Using Seq2Seq translation technique, words in the form of glosses are translated into sentences.

The generated sentences are stored in a text file which is downloadable using UI.

Django is used for backend. Whereas, HTML and CSS cover the frontend. 
