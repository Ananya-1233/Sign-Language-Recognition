# Sign language recognition using CNN

# Dataset

Word level American Sign Langauge(WLASL) dataset is a public dataset that contains 2000 glosses from the English language. The glosses are words such as how, where, which etc. 
The advantage of using words as glosses is that signs can directly be converted into words, not needing the user to spell each word alphabetically. This saves time and improves the efficiency of the model. On the other hand, time and space requirements are increased, making the process of training tedious and heavy. 
For each word in the dataset, nearly 7 videos of about 2-3 seconds exist. The labels, i.e., glosses are marked using json file. Duration of the video, label, number of frames etc., are some of the features mentioned in the json file. It also contains frames to which the glosses belong and bounding boxes to represent the location of where the sign was generated in the video.

Source: Kaggle

Size: 5 GB


# Methodology

A 3d convolutional neural network is used as the baseline model for the data. The three dimensions that are taken into account are the height, width and duration of the frames in the video file. By considering the time factor as well, the model is able to learn the statistical correlation between multiple frames of the video. The videos are first broken down into frames using the opencv library. The frames are used as the input for the 3D-CNN model. Features are extracted from the input frames and passed on to subsequent layers for processing.

The model contains two Conv3D layers with 32 and 64 filters each of size (2,2) with 'relu' activation. Each convolutional layer is followed by a max pool layer of size (3,3). A dropout layer is applied at the end of the max pool layer to reduce overfitting. The feature map outputs are flattened and passed on to a fully connected layer of neurons equal to the number of classes.

Loss: Categorical cross entropy\
Optimizer: Adam\
Metric: Accuracy\
A model trained on WLASL detects the sign and predicts a class.

The predicted class is shown with the video as a caption.

Using Seq2Seq translation technique, words in the form of glosses are translated into sentences.

The generated sentences are stored in a text file which is downloadable using UI.

Django is used for backend. Whereas, HTML and CSS cover the frontend. 
