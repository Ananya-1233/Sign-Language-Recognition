from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from .forms import VideoForm
from keras.models import load_model
import cv2
import string
from processing import square_pad, preprocess_for_vgg
from model import create_model
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
# Map model names to classes
# MODELS = {
#     "vgg16": VGG16,
#     "inception": InceptionV3,
#     "xception": Xception,
#     "resnet": ResNet50,
#     "mobilenet": MobileNet
# }

# # Define path to pre-trained classification block weights - this is
# vgg_weights_path = "weights/snapshot_vgg_weights.hdf5"
# res_weights_path = "weights/snapshot_res_weights.hdf5"
# mob_weights_path = "weights/snapshot_mob_weights.hdf5"

# my_model = create_model(model='vgg16',
#                         model_weights_path=vgg_weights_path)

model = load_model('CNN_model2.h5')

label_dict = {pos: letter
              for pos, letter in enumerate(string.ascii_uppercase)}
text_file = 'wlasl_class_list.txt'
labels = []

with open(text_file, 'r') as file:
    # labels = [line.strip() for line in f]
    for line in file:
        values = line.strip().split('\t')
        if len(values) > 1:
            label = values[1].strip()
            labels.append(label)



def process_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
        # video_file = request.FILES['videoElement']
        video_path = str(video.video.path)
        cap = cv2.VideoCapture(video_path)

       # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
        def gen_frames():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform sign language translation or any other processing here
                # For this example, we'll overlay some text on the frame

                # hand = frame[83:650, 82:764]
                # hand = square_pad(hand)
                # hand = preprocess_for_vgg(hand)
                
                hand = cv2.resize(frame, (64, 64))  # Repeat the single frame 20 times
                hand = np.expand_dims(hand, axis=0)
                hand = np.repeat(hand, 2, axis=0)
                hand = hand[:, :64, :64, :] 
                hand = np.expand_dims(hand, axis=0)


                # hand = np.repeat(hand, 20, axis=0)  # Repeat the single frame 20 times
                # hand = hand[:, :64, :64, :] 
                # hand = np.expand_dims(hand, axis=0)
                # Make prediction
                preds = model.predict(hand, batch_size=1, verbose=0)
                

                # Predict letter
                top_prd = np.argmax(preds)
                with open('output.txt','w') as f:
                    f.write(str(top_prd))
                # Only display predictions with probabilities greater than 0.5
                #if np.max(my_predict) >= 0.50:
                #  if np.max(preds) >= 0.50 else 0
                prediction_result = (labels[top_prd])
                
                # preds_list = np.argsort(my_predict)[0]
                # pred_2 = label_dict[preds_list[-2]]
                # pred_3 = label_dict[preds_list[-3]]

                # width = int(cap.get(3) + 1)
                # height = int(cap.get(4) + 1)
                cv2.putText(frame, prediction_result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

               # cv2.imshow('Video', frame)
                # out.write(frame)

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break


                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        response = StreamingHttpResponse(gen_frames(), content_type="multipart/x-mixed-replace; boundary=frame")
        return response
        #         cap.release()
        # return redirect('process_video')
    else:
        form = VideoForm()

    return render(request, 'index.html', {'form':form})



















# # @csrf_exempt
# def process_video(request):
#     if request.method == 'POST':
#         form = VideoForm(request.POST, request.FILES)
#         if form.is_valid():
#             video = form.save()
#         # video_file = request.FILES['videoElement']
#         video_path = str(video.video.path)
#         cap = cv2.VideoCapture(video_path)

#        # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
#         def gen_frames():
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Perform sign language translation or any other processing here
#                 # For this example, we'll overlay some text on the frame
#                 hand = frame[83:650, 82:764]
#                 hand = square_pad(hand)
#                 hand = preprocess_for_vgg(hand)

#                 # Make prediction
#                 my_predict = my_model.predict(hand,
#                                             batch_size=1,
#                                             verbose=0)

#                 # Predict letter
#                 top_prd = np.argmax(my_predict)

#                 # Only display predictions with probabilities greater than 0.5
#                 #if np.max(my_predict) >= 0.50:

#                 prediction_result = (label_dict[top_prd] if np.max(my_predict) >= 0.50 else 0)
#                 # preds_list = np.argsort(my_predict)[0]
#                 # pred_2 = label_dict[preds_list[-2]]
#                 # pred_3 = label_dict[preds_list[-3]]

#                 width = int(cap.get(3) + 1)
#                 height = int(cap.get(4) + 1)
#                 cv2.putText(frame, prediction_result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                # cv2.imshow('Video', frame)
#                 # out.write(frame)

#                 # if cv2.waitKey(1) & 0xFF == ord('q'):
#                 #     break


#                 _, buffer = cv2.imencode('.jpg', frame)
#                 frame_bytes = buffer.tobytes()

#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         response = StreamingHttpResponse(gen_frames(), content_type="multipart/x-mixed-replace; boundary=frame")
#         return response
#         #         cap.release()
#         # return redirect('process_video')
#     else:
#         form = VideoForm()

#     return render(request, 'index.html', {'form':form})
