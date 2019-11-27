
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os


class EmotionDetection:
    def __init__(self, base_path, num_emotions, input_model):
        self.base_path = base_path
        print("[STATUS:] Loading Face Detection Network... Please wait")
        self.face_net = cv2.dnn.readNetFromCaffe(os.path.join(self.base_path, 'output', 'deploy.prototxt.txt'),
                                                 os.path.join(self.base_path, 'output', 'face.caffemodel'))

        print("[STATUS:] Loading Emotion Classification Network... Please wait")
        self.emo_net = load_model(os.path.join(base_path, 'output', input_model))

        if num_emotions == 7:
            self.emotion_list = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
        elif num_emotions == 6:
            self.emotion_list = ['angry', 'scared', 'happy', 'sad', 'surprised', 'neutral']

    def image_resize(self, image, width=None, height=None):  # aspect aware image resizing using CV2
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
        return resized

    def detect_emotions_image(self, input_image, output_image):
        if type(input_image) is str:
            input_img = cv2.imread(input_image, cv2.IMREAD_ANYCOLOR)

        input_img_clone = input_img.copy()
        (height, width) = input_img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(input_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        print("\n Computing Detections...")
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        # loop over the individual detections (for regression based bounding boxes and non-maximal suppression
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")     # now we get the bounding box

                # use this bounding box to detect the emotions of that particular face
                gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

                roi = gray[startY:endY, startX:endX]

                # resize the ROI to (48, 48, 1) for emotion network
                roi = cv2.resize(roi, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
                roi = roi.astype(np.float32) / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # predict the class label using the emotion classifier
                predicted_class = np.argmax(self.emo_net.predict(roi))
                predicted_label = self.emotion_list[predicted_class]

                # put the rectangle, face probability and emotion probability onto the image
                output_text = "Emotion = {0}:".format(predicted_label)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(input_img_clone, (startX, startY), (endX, endY), (0, 255, 0), 1)
                cv2.putText(input_img_clone, output_text, (startX, y), cv2.FONT_HERSHEY_TRIPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Output", input_img_clone)
        cv2.waitKey(0)
        cv2.imwrite(os.path.join(self.base_path, 'output', output_image), input_img_clone)

    def detect_emotions_video(self, video_path):
        if video_path is None:
            vid_input_stream = cv2.VideoCapture(0)
        else:
            vid_input_stream = cv2.VideoCapture(video_path)

        while True:
            (grabbed, frame) = vid_input_stream.read()
            if grabbed is None:
                break
            frame = self.image_resize(frame, width=320)
            det_frame = self.detect_emotions_image(frame)
            cv2.imshow('Video Stream', det_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        vid_input_stream.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path",
                        type=str, required=True)

    parser.add_argument("-n", "--num_emotions",
                        type=int, required=True)

    parser.add_argument("-m", "--input_model",
                        type=str, required=True)

    parser.add_argument("-i", "--image_path",
                        type=str, required=True)

    parser.add_argument("-o", "--output_path",
                        type=str)

    args = parser.parse_args()

    emo_detect = EmotionDetection(base_path=args.base_path, num_emotions=args.num_emotions,
                                  input_model=args.input_model)

    if args.output_path is None:
        output_path = os.path.join(args.base_path, 'output', 'TestOutput.jpg')
    else:
        output_path = args.output_path

    emo_detect.detect_emotions_image(input_image=args.image_path, output_image=output_path)

    """
    Will add the video input later (mutually exclusive group with image
    """
