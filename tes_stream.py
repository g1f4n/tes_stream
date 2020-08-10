# from main_code import liveness_detection
import face_recognition
import cv2
import numpy as np
import mysql.connector
import pickle
import requests
import imutils
import threading
from flask import render_template
from flask import Response
from flask import Flask
import json
from imutils.video import VideoStream
from time import strftime
import math
import time

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

connection = mysql.connector.connect(
    host='localhost', user='root', passwd='', database='sadewa_absensi')

# video_capture = cv2.VideoCapture(0)
video_capture = VideoStream(src=0).start()
time.sleep(2.0)

# Get a reference to webcam #0 (the default one)


def face_video_streaming():
    global video_capture, outputFrame, lock

    # create cursor
    cursor = connection.cursor()

    # select data from database
    cursor.execute(
        """SELECT nis, nama_siswa, data_wajah FROM tbl_siswa""")

    # dump the result to a string
    rows = cursor.fetchall()
    # get the result
    encoded = {}
    for each in rows:
        nis = each[0]
        nama_siswa = each[1]
        data_wajah = pickle.loads(each[2])
        encoded[nis + "_" + nama_siswa] = data_wajah

    # list face
    known_face_encodings = list(encoded.values())
    known_face_names = list(encoded.keys())

    # Load a sample picture and learn how to recognize it.
    # obama_image = face_recognition.load_image_file("img_reco/MUH_HUSAIN_GIFFARY_ALSAERA.jpg")
    # obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # # Load a second sample picture and learn how to recognize it.
    # biden_image = face_recognition.load_image_file("img_reco/Bayu.jpg")
    # biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Create arrays of known face encodings and their names
    # known_face_encodings = [
    #     obama_face_encoding,
    #     biden_face_encoding
    # ]
    # known_face_names = [
    #     "Gifan",
    #     "Aditya Bayu"
    # ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    list_names = []
    list_time = []
    process_this_frame = True

    start_time = time.time()
    elapstime = 0

    while True:
        # Grab a single frame of video
        # ret, frame = video_capture.read()
        frame = video_capture.read()
        frame = imutils.resize(frame, width=1080, height=600)
        foto = frame.copy()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                #     # send data to api
                #     _, foto_wajah = cv2.imencode('.jpg', foto)
                #     nama = name + "_" + \
                #         "{}.jpg".format(strftime("%Y%m%d%H%M%S"))
                #     response = requests.post(
                #         'http://192.168.43.38:4000/absen/' + nama, data=foto_wajah.tostring())
                #     print(response.text)
                # else:
                #     name = 'Unknown'

                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # send data to api
                # _, foto_wajah = cv2.imencode('.jpg', foto)
                # nama = name + "_" + \
                #     "{}.jpg".format(strftime("%Y%m%d%H%M%S"))
                # response = requests.post(
                #     'http://192.168.43.38:4000/absen/' + nama, data=foto_wajah.tostring())
                # print(response.text)

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)

                # accuracy face
                face_match_threshold = 0.4
                face_distance = min(face_distances)

                # confidence
                if face_distance > face_match_threshold:
                    range = (1.0 - face_match_threshold)
                    linear_val = (1.0 - face_distance) / (range * 2.0)
                    percentage = linear_val * 100.0
                    result = str(round(percentage, 2)) + "%"
                else:
                    range = face_match_threshold
                    linear_val = 1.0 - (face_distance / (range * 2.0))
                    calculate = linear_val + \
                        ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
                    percentage = calculate * 100.0
                    result = str(round(percentage, 2)) + "%"

                best_match_index = np.argmin(face_distances)

                if result > '70.0%' or result == '100.0%':

                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        splitName = name.split("_")
                        elapstime = time.time() - start_time
                        mergeNameElapstime = splitName[0] + \
                            "_" + str(start_time)

                        if splitName[0] not in list_names:
                            list_names.append(splitName[0])
                            list_time.append(mergeNameElapstime)
                            # send data to api
                            _, foto_wajah = cv2.imencode('.jpg', foto)
                            nama = name + "_" + \
                                "{}.jpg".format(strftime("%Y%m%d%H%M%S"))
                            response = requests.post(
                                'http://192.168.43.38:4000/absen/' + nama, data=foto_wajah.tostring())
                            print(response.text)
                            # print("absen ke 1 " + splitName[0])
                        else:
                            for i in list_time:
                                explodeListTime = i.split("_")
                                timeCount = time.time() - \
                                    float(explodeListTime[1])
                                if splitName[0] == explodeListTime[0] and timeCount > 30.0:
                                    # send data to api
                                    _, foto_wajah = cv2.imencode('.jpg', foto)
                                    nama = name + "_" + \
                                        "{}.jpg".format(
                                            strftime("%Y%m%d%H%M%S"))
                                    response = requests.post(
                                        'http://192.168.43.38:4000/absen/' + nama, data=foto_wajah.tostring())
                                    print(response.text)
                                    # print("absen Ke 2 " + explodeListTime[0])
                                    for niss, i in enumerate(list_time):
                                        explodeI = i.split("_")
                                        if explodeI[0] == explodeListTime[0]:
                                            list_time[niss] = explodeListTime[0] + \
                                                "_" + str(time.time())
                                else:
                                    pass
                        # send data to api
                        # _, foto_wajah = cv2.imencode('.jpg', foto)
                        # nama = name + "_" + \
                        #     "{}.jpg".format(strftime("%Y%m%d%H%M%S"))
                        # response = requests.post(
                        #     'http://192.168.43.38:4000/absen/' + nama, data=foto_wajah.tostring())
                        # print(response.text)
                    else:
                        name = 'Unknown'
                else:
                    name = 'Unknown'

                face_names.append(name)
                # else:
                #     name = 'Unknown'
                #     face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)
            # cv2.putText(frame, str(elapstime), (left + 17, bottom - 17),
            #             font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/face_video_reco")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start a thread that will perform motion detection
    t = threading.Thread(target=face_video_streaming)
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host='0.0.0.0', port=5000, debug=True,
            threaded=True, use_reloader=False)

# Release handle to the webcam
# video_capture.release()
cv2.destroyAllWindows()
video_capture.stop()


# face_video_streaming()
