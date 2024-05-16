import cv2
import dlib
import numpy as np
from scipy import ndimage

### Hàm
# Thay đổi kích thước ảnh với tỉ lệ ban đầu
def resize(img, width):
    ratio = float(width) / img.shape[1]
    size = (width, int(img.shape[0] * ratio))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img

# Kết hợp hình ảnh có kênh alpha trong suốt
def blend_transparent(face_img, glasses_img):
    overlay_img = glasses_img[:, :, :3]
    overlay_mask = glasses_img[:, :, 3:]

    background_mask = 255 - overlay_mask

    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

### Thông số
# Tìm khuôn mặt (Bounding Box)
face_detector = dlib.get_frontal_face_detector()
# Tìm điểm khuôn mặt (Landmark)
landmark_detector = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
# Tạo model
ageProto = "model/age_deploy.prototxt"
ageModel = "model/age_net.caffemodel"
genderProto = "model/gender_deploy.prototxt"
genderModel = "model/gender_net.caffemodel"
# Tạo mạng
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Create Parameter
# Giá trị trung bình đề dịch kiếng vào trọng tâm
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
padding = 20
# Danh sách output tuổi và giới tính
ageList = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']
genderList = ['Male', 'Female']
# Gán giá trị mặc định
gender = 'Unknown'
age = 'Unknown'

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.glasses = ''

    def __del__(self):
        self.video.release()

    def set_Glasses(self, GlassesID):
        self.glasses = cv2.imread(GlassesID, -1)

    def get_frame(self):
        ret, frame = self.video.read()

        frame = resize(frame, 700)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            # Phát hiện khuôn mặt
            faces = face_detector(gray_frame, 1)
            if faces:
                # Lấy vị trí của khuôn mặt
                for face in faces:
                    face_left = face.left()
                    face_top = face.top()
                    face_right = face.right()
                    face_bottom = face.bottom()
                    faceBox = [face_left, face_top, face_right, face_bottom]


                    ##############   Tìm các điểm trên khuôn mặt   ##############

                    landmarks = landmark_detector(gray_frame, face).parts()

                    landmarks = np.matrix([[landmark.x, landmark.y] for landmark in landmarks])

                    for index, point in enumerate(landmarks):
                        location = (point[0, 0], point[0, 1])
                        if index == 0:
                            eye_left = location
                        elif index == 16:
                            eye_right = location
                        try:
                            # Tính góc giữa 2 mắt để xoay kiếng
                            degree = np.rad2deg(np.arctan2(eye_left[0] - eye_right[0], eye_left[1] - eye_right[1]))
                        except:
                            pass

                    ##############   Dự đoán giới tính và tuổi  ##############

                    global gender
                    global age
                    face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                           max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]

                    cv2.putText(frame, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

                    ##############   Thay đổi kích thước và xoay kiếng   ##############

                    if self.glasses == '':
                        Text = 'Choose the glasses!'
                        cv2.putText(frame, Text, (frame.shape[0] // 5, frame.shape[1] // 3), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        # Thay đổi kích thước kiếng theo kích thước khuôn mặt
                        face_width = faceBox[2] - faceBox[0]
                        glasses_resized = resize(self.glasses, face_width)
                        # Xoay kiếng theo khuôn mặt
                        glasses_height, glasses_width, _ = glasses_resized.shape
                        glasses_resized_rotated = ndimage.rotate(glasses_resized, (degree + 90))
                        # Lấy điểm trục tung giữa 2 mắt
                        eye_center = (eye_left[1] + eye_right[1]) / 2
                        # Tỉ lệ điều chỉnh kiếng xuống từ điểm đầu
                        scaleDown = 0.2
                        # Thông số điều chỉnh kiếng xuống từ điểm đầu
                        glass_trans = int(scaleDown * (eye_center - faceBox[1]))
                        # Lấy điểm gắn kiếng và xoay
                        glasses_mask_location_rotated = ndimage.rotate(frame[faceBox[1] + glass_trans:faceBox[1] + glasses_height + glass_trans, faceBox[0]:faceBox[2]], (degree + 90))
                        height, wight, _ = glasses_mask_location_rotated.shape
                        # Đổi kích thước điểm gắn kiếng
                        glasses_mask_location_rotated_resized = frame[
                                                                faceBox[1] + glass_trans:faceBox[1] + height + glass_trans,
                                                                faceBox[0]:faceBox[0] + wight]

                        # Kết hợp 2 phần vào với nhau
                        try:
                            blend_glasses = blend_transparent(glasses_mask_location_rotated_resized, glasses_resized_rotated)
                        except:
                            pass

                        # Gắn phần trộn vào khung hình
                        frame[faceBox[1] + glass_trans: faceBox[1] + height + glass_trans, faceBox[0]:faceBox[0] + wight] = blend_glasses

            # Trường hợp không tìm được khuôn mặt
            else:
                Text = 'No face detected!'
                cv2.putText(frame, Text, (frame.shape[0] // 4, frame.shape[1] // 3), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 0, 255), 2, cv2.LINE_AA)
        except:
            pass

        ret, jpeg = cv2.imencode('.jpeg', frame)

        result = [jpeg.tobytes(), gender, age]

        return result
