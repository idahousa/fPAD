import cv2
import os
import dlib
import support_functions as sp
from  keras.layers import *

def rect_to_box(i_rect):
    """
    +) Convert from rectangular of box to a bounding box
    :param i_rect:= (x1,x2,x3,x4) that represent the coordinates of left,right,up,bottom points of a rect
    :return: a box with (x,y,width,height) of a bounding box.
    """
    _x = i_rect.left() #TOP-LEFT POSITION
    _y = i_rect.top()  #TOP-LEFT POSITION
    _width = i_rect.right() - _x + 1   #Width of face box
    _height = i_rect.bottom() - _y + 1 #Height of face box
    return _x,_y,_width,_height

def shape_to_ndarray(i_shape,i_num_points=68):
    """
    :param i_shape: the list of points which specify a shape of face
    :param i_num_points:
    :return:an array of the points which specify the shape
    +) #The use of this one is just for compatible with array manupulation.
    """
    _rtn_array = np.zeros(shape=[i_num_points,2],dtype=np.int)
    for index in range(i_num_points):
        _rtn_array[index]=(i_shape.part(index).x,i_shape.part(index).y)
    return _rtn_array

def draw_circle_on_image(i_image,i_point,i_radius = 1, i_color = 'red'):
    """
    :param i_image: an input image to draw a circle on it
    :param i_point: the center point where we draw a circle
    :param i_radius: the radius of circle point
    :param i_color: color of circle
    :return: an image with draw circle on it
    """
    _shape = i_image.shape
    if len(_shape)==2: #i_image is a gray image
        for _x in range(i_radius*2):
            for _y in range(i_radius*2):
                _pos_x = i_point[0] + _x-i_radius
                _pos_y = i_point[1] + _y-i_radius
                i_image[_pos_y,_pos_x]=255
        return i_image
    elif len(_shape)==3:#i_image is a color image
        if i_color == 'red':
            for _x in range(i_radius * 2):
                for _y in range(i_radius * 2):
                    _pos_x = i_point[0] + _x - i_radius
                    _pos_y = i_point[1] + _y - i_radius
                    i_image[_pos_y, _pos_x, 0] = 255
                    i_image[_pos_y, _pos_x, 1] = 0
                    i_image[_pos_y, _pos_x, 2] = 0
            return i_image
        elif i_color == 'blue':
            for _x in range(i_radius * 2):
                for _y in range(i_radius * 2):
                    _pos_x = i_point[0] + _x - i_radius
                    _pos_y = i_point[1] + _y - i_radius
                    i_image[_pos_y, _pos_x, 0] = 0
                    i_image[_pos_y, _pos_x, 1] = 255
                    i_image[_pos_y, _pos_x, 2] = 0
            return i_image
        else:
            return i_image
    else:
        return i_image

def draw_shape_on_image(i_image,i_shape,i_radius=1,i_color='red'):
    """
    :param i_image: an input image
    :param i_shape: an shape (face)
    :param i_radius: radius of circle at each circle point.
    :param i_color: color of circle
    :return: an image with draw-shape
    """
    if len(i_shape.shape) !=2:
        return i_image
    _num_points,_ = i_shape.shape
    for _index in range(_num_points):
        i_point = i_shape[_index,:]
        i_image = draw_circle_on_image(i_image=i_image,i_point=i_point,i_radius=i_radius,i_color=i_color)
    return i_image

def find_faces(i_image,i_detector,i_up_sample_level=1,i_rtn_box=True):
    """
    :param i_image: an input face image
    :param i_detector: a face detector
    :param i_up_sample_level: level of up-samplie to find a small face
    :param i_rtn_box: = False => Return the bounding rect of face with format (x1,y1,x2,y2)
    :param i_rtn_box: = True  => Return the bounding box of face with format (x,y,widht,height)
    :return:
    """
    _faces = i_detector(i_image,i_up_sample_level)
    if i_rtn_box:
        o_rtn_faces = []
        for _face in enumerate(_faces):
            _x, _y, _width, _height = rect_to_box(_face)
            o_rtn_faces.append([_x, _y, _width, _height])
        return o_rtn_faces
    else:
        return _faces

def find_small_faces(i_image,i_detector,i_max_upsample_level=2,i_rtn_box=True):
    """
    :param i_image:
    :param i_detector:
    :param i_max_upsample_level:
    :param i_rtn_box:= True  => Return the bounding box of face with format (x,y,widht,height)
    :param i_rtn_box:= False => Return the bounding rect of face with format (x1,y1,x2,y2)
    :return:
    """
    for level in range(i_max_upsample_level):
        #print('Level {}'.format(level))
        o_faces = find_faces(i_image=i_image,i_detector=i_detector,i_up_sample_level=level+1,i_rtn_box=i_rtn_box)
        if len(o_faces)>0:
            return o_faces
    return []

def find_facial_landmarks_rect(i_image,i_shape_predictor,i_face_rect):
    """
    => Detect facial landmark points of a face using dlib.
    => Reference: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    => rect = (x1,x2,x3,x4) that represent the coordinates of left,right,up,bottom points of a rect
    => Note: _face_rect is a result of face detection
    :param i_image:
    :param i_shape_predictor:
    :param i_face_rect:
    :return:
    """
    _shape = i_shape_predictor(i_image,i_face_rect)
    o_shape = shape_to_ndarray(_shape)
    return o_shape

def find_facial_landmarks_image(i_image,i_face_detector,i_face_shape_detector):
    #Detect any face from input image
    _faces = find_small_faces(i_image=i_image,i_detector=i_face_detector,i_rtn_box=False)
    _num_face = len(_faces)
    if _num_face<=0:
        return np.zeros(shape=[1,1,1])
    o_shapes = np.zeros(shape=[_num_face,68,2],dtype=np.int)
    for _index,_face in enumerate(_faces):
        _shape = find_facial_landmarks_rect(i_image=i_image,i_shape_predictor=i_face_shape_detector,i_face_rect=_face)
        o_shapes[_index]=_shape
    return o_shapes

def rotate_image(i_image,i_center,i_angle):
    #Rotate the i_image around i_center with i_angle degree.
    #i_angle is the rotation angle in degree
    _height,_width,_nchannel = i_image.shape
    _rotate_matrix = cv2.getRotationMatrix2D(center=i_center,angle=i_angle,scale=1)
    i_image = cv2.warpAffine(src=i_image,M=_rotate_matrix,dsize=(_width,_height))
    return i_image,_rotate_matrix

def find_center_of_point_sequence(i_shape,i_start=0,i_stop=68):
    #i_shape is the (68,2) landmark points
    _x = 0
    _y = 0
    _cnt = 0
    for _index in range(68):
        if (_index>=i_start) and (_index<i_stop):
            _x += i_shape[_index, 0]
            _y += i_shape[_index, 1]
            _cnt +=1
    _x /= _cnt
    _y /= _cnt
    return _x,_y

def normalize_face(i_image,i_shape):
    #i_shape is an numpy.ndarray of the shape of (68,2) that specifies the landmark coordinate of landmark points
    _left_eye_x, _left_eye_y = find_center_of_point_sequence(i_shape=i_shape, i_start=36, i_stop=42)
    _right_eye_x, _right_eye_y = find_center_of_point_sequence(i_shape=i_shape, i_start=42, i_stop=48)
    _center_face_x, _center_face_y = find_center_of_point_sequence(i_shape=i_shape, i_start=31, i_stop=36)
    _diff_x = _right_eye_x - _left_eye_x
    _diff_y = _right_eye_y - _left_eye_y
    _rotate_angle = np.arctan(_diff_y/_diff_x) #Return as the angle from [-pi/2 pi/2]
    _rotate_angle *= (180./np.pi)
    o_image,rot_matrix = rotate_image(i_image=i_image,i_center=(_center_face_y,_center_face_x),i_angle=_rotate_angle)
    o_shape = np.zeros_like(i_shape)
    for _index in range(68):
        o_shape[_index, 0] = i_shape[_index, 0] * rot_matrix[0, 0] + i_shape[_index, 1] * rot_matrix[0, 1] + rot_matrix[0, 2]
        o_shape[_index, 1] = i_shape[_index, 0] * rot_matrix[1, 0] + i_shape[_index, 1] * rot_matrix[1, 1] + rot_matrix[1, 2]
    return o_image,o_shape

def extract_mixed_face(i_image,i_shape,i_dsize=(224,224),i_extend_factor = 1.0):
    _face = extract_face(i_image=i_image,i_shape=i_shape,i_dsize=i_dsize,i_extend_factor=i_extend_factor)
    _leye = extract_left_eye(i_image=i_image,i_shape=i_shape,i_dsize=i_dsize)
    _reye = extract_right_eye(i_image=i_image,i_shape=i_shape,i_dsize=i_dsize)
    #cv2.imwrite('models/_face.jpg',_face)
    #cv2.imwrite('models/_leye.jpg',_leye)
    #cv2.imwrite('models/_reye.jpg',_reye)
    _face[:,:,0] = np.uint8(np.divide(np.sum(_face,axis=2),3))
    _face[:,:,1] = np.uint8(np.divide(np.sum(_leye,axis=2),3))
    _face[:,:,2] = np.uint8(np.divide(np.sum(_reye,axis=2),3))
    #cv2.imwrite('models/_face_mixed.jpg',_face)
    #cv2.imwrite('models/_face_gray.jpg', _face[:,:,0])
    #cv2.imwrite('models/_leye_gray.jpg', _face[:,:,1])
    #cv2.imwrite('models/_reye_gray.jpg', _face[:,:,2])
    return _face

def extract_face(i_image,i_shape,i_dsize=(224,224),i_extend_factor = 1.0):
    """
    :param i_image: an input image
    :param i_shape: detected face shape
    :param i_dsize: destination size of extracted image
    :param i_extend_factor: ratio of taking image
    :return: face image in size of dsize.
    """
    _height,_width,_nchannel = i_image.shape
    o_image, o_shape = normalize_face(i_image=i_image,i_shape=i_shape)
    _min_x,_min_y = o_shape.min(axis=0)
    _max_x,_max_y = o_shape.max(axis=0)
    _face_width = _max_x - _min_x
    _face_height = _max_y - _min_y
    _extend_size_x = _face_width*(i_extend_factor - 1.0)
    _extend_size_y = _face_height*(i_extend_factor - 1.0)
    _extend_size_x = int(_extend_size_x/2)
    _extend_size_y = int(_extend_size_y/2)
    _min_x -= _extend_size_x
    _max_x += _extend_size_x
    _min_y -= _extend_size_y
    _max_y += _extend_size_y
    if _min_x<0:
        _min_x=0
    if _min_y<0:
        _min_y=0
    if _max_x>_width:
        _max_x=_width
    if _max_y>_height:
        _max_y=_height
    _face_roi = o_image[_min_y:_max_y,_min_x:_max_x,:]
    return cv2.resize(src=_face_roi,dsize=i_dsize)

def extract_left_eye(i_image,i_shape,i_dsize=(224,224)):
    # Resize to a specific size
    _height, _width, _nchannel = i_image.shape
    o_image, o_shape = normalize_face(i_image=i_image, i_shape=i_shape)
    _min_x = o_shape[17,0]
    _min_y = o_shape[19,1]
    _max_x = o_shape[29,0]
    _max_y = o_shape[29,1]
    if _min_x < 0:
        _min_x = 0
    if _min_y < 0:
        _min_y = 0
    if _max_x > _width:
        _max_x = _width
    if _max_y > _height:
        _max_y = _height
    _face_roi = o_image[_min_y:_max_y, _min_x:_max_x, :]
    return cv2.resize(src=_face_roi, dsize=i_dsize)

def extract_right_eye(i_image,i_shape,i_dsize=(224,224)):
    # Resize to a specific size
    _height, _width, _nchannel = i_image.shape
    o_image, o_shape = normalize_face(i_image=i_image, i_shape=i_shape)
    _min_x = o_shape[29,0]
    _min_y = o_shape[24,1]
    _max_x = o_shape[26,0]
    _max_y = o_shape[29,1]
    if _min_x < 0:
        _min_x = 0
    if _min_y < 0:
        _min_y = 0
    if _max_x > _width:
        _max_x = _width
    if _max_y > _height:
        _max_y = _height
    _face_roi = o_image[_min_y:_max_y, _min_x:_max_x, :]
    return cv2.resize(src=_face_roi, dsize=i_dsize)

def extract_face_full(i_image,i_face_detector,i_shape_predictor,i_dsize,i_extend_factor=1.0):
    _shapes = find_facial_landmarks_image(i_image=i_image,
                                          i_face_detector=i_face_detector,
                                          i_face_shape_detector=i_shape_predictor)
    [_num_faces, _num_point, _num_coordinate] = _shapes.shape
    if _num_faces==1 and _num_point==1 and _num_coordinate==1:
        #No face was detected
        return []
    o_ext_faces = []
    for _index in range(_num_faces):
        _shape = _shapes[_index, :, :].reshape((_num_point, _num_coordinate))
        _face  = extract_face(i_image=i_image, i_shape=_shape,i_dsize=i_dsize,i_extend_factor=i_extend_factor)
        o_ext_faces.append(_face)
    return o_ext_faces

def extract_mixed_face_full(i_image,i_face_detector,i_shape_predictor,i_dsize,i_extend_factor=1.0):
    """
    :param i_image:
    :param i_face_detector:
    :param i_shape_predictor:
    :param i_dsize:
    :param i_extend_factor:
    :return:
    """
    _shapes = find_facial_landmarks_image(i_image=i_image,
                                          i_face_detector=i_face_detector,
                                          i_face_shape_detector=i_shape_predictor)
    [_num_faces, _num_point, _num_coordinate] = _shapes.shape
    if _num_faces==1 and _num_point==1 and _num_coordinate==1:
        #No face was detected
        return []
    o_ext_faces = []
    for _index in range(_num_faces):
        _shape = _shapes[_index, :, :].reshape((_num_point, _num_coordinate))
        _face  = extract_mixed_face(i_image=i_image, i_shape=_shape,i_dsize=i_dsize,i_extend_factor=i_extend_factor)
        o_ext_faces.append(_face)
    return o_ext_faces

def extract_face_from_video(i_video_path,
                            i_face_detector,
                            i_shape_predictor,
                            i_dsize = (224,224),
                            i_extend_factor=1.0,
                            i_mixed_face = False,
                            i_save_sequence_flag=False):
    """
    :param i_video_path:
    :param i_face_detector:
    :param i_shape_predictor:
    :param i_dsize:
    :param i_extend_factor:
    :param i_mixed_face:
    :param i_save_sequence_flag:
    :return:
    """
    #Using opencv to read an video file and extract frames
    cap = cv2.VideoCapture(i_video_path)
    if not cap.isOpened():
        return False
    o_faces = []
    while True:
        _ret, _frame = cap.read()
        if not _ret:
            break
        if i_mixed_face:
            _faces = extract_mixed_face_full(i_image=_frame,
                                             i_face_detector=i_face_detector,
                                             i_shape_predictor=i_shape_predictor,
                                             i_dsize=i_dsize,
                                             i_extend_factor=i_extend_factor)
        else:
            _faces = extract_face_full(i_image=_frame,
                                       i_face_detector=i_face_detector,
                                       i_shape_predictor=i_shape_predictor,
                                       i_dsize=i_dsize,
                                       i_extend_factor=i_extend_factor)
        for _face in _faces:
            o_faces.append(_face)
    cap.release()
    if i_save_sequence_flag:
        save_sequence_path = os.path.split(i_video_path)
        video_file_name = save_sequence_path[1]
        video_file_name = video_file_name[0:len(video_file_name)-len(sp.get_file_extension(video_file_name))-1]+'.npy'
        save_sequence_path = os.path.join(save_sequence_path[0],video_file_name)
        print(save_sequence_path)
        np.save(save_sequence_path,o_faces)
    return o_faces

if __name__=="__main__":
    print('Module to extract human face from input image!')
    face_detector = dlib.get_frontal_face_detector()
