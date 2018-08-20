import dlib
import os
import cv2
import face_lib
import keras
import numpy as np
import support_functions as sp
import dnn_models
from keras.models import load_model

class casia_data_object:
    def __init__(self,
                 dataset_path='datasets/casia/',
                 mode='train',
                 sub_db_mode='all',
                 save_face_extern = 'faces',
                 face_extend_factor = 1.0,
                 save_image_flag = False):
        self.dataset_path = dataset_path
        self.face_shape_predictor_path = 'face_shape_predictor/sp68.dat'
        self.face_size  = (224,224)
        self.data = []
        self.labels = []
        self.save_image_flag = save_image_flag
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_shape_predictor = dlib.shape_predictor(self.face_shape_predictor_path)
        self.mode = mode
        self.sub_db_mode = sub_db_mode
        self.face_extend_factor = face_extend_factor
        self.save_face_extern = '_{}_{}.npy'.format(save_face_extern, face_extend_factor)
        if save_face_extern=='mixed_faces':
            self.mixed_face = True
        else:
            self.mixed_face = False
        if self.mode=='train':
            self.dataset_path = os.path.join(self.dataset_path,'train_release')
        elif self.mode=='test':
            self.dataset_path = os.path.join(self.dataset_path, 'test_release')
        else:
            self.dataset_path = os.path.join(self.dataset_path, 'train_release')
        if self.sub_db_mode == 'low_quality':
            self.video_name_list = ['2', '4','6','8']
        elif self.sub_db_mode == 'norm_quality':
            self.video_name_list = ['1', '3', '5', '7']
        elif self.sub_db_mode == 'high_quality':
            self.video_name_list = ['HR_1','HR_2','HR_3','HR_4']
        elif self.sub_db_mode == 'wrap_photo':
            self.video_name_list = ['1', '2', '3', '4', 'HR_1', 'HR_2']
        elif self.sub_db_mode == 'cut_photo':
            self.video_name_list = ['1', '2', '5', '6', 'HR_1', 'HR_3']
        elif self.sub_db_mode == 'video':
            self.video_name_list = ['1', '2', '7', '8', 'HR_1', 'HR_4']
        else:
            self.video_name_list = ['1',  # Real, Normal Quality
                                    '2',  # Real, Low Quality
                                    '3',  # Fake, Normal Quality, Wrap Photo
                                    '4',  # Fake, Low Quality, Wrap Photo
                                    '5',  # Fake, Normal Quality, Cut Photo
                                    '6',  # Fake, Low Quality, Cut Photo
                                    '7',  # Fake, Normal Quality, Video Access
                                    '8',  # Fake, Low Qualtiy, Video Access
                                    'HR_1',  # Real, High Quality
                                    'HR_2',  # Fake, High Quality, Wrap Photo
                                    'HR_3',  # Fake, High Quality, Cut Photo
                                    'HR_4',  # Fake, High Quality, Video Access
                                    ]
        self.load_data()
        #----------------------------------------End of initialiazation------------------------------------------------#
    def load_data(self):
        _user_list = os.listdir(self.dataset_path)
        #print(_user_list)
        for _user_name in _user_list:
            _user_data_path = os.path.join(self.dataset_path,_user_name)
            _video_list = os.listdir(_user_data_path)
            #print(_video_list)
            for _video_name in _video_list:
                _video_path = os.path.join(_user_data_path,_video_name)
                _video_name = _video_name[0:len(_video_name)-len(sp.get_file_extension(_video_name))-1]
                _npy_file_path = os.path.join(_user_data_path,_video_name+self.save_face_extern)
                if _video_name in self.video_name_list:
                    if os.path.exists(_npy_file_path):
                        #print('Load from npy file.....................................................................')
                        print('=>', _npy_file_path)
                        _video_faces = np.load(_npy_file_path)
                    else:
                        #print('Load from avi file.....................................................................')
                        print('=>',_video_path)
                        _video_faces = face_lib.extract_face_from_video(i_video_path=_video_path,
                                                                        i_face_detector=self.face_detector,
                                                                        i_shape_predictor=self.face_shape_predictor,
                                                                        i_extend_factor=self.face_extend_factor,
                                                                        i_dsize=self.face_size,
                                                                        i_mixed_face=self.mixed_face,
                                                                        i_save_sequence_flag=False)
                        np.save(_npy_file_path,_video_faces)
                    #print('Video file {}.avi => Num detected faces = {}'.format(_video_name, len(_video_faces)))
                    if _video_name == '1' or _video_name=='2' or _video_name=='HR_1':
                        _video_label = 0 #Real face
                    else:
                        _video_label = 1 #Fake face
                    self.data.append(_video_faces)
                    self.labels.append(_video_label)
                    #Save Image for testing.............................................................................
                    if self.save_image_flag:
                        _save_image_path = os.path.join(_user_data_path, _video_name)
                        if not os.path.exists(_save_image_path):
                            os.mkdir(_save_image_path)
                        for _index, _face in enumerate(_video_faces):
                            _save_image_full_path = os.path.join(_save_image_path, '{}.png'.format(_index))
                            cv2.imwrite(_save_image_full_path, _face)
                else:
                    continue
        return self

class casia_data_generator(keras.utils.Sequence):
    def __init__(self,
                 i_data = None,
                 i_labels = None,
                 i_dsize = (224,224),
                 i_timestep = 5,
                 i_batch_size = 5,
                 i_z_sscore = True,
                 i_num_aug = -1, #Smaller or equal to zero means DONOT perform data augmentation.
                 i_image_gap = 1,
                 i_mean_image=None,
                 i_std_image = None):
        self.data = i_data
        self.labels = i_labels
        self.dsize = i_dsize
        self.timestep = i_timestep
        self.batch_size = i_batch_size
        self.z_score = i_z_sscore
        self.num_aug = i_num_aug
        self.image_gap = i_image_gap
        self.shift_amount_x = 11 #5% of horizontal axix
        self.shift_amount_y = 11 #5% of vertical axix
        self.sigma = 1e-10
        self.num_real_sequence, self.num_fake_sequence = self.cal_num_original_sequence(self.data, self.labels,self.timestep,self.image_gap)
        _max_num_sequences = max(self.num_real_sequence, self.num_fake_sequence)
        if _max_num_sequences == self.num_real_sequence:
            if self.num_aug<=0:
                self.num_aug_real=1
                self.num_aug_fake=1
            else:
                self.num_aug_real = self.num_aug
                self.num_aug_fake = int(np.ceil((self.num_real_sequence * self.num_aug_real) / self.num_fake_sequence))
        else:
            if self.num_aug<=0:
                self.num_aug_real=1
                self.num_aug_fake=1
            else:
                self.num_aug_fake = self.num_aug
                self.num_aug_real = int(np.ceil((self.num_fake_sequence * self.num_aug_fake) / self.num_real_sequence))
        print('Num AUG Real = ',self.num_aug_real)
        print('Num AUG Fake = ',self.num_aug_fake)
        self.num_image_sequences = self.num_real_sequence * self.num_aug_real + self.num_fake_sequence * self.num_aug_fake
        print('Total Sequence = ',self.num_image_sequences)
        self.image_sequence_index = np.zeros(shape=[self.num_image_sequences, 4], dtype=np.int64)
        self.decode_table = np.zeros(shape=[25,2],dtype=np.int)
        self.make_lookup_table()
        if self.z_score:
            if (i_mean_image is not None) and (i_std_image is not None):
                """
                => This is the testing data in the case of using z-score normalization
                """
                self.mean_image = i_mean_image
                self.std_image  = i_std_image
                self.mode = 'test'
            else:
                """
                => This is the training data in the case of using z-score normalization
                """
                self.mean_image= np.zeros(shape=self.dsize,dtype=np.float64)
                self.std_image = np.ones(shape=self.dsize,dtype=np.float64)
                self.get_mean_std_images()
                self.mode='train'
        else:
            """
            => This is the training and testing data in the case of NOT usiing z-score normalization
            """
            self.mean_image = 0
            self.std_image = 1
            self.mode='train'
    @staticmethod
    def cal_num_original_sequence(i_data,i_labels,i_timestep,i_image_gap):
        _num_real_sequences=0
        _num_fake_sequences=0
        _num_videos = len(i_data)
        for _v_index in range(_num_videos):
            _num_images = len(i_data[_v_index])
            _num_current_sequence = int(_num_images - (i_timestep-1)*i_image_gap)
            if i_labels[_v_index]==0:
                _num_real_sequences +=_num_current_sequence
            elif i_labels[_v_index]==1:
                _num_fake_sequences += _num_current_sequence
            else:
                pass
        print('Num Original REAL Sequences = {}'.format(_num_real_sequences))
        print('Num Original FAKE Sequences = {}'.format(_num_fake_sequences))
        return int(_num_real_sequences),int(_num_fake_sequences)

    @staticmethod
    def decode_augumentation(i_code,i_decode_table):
        """
        :param i_code:
        :param i_decode_table:
        :return:
        """
        _height,_width = i_decode_table.shape
        if i_code>=_height:
            _x,_y = 0,0
        else:
            _x = i_decode_table[i_code,0]
            _y = i_decode_table[i_code,1]
        return _x,_y
    @staticmethod
    def do_augmentation(i_image, i_num_shift_x, i_num_shift_y, i_dsize):
        """
        :param i_image:
        :param i_num_shift_x:
        :param i_num_shift_y:
        :param i_dsize:
        :return:
        """
        _height, _width, _nchannel = i_image.shape
        _begin_x = int(i_num_shift_x)
        _begin_y = int(i_num_shift_y)
        _end_x = int(_begin_x + _width)
        _end_y = int(_begin_y + _height)
        if _begin_x < 0:
            _begin_x = int(0)
        if _begin_y < 0:
            _begin_y = int(0)
        if _end_x > _width:
            _end_x = int(_width)
        if _end_y > _height:
            _end_y = int(_height)
        o_image = i_image[_begin_y:_end_y, _begin_x:_end_x, :]
        return cv2.resize(o_image, i_dsize)

    def make_lookup_table(self):
        decode_table = []
        _total_num_sequences = 0
        _num_videos = len(self.data)
        for _v_index in range(_num_videos):
            _num_images = len(self.data[_v_index])
            for _i_index in range(_num_images):
                if _i_index>=(self.timestep-1)*self.image_gap:
                    if self.labels[_v_index] == 0:
                        for _aug_index in range(self.num_aug_real):
                            self.image_sequence_index[_total_num_sequences, 0] = _v_index
                            self.image_sequence_index[_total_num_sequences, 1] = _i_index
                            self.image_sequence_index[_total_num_sequences, 2] = _aug_index
                            self.image_sequence_index[_total_num_sequences, 3] = 0
                            _total_num_sequences += 1
                    elif self.labels[_v_index] == 1:
                        for _aug_index in range(self.num_aug_fake):
                            self.image_sequence_index[_total_num_sequences, 0] = _v_index
                            self.image_sequence_index[_total_num_sequences, 1] = _i_index
                            self.image_sequence_index[_total_num_sequences, 2] = _aug_index
                            self.image_sequence_index[_total_num_sequences, 3] = 1
                            _total_num_sequences += 1
                    else:
                        pass
        print('Total number of image sequences = {}'.format(_total_num_sequences))
        decode_table.append([ 0,  0])
        decode_table.append([ 1,  0])
        decode_table.append([ 0,  1])
        decode_table.append([-1,  0])
        decode_table.append([ 0, -1])
        decode_table.append([ 1,  1])
        decode_table.append([-1,  1])
        decode_table.append([-1, -1])
        decode_table.append([ 1, -1])
        decode_table.append([ 2,  0])
        decode_table.append([ 0,  2])
        decode_table.append([-2,  0])
        decode_table.append([ 0, -2])
        decode_table.append([ 2,  2])
        decode_table.append([-2,  2])
        decode_table.append([-2, -2])
        decode_table.append([ 2, -2])
        self.decode_table = np.array(decode_table)
        return self

    def get_mean_std_images(self):
        _total_image = 0
        _dsize = (self.dsize[0], self.dsize[1])
        print('(000/100)||'+'.'*100+'||')
        _prev_ratio=0
        for _index in range(self.num_image_sequences):
            _sequence_index  = self.image_sequence_index[_index, 0]  # As my design
            _image_index     = self.image_sequence_index[_index, 1]  # As my design
            _aug_image_index = self.image_sequence_index[_index, 2]  # As my design
            _label           = self.image_sequence_index[_index, 3]  # As my design
            _ratio = int((_index*100)/self.num_image_sequences)
            if _ratio>_prev_ratio:
                _prev_ratio=_ratio
                print('({:0>3d}/100)||'.format(_prev_ratio)+'='*(_prev_ratio-1)+'>'+'.'*(100-_prev_ratio)+'||')
            for _prev_i_index in range(self.timestep):
                _image = self.data[_sequence_index][_image_index - _prev_i_index*self.image_gap]
                _x, _y = self.decode_augumentation(_aug_image_index, self.decode_table)
                _image = self.do_augmentation(_image, _x * self.shift_amount_x, _y * self.shift_amount_y,_dsize)
                _image = _image.astype(dtype=np.float)
                self.mean_image = np.add(self.mean_image, _image)
                self.std_image = np.add(self.std_image, np.power(_image, 2))
                _total_image +=1
        self.mean_image = np.divide(self.mean_image, _total_image)  # Final E[X]
        self.std_image = np.divide(self.std_image, _total_image)  # Final E[X^2]
        _diff_std_mean = np.subtract(self.std_image, np.power(self.mean_image, 2))
        self.std_image = np.add(np.sqrt(_diff_std_mean), self.sigma)
        print('Total Number of Images = {}'.format(_total_image))
        return True

    def save_mean_std_image(self,save_path):
        _save_mean = np.uint8(self.mean_image)
        _save_std = sp.ndarray_to_image(self.std_image)
        if self.mode == 'train':
            dlib.save_image(_save_mean,'{}/mean_image_train.jpg'.format(save_path))
            dlib.save_image(_save_std, '{}/std_image_train.jpg'.format(save_path))
        else:
            dlib.save_image(_save_mean, '{}/mean_image_test.jpg'.format(save_path))
            dlib.save_image(_save_std, '{}/std_image_test.jpg'.format(save_path))

    def __len__(self):
        return np.int64(np.ceil(self.num_image_sequences/self.batch_size))

    def __getitem__(self, item):
        if item==0:
            print('Performing shuffling data at the begining of each epoch!...')
            np.random.shuffle(self.image_sequence_index)
        _begin_batch_segment = item * self.batch_size
        _end_batch_segment = (item + 1) * self.batch_size
        _batch_index = self.image_sequence_index[_begin_batch_segment:_end_batch_segment]
        _dsize = (self.dsize[0], self.dsize[1])
        batch_data = []
        batch_labels = []
        for _index in _batch_index:
            _sequence_index  = _index[0]  # As my design
            _image_index     = _index[1]  # As my design
            _aug_image_index = _index[2]  # As my design
            _label           = _index[3]  # As my design
            batch_labels.append(_label)
            _current_sequence = []
            for _prev_i_index in range(self.timestep):
                _image = self.data[_sequence_index][_image_index - _prev_i_index*self.image_gap]
                _x, _y = self.decode_augumentation(_aug_image_index, self.decode_table)
                _image = self.do_augmentation(_image,_x * self.shift_amount_x,_y * self.shift_amount_y,_dsize)
                _image = _image.astype(dtype=np.float)
                if self.z_score:
                    _image = np.divide(np.subtract(_image, self.mean_image), self.std_image)
                _current_sequence.append(_image)
            batch_data.append(_current_sequence)
        batch_data = np.array(batch_data)
        batch_labels = np.array(keras.utils.to_categorical(batch_labels, num_classes=2))
        return batch_data,batch_labels

def train_casia(i_db_name = 'casia',
                i_sub_db_name = 'all',
                i_kind_of_face = 'faces',
                i_f_factor = 1.0,
                i_dsize = (224,224,3),
                i_time_step = 5,
                i_batch_size = 5,
                i_z_score_norm_flag = True,
                i_num_aug_factor = 1,
                i_image_gap = 1,
                i_save_model_path=None):
    """
    :param i_db_name:
    :param i_sub_db_name:
    :param i_kind_of_face:
    :param i_f_factor:
    :param i_dsize:
    :param i_time_step:
    :param i_batch_size:
    :param i_z_score_norm_flag:
    :param i_num_aug_factor:
    :param i_save_model_path:
    :return:
    """
    print('Start TRAINING DNN Model Using CASIA dataset...............................................................')
    save_image_for_visualization = False #Set to True for saving face images.
    _model = dnn_models.vgg_19_lstm_v2(input_shape=i_dsize,
                                       timesteps=i_time_step,
                                       dropout_value=0.25,
                                       num_class=2,
                                       lstm_size=2048)
    print(_model.summary())
    keras.utils.plot_model(_model, to_file='{}/model.png'.format(i_save_model_path))
    # Train data
    train_data = casia_data_object(dataset_path='datasets/{}/'.format(i_db_name),
                                   mode='train',
                                   sub_db_mode=i_sub_db_name,
                                   save_face_extern=i_kind_of_face,
                                   face_extend_factor=i_f_factor,
                                   save_image_flag=save_image_for_visualization)
    train_data_gen = casia_data_generator(i_data=train_data.data,
                                          i_labels=train_data.labels,
                                          i_dsize=dsize,
                                          i_timestep=time_step,
                                          i_batch_size=batch_size,
                                          i_z_sscore=z_score_norm_flag,
                                          i_mean_image=None,
                                          i_std_image=None,
                                          i_image_gap=i_image_gap,
                                          i_num_aug=i_num_aug_factor)
    # Testing
    test_flag = True
    if test_flag:
        _test_data,_test_label = train_data_gen.__getitem__(0)
        for _index,_image in enumerate(_test_data):
            for _sub_index,_sub_image in enumerate(_image):
                cv2.imwrite('{}/image_{}_{}_{}.jpg'.format(i_save_model_path,_index,_sub_index,i_image_gap),_sub_image)
    # Test data
    test_data = casia_data_object(dataset_path='datasets/{}/'.format(i_db_name),
                                  mode='test',
                                  sub_db_mode=i_sub_db_name,
                                  save_face_extern=i_kind_of_face,
                                  face_extend_factor=i_f_factor,
                                  save_image_flag=save_image_for_visualization)
    test_data_gen = casia_data_generator(i_data=test_data.data,
                                         i_labels=test_data.labels,
                                         i_dsize=i_dsize,
                                         i_timestep=i_time_step,
                                         i_batch_size=i_batch_size,
                                         i_z_sscore=i_z_score_norm_flag,
                                         i_mean_image=train_data_gen.mean_image,
                                         i_std_image=train_data_gen.std_image,
                                         i_image_gap=i_image_gap,
                                         i_num_aug=0)
    if z_score_norm_flag:
        train_data_gen.save_mean_std_image(save_path=i_save_model_path)
        test_data_gen.save_mean_std_image(save_path=i_save_model_path)
    # Start training....
    lrate = keras.callbacks.LearningRateScheduler(dnn_models.lrate_schedule, verbose=1)
    save_model = dnn_models.save_model_during_train(i_save_model_path=i_save_model_path)
    _model.fit_generator(train_data_gen,
                         steps_per_epoch=train_data_gen.__len__(),
                         epochs=num_epoches,
                         verbose=1,
                         shuffle=False,  # Must be False because I manually perfom shuffling in data generator code
                         callbacks=[lrate,save_model])
    _model.save(filepath='{}/model-train.h5'.format(i_save_model_path))
    if z_score_norm_flag:
        np.save('{}/mean_image.npy'.format(i_save_model_path), train_data_gen.mean_image)
        np.save('{}/std_image.npy'.format(i_save_model_path), test_data_gen.std_image)
    _model.fit_generator(train_data_gen,
                         steps_per_epoch=train_data_gen.__len__(),
                         epochs=1,
                         verbose=1,
                         callbacks=[lrate],
                         shuffle=False,#Must be False because I manually perfom shuffling in data generator code
                         validation_data=test_data_gen,
                         validation_steps=test_data_gen.__len__())
    _model.save(filepath='{}/model.h5'.format(i_save_model_path))
    train_roc, train_error, train_th = performance_measurement_online(i_model=_model, i_data_generator=train_data_gen,
                                                                      i_save_result=i_save_model_path)
    with open('{}/train-roc.txt'.format(i_save_model_path), 'w') as file:
        for _roc in train_roc:
            file.write('{}\t{}\n'.format(_roc[0], _roc[1]))
    test_roc, test_error, test_th = performance_measurement_online(i_model=_model, i_data_generator=test_data_gen,
                                                                   i_train_th=train_th, i_save_result=i_save_model_path)
    with open('{}/test-roc.txt'.format(i_save_model_path), 'w') as file:
        for _roc in test_roc:
            file.write('{}\t{}\n'.format(_roc[0], _roc[1]))
    with open('{}/train_reults.txt'.format(i_save_model_path), 'w') as file:
        file.write('Train APCER = {}\t BPCER = {}\t HTER = {}\n'.format(train_error[0], train_error[1], train_error[2]))
        file.write('Low TH = {}\n'.format(train_th[0]))
        file.write('Hight TH = {}\n'.format(train_th[1]))
    with open('{}/test_reults.txt'.format(i_save_model_path), 'w') as file:
        file.write('Test APCER = {}\t BPCER = {}\t HTER = {}\n'.format(test_error[0], test_error[1], test_error[2]))
        file.write('Low TH = {}\n'.format(test_th[0]))
        file.write('Hight TH = {}\n'.format(test_th[1]))
    print('Train HTER = {} with threshold = {} ~ {}'.format(train_error[2], train_th[0], train_th[1]))
    print('Test  HTER = {} with threshold = {} ~ {}'.format(test_error[2], test_th[0], test_th[1]))
    return _model

"""
=> Function for peformance measurement.
"""
def performance_measurement_offline(i_db_name = 'casia',
                                    i_sub_db_name = 'all',
                                    i_kind_of_face = 'faces',
                                    i_f_factor = 1.0,
                                    i_dsize = (224,224,3),
                                    i_time_step = 5,
                                    i_batch_size = 5,
                                    i_z_score_norm_flag = True,
                                    i_num_aug_factor = 1,
                                    i_save_model_path = None):
    """
    :param i_db_name:
    :param i_sub_db_name:
    :param i_kind_of_face:
    :param i_f_factor:
    :param i_dsize:
    :param i_time_step:
    :param i_batch_size:
    :param i_z_score_norm_flag:
    :param i_num_aug_factor:
    :param i_save_model_path:
    :return:
    """
    print('Off-line Performance Measurement...........................................................................')
    if i_z_score_norm_flag:
        _loaded_mean_image = np.load('{}/mean_image.npy'.format(i_save_model_path))
        _loaded_std_image  = np.load('{}/std_image.npy'.format(i_save_model_path))
    else:
        _loaded_mean_image = 0
        _loaded_std_image = 1
    pretraind_model = load_model('{}/model.h5'.format(i_save_model_path))
    # Train data generator.
    # Train data
    train_data = casia_data_object(dataset_path='datasets/{}/'.format(i_db_name),
                                   mode='train',
                                   sub_db_mode=i_sub_db_name,
                                   save_face_extern=i_kind_of_face,
                                   face_extend_factor=i_f_factor,
                                   save_image_flag=False)
    train_data_gen = casia_data_generator(i_data=train_data.data,
                                          i_labels=train_data.labels,
                                          i_dsize=dsize,
                                          i_timestep=time_step,
                                          i_batch_size=batch_size,
                                          i_z_sscore=z_score_norm_flag,
                                          i_mean_image=None,
                                          i_std_image=None,
                                          i_num_aug=i_num_aug_factor)
    # Test data
    test_data = casia_data_object(dataset_path='datasets/{}/'.format(i_db_name),
                                  mode='test',
                                  sub_db_mode=i_sub_db_name,
                                  save_face_extern=i_kind_of_face,
                                  face_extend_factor=i_f_factor,
                                  save_image_flag=False)
    test_data_gen = casia_data_generator(i_data=test_data.data,
                                         i_labels=test_data.labels,
                                         i_dsize=i_dsize,
                                         i_timestep=i_time_step,
                                         i_batch_size=i_batch_size,
                                         i_z_sscore=i_z_score_norm_flag,
                                         i_mean_image=train_data_gen.mean_image,
                                         i_std_image=train_data_gen.std_image,
                                         i_num_aug=i_num_aug_factor)
    #Evaluate the performance.
    train_roc,train_error,train_th = performance_measurement_online(i_model=pretraind_model,
                                                                    i_data_generator=train_data_gen,
                                                                    i_save_result=i_save_model_path)
    with open('{}/train-roc.txt'.format(i_save_model_path),'w') as file:
        for _roc in train_roc:
            file.write('{}\t{}\n'.format(_roc[0],_roc[1]))
    with open('{}/train_reults.txt'.format(i_save_model_path),'w') as file:
        file.write('Train APCER = {}\t BPCER = {} \t HTER = {}\n'.format(train_error[0],train_error[1],train_error[2]))
        file.write('Low TH = {}\n'.format(train_th[0]))
        file.write('Hight TH = {}\n'.format(train_th[1]))
    test_roc,test_error, test_th   = performance_measurement_online(i_model=pretraind_model,
                                                                    i_data_generator=test_data_gen,
                                                                    i_train_th=train_th,
                                                                    i_save_result=i_save_model_path)
    with open('{}/test-roc.txt'.format(i_save_model_path),'w') as file:
        for _roc in test_roc:
            file.write('{}\t{}\n'.format(_roc[0],_roc[1]))
    with open('{}/test_reults.txt'.format(i_save_model_path), 'w') as file:
        file.write('Test APCER = {} \t BPCER = {} \t HTER = {}\n'.format(test_error[0],test_error[1],test_error[2]))
        file.write('Low TH = {}\n'.format(test_th[0]))
        file.write('Hight TH = {}\n'.format(test_th[1]))
    print('Train HTER = {} with threshold = {} ~ {}'.format(train_error[2], train_th[0],train_th[1]))
    print('Test  HTER = {} with threshold = {} ~ {}'.format(test_error[2], test_th[0], test_th[1]))
    return train_roc,train_th,test_roc,test_th

def performance_measurement_online(i_model=None,
                                   i_data_generator=None,
                                   i_train_th=(-10.,10.),
                                   i_save_result=None):
    print('Online Performance Measurement.............................................................................')
    _th_step = 0.001  #Should be fixed!
    _hter_min = 101.  #Fixed
    _apcer_min = 0.
    _bpcer_min = 0.
    _num_th = int(2./_th_step)+ 1
    train_flag = False
    print('Train TH = ',i_train_th)
    if i_train_th[0]<-1 and i_train_th[1]>1:
        train_flag=True
    if train_flag:
        _th_low = -1.
        _th_high = 1.
    else:
        #Evaluate the performance of test (val) data.
        _th_low = i_train_th[0]
        _th_high = i_train_th[1]

    _num_batches = i_data_generator.__len__()
    predict_scores = []
    ground_labels = []
    for _item in range(_num_batches):
        _batch,_label = i_data_generator.__getitem__(item=_item)
        _pred = i_model.predict_on_batch(_batch)
        predict_scores.append(_pred)
        ground_labels.append(_label)
    _scores = []
    _labels = []
    for _batch_index,_pred_batch in enumerate(predict_scores):
        for _image_index, _pred in enumerate(_pred_batch):
            _score = _pred[0]-_pred[1]
            _label = ground_labels[_batch_index][_image_index]
            _scores.append(_score)
            _labels.append(_label)
    #Save data for future usage.(optional)
    if train_flag:
        np.save('{}/train_scores.npy'.format(i_save_result),_scores)
        np.save('{}/train_labels.npy'.format(i_save_result),_labels)
    else:
        np.save('{}/test_scores.npy'.format(i_save_result), _scores)
        np.save('{}/test_labels.npy'.format(i_save_result), _labels)
    #Accumulate the ROC.
    _threshold = np.linspace(start=-1.,stop=1.,num=_num_th)
    _ROC   = []
    _accumulate = 0
    for _th in _threshold:
        _apcer_temp = 0
        _bpcer_temp = 0
        _C1_CORRECT = 0
        _C2_CORRECT = 0
        _num_class_2_sample=0
        _num_class_1_sample=0
        index = 0
        for _score in _scores:
            if _score>= _th:
                #Predicting as class-1 because scores = score_1 - score_2
                if _labels[index][0]-_labels[index][1] > 0:
                   #Ground-truth is class_1 => Correct (class_1 => class_1)
                    _C1_CORRECT +=1
                    _num_class_1_sample +=1
                else:
                   #Ground-truth is class_2 => Error (class_2 => class_1). If class_1 is live, class_2 is fake => APCER
                   _apcer_temp +=1
                   _num_class_2_sample+=1
            else:
                #Predicting as class-2 because scores = score_1 - score_2
                if _labels[index][0]-_labels[index][1] > 0:
                   #Ground-truth is class_1 => Error (class_1 => class_2. If class_1 is live, class_2 is fake => BPCER
                    _bpcer_temp += 1
                    _num_class_1_sample+=1
                else:
                   #Ground-truth is class_2 => Correct (class_2 => class_2)
                    _C2_CORRECT += 1
                    _num_class_2_sample += 1
            index +=1
        _apcer_temp/=_num_class_2_sample
        _bpcer_temp/=_num_class_1_sample
        _apcer_temp *=100
        _bpcer_temp *=100
        _bpar_temp   = 100. - _bpcer_temp
        _ROC.append([_apcer_temp,_bpar_temp])
        if (_th >=_th_low) and (_th<=_th_high):
            _hter = (_apcer_temp+_bpcer_temp)/2.
            if _hter < _hter_min:
                _hter_min = _hter
                _apcer_min = _apcer_temp
                _bpcer_min = _bpcer_temp
                _th_low = _th
                _accumulate = 0
            elif _hter==_hter_min:
                _accumulate +=1
            else:
                pass
        else:
            pass
    _th_high = _th_low + _accumulate*_th_step
    return _ROC,(_apcer_min, _bpcer_min,_hter_min),(_th_low,_th_high)

if __name__=="__main__":
    dataset_name = 'casia'
    sub_db_name = 'all'
    kind_of_face = 'faces' #Only 'faces' or 'mixed_faces' are accepted.
    f_factor  = 1.0
    time_step = 5
    batch_size = 5
    z_score_norm_flag = True
    num_aug_factor = 0 #Smaller or equal to zero means DONOT perform data augmentation.
    dsize = (224,224,3)
    num_epoches = 3
    image_gap = 10
    #Start training CNN-RNN model using CASIA dataset.
    save_model_path = 'models/{}/'.format(dataset_name)
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    save_model_path = '{}/{}_{}_tstep_{}_ffactor_{}_aug_{}/'.format(save_model_path,kind_of_face,sub_db_name,time_step,f_factor,num_aug_factor)
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    casia_model = train_casia(i_db_name=dataset_name,
                              i_sub_db_name=sub_db_name,
                              i_kind_of_face=kind_of_face,
                              i_f_factor=f_factor,
                              i_dsize=dsize,
                              i_time_step=time_step,
                              i_batch_size=batch_size,
                              i_z_score_norm_flag=z_score_norm_flag,
                              i_num_aug_factor=num_aug_factor,
                              i_image_gap=image_gap,
                              i_save_model_path=save_model_path)
    print('End Of Training Procedures.................................................................................')
