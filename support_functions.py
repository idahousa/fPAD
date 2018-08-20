import os
import numpy as np
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
from scipy import misc
from PIL import  Image as pil_image_utils
from PIL import  ImageFilter as PIL_ImageFilter
import tensorflow as tf
import keras.backend as K
#Description: create a directory nested in  current working directory.
def create_sub_dir(dir_name):
    cwd = os.getcwd()
    sub_dir_path = os.path.join(cwd,dir_name)
    if not os.path.exists(sub_dir_path):
        os.mkdir(path=sub_dir_path)
    return sub_dir_path

def convert_list2ndarray(input_list):
    return np.asanyarray(input_list)

def convert_tuple2ndarray(input_tuple):
    return np.asanyarray(input_tuple)

def list_files_in_directory(directory_path):
    return os.listdir(directory_path)

def get_file_extension(file_name_with_extension):
    file_len = len(file_name_with_extension)
    index = 0
    for i in range(1,file_len):
        if file_name_with_extension[-i]=='.':
            index = -i+1
            break
    if index ==0:
        return 'error'
    else:
        return file_name_with_extension[index:file_len]

def get_file_name(file_name_with_extension):
    file_len = len(file_name_with_extension)
    file_ext = get_file_extension(file_name_with_extension=file_name_with_extension)
    file_ext_len = len(file_ext)
    if file_len>file_ext_len:
        return file_name_with_extension[0:file_len-file_ext_len-1] # 1 is because of dot character
    else:
        return 'error'

def list_image_files(directory_path,shuffle = True):
    rtn_list = []
    file_extern_list = ['bmp', 'BMP', 'jpg', 'JPG', 'jpeg', 'JPEG', 'tiff', 'TIFF', 'PNG', 'png', 'gif', 'GIF']
    all_file_list = list_files_in_directory(directory_path)
    for file_name in all_file_list:
        file_extern = get_file_extension(file_name)
        if file_extern in file_extern_list:
            rtn_list.append(file_name)
    num_image = len(rtn_list)
    if shuffle:
        for index in range(num_image):
            rand_index_1 = np.random.randint(low=0,high=num_image)
            rand_index_2 = np.random.randint(low=0,high=num_image)
            temp_val = rtn_list[rand_index_1]
            rtn_list[rand_index_1] = rtn_list[rand_index_2]
            rtn_list[rand_index_2] =temp_val
    return rtn_list, num_image
"""
=>Function to read image in a dataset and store in RAM memory...........................................................
=>Use for small dataset whose size is fit to RAM memory.................................................................
path_to_dataset: a path to directory where we store images. Ex. C://DATASETS//train//
class_marker: a list of marker which is associated in image file name to recognize class of image. Ex. 'Live', 'Fake'...
mode = 'train' or 'test' or 'val'.......................................................................................
"""
def lrate_schedule(epoch, c_lrate):
    drop_period = 3
    drop_factor = 0.1
    if epoch==0:
        return c_lrate
    elif epoch%drop_period==0:
        return c_lrate*drop_factor
    else:
        return c_lrate

def read_image_data(path_to_dataset, class_marker = ('Fake','Live'), max_image = 60000, mode = 'train'):
    if mode == 'train':
        file_list = os.path.join(path_to_dataset,'train.txt')
    else:
        file_list = os.path.join(path_to_dataset,'test.txt')
    #Check the list file. If it is not existed, then create it.........................................................
    if not os.path.exists(file_list):
        image_list, num_image = list_image_files(directory_path=os.path.join(path_to_dataset,''))
        if num_image>max_image:
            return 0,0,False
        with open(file_list,'w') as file:
            for index in range(num_image):
                image_name = image_list[index]
                class_label = 0
                for marker in class_marker:
                    if marker in image_name:
                        file.write("{}\t{}\n".format(image_name,class_label))
                        break
                    else:
                        class_label +=1
    else:
        num_image = 0
        with open(file_list,'r') as file:
            for _ in file:
                num_image +=1
            if num_image>max_image:
                return 0,0,False
    print('Loading images.............................................................................................')
    data = []
    labels = []
    with open(file_list,'r') as file:
        for line in file:
            image_name = line[0:line.find('\t')]
            image_label = line[line.find('\t')+1:len(line)]
            image_path = os.path.join(path_to_dataset,image_name)
            image_data = image.load_img(image_path, target_size=(224,224))
            image_data = image.img_to_array(image_data)
            data.append(image_data)
            labels.append(image_label)
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)
    print('Finished loading images....................................................................................')
    return data,labels,True

def make_image_data_generator(path_to_dataset,class_marker = ('Fake','Live'), batch_size = 64, mode='train'):
    if mode == 'train':
        file_list = os.path.join(path_to_dataset,'train.txt')
    else:
        file_list = os.path.join(path_to_dataset,'test.txt')
    #Check the list file. If it is not existed, then create it.........................................................
    if not os.path.exists(file_list):
        image_list, num_image = list_image_files(directory_path=os.path.join(path_to_dataset,''))
        with open(file_list,'w') as file:
            for index in range(num_image):
                image_name = image_list[index]
                class_label = 0
                for marker in class_marker:
                    if marker in image_name:
                        file.write("{}\t{}\n".format(image_name,class_label))
                        break
                    else:
                        class_label +=1
    else:
        num_image = 0
        with open(file_list,'r') as file:
            for _ in file:
                num_image +=1
    print('Loading images.............................................................................................')
    image_index = 0
    _data = []
    _labels = []
    with open(file_list,'r') as file:
        for line in file:
            image_name = line[0:line.find('\t')]
            image_label = line[line.find('\t') + 1:len(line)]
            image_path = os.path.join(path_to_dataset, image_name)
            image_data = image.load_img(image_path, target_size=(224, 224))
            image_data = image.img_to_array(image_data)
            _data.append(image_data)
            _labels.append(image_label)
            image_index += 1
            if image_index%batch_size==0 or image_index==num_image:
                out_data   = np.array(_data)
                out_labels = keras.utils.to_categorical(np.array(_labels),num_classes=len(class_marker))
                _data = []
                _labels = []
                yield (out_data, out_labels)

    return True
#----------------------------------------------------------------------------------------------------------------------#
def get_list_file(mode, path_to_dataset,class_marker):
    if mode == 'train':
        file_list = os.path.join(path_to_dataset, 'train.txt')
    else:
        file_list = os.path.join(path_to_dataset, 'test.txt')
    if not os.path.exists(file_list):
        image_list, num_image = list_image_files(directory_path=os.path.join(path_to_dataset, ''))
        with open(file_list, 'w') as file:
            for index in range(num_image):
                image_name = image_list[index]
                class_label = 0
                for marker in class_marker:
                    if marker in image_name:
                        file.write("{}\t{}\n".format(image_name, class_label))
                        break
                    else:
                        class_label += 1
    list_of_file_names = []
    list_of_labels = []
    with open(file_list, 'r') as file:
        for line in file:
            _image_name = line[0:line.find('\t')]
            _image_label = line[line.find('\t') + 1:len(line)]
            list_of_file_names.append(_image_name)
            list_of_labels.append([int(_image_label)])
    return list_of_file_names, list_of_labels

def cmp_list(list_1,list_2):
    for index, val in enumerate(list_1):
        if val==list_2[index]:
            return_val = True
        else:
            return_val = False
    return return_val

def get_mean_std_image(path_to_dataset,input_shape=(224,224,3),mode = 'train',retinex_filtering_flag=False):
    #Measuring the mean and std image of training dataset
    #STD = sqrt(E[X^2] - (E[X])^2)
    print('Measuring the mean and std image of training dataset........................................................')
    sigma = 1e-10  #a small value is added to std to zero-preventing case.
    if mode=='train':
        mean_image = np.zeros(shape=input_shape,dtype=np.float32)
        std_image = np.zeros(shape=input_shape,dtype=np.float32)
        image_list, num_image = list_image_files(directory_path=path_to_dataset)
        for index in range(num_image):
            _image_name = image_list[index]
            _image_path = os.path.join(path_to_dataset, _image_name)
            _image_data = image.img_to_array(image.load_img(path=_image_path, target_size=input_shape))
            if retinex_filtering_flag:
                _image_data = retinex_filtering(_image_data)
            #Add two image element-wise
            mean_image = np.add(mean_image,_image_data) #E[X]
            std_image  = np.add(std_image,np.power(_image_data,2)) #E[X^2]
        mean_image = np.divide(mean_image,num_image) #Final E[X]
        std_image  = np.divide(std_image,num_image)  #Final E[X^2]
        std_image  = np.add(np.sqrt(np.subtract(std_image,np.power(mean_image,2))),sigma)
        """
        for index in range(num_image):
            _image_name = image_list[index]
            _image_path = os.path.join(path_to_dataset, _image_name)
            _image_data = image.img_to_array(image.load_img(path=_image_path, target_size=input_shape))
            std_image = np.add(std_image,np.power(np.subtract(_image_data,mean_image),2))
        std_image = np.sqrt(np.divide(std_image,num_image))
        """
        print('End of measuring the mean and std image of training dataset..............................................')
        return mean_image, std_image
    else:
        return False

def get_statisticals_dataset(path_to_dataset,input_shape=(224,224,3),mode = 'train',retinex_filtering_flag=False,class_marker = ('Fake','Live')):
    #Measuring the mean and std image of training dataset
    #STD = sqrt(E[X^2] - (E[X])^2)
    print('Measuring the mean and std image of training dataset........................................................')
    sigma = 1e-10  #a small value is added to std to zero-preventing case.
    if mode=='train':
        mean_image = np.zeros(shape=input_shape,dtype=np.float32)
        std_image  = np.zeros(shape=input_shape,dtype=np.float32)
        mean_live  = np.zeros(shape=input_shape,dtype=np.float32)
        mean_fake  = np.zeros(shape=input_shape,dtype=np.float32)
        image_list, label_list = get_list_file(mode, path_to_dataset, class_marker)
        num_live_image = 0
        num_fake_image = 0
        num_image = len(image_list)
        for index in range(num_image):
            _image_name = image_list[index]
            _image_path = os.path.join(path_to_dataset, _image_name)
            _image_data = image.img_to_array(image.load_img(path=_image_path, target_size=input_shape))
            if retinex_filtering_flag:
                _image_data = retinex_filtering(_image_data)
            #Add two image element-wise
            mean_image = np.add(mean_image,_image_data) #E[X]
            std_image  = np.add(std_image,np.power(_image_data,2)) #E[X^2]
            if cmp_list(label_list[index],[0]):#Fake image
                num_fake_image+=1
                mean_fake = np.add(mean_fake,_image_data)
            else:
                num_live_image+=1
                mean_live = np.add(mean_live,_image_data)
        mean_image = np.divide(mean_image,num_image) #Final E[X]
        mean_live  = np.divide(mean_live,num_live_image)
        mean_fake  = np.divide(mean_fake,num_fake_image)
        std_image  = np.divide(std_image,num_image)  #Final E[X^2]
        std_image  = np.add(np.sqrt(np.subtract(std_image,np.power(mean_image,2))),sigma)
        print('End of measuring the mean and std image of training dataset..............................................')
        return mean_image, std_image,mean_live,mean_fake
    else:
        return False

def generate_image_patches(input_array,num_patch_width=2,num_path_height=2,over_lapped_ratio=0.250,full_patch=True):
    #input_array is an gray or color RBG image => shape [height, width, nchannel]
    #Output is the patches extracted from input image => shape [num_patch,patch_height,patch_width,nchannel]
    #num_path is the number of pathes => num_patch = num_path_width*num_path_width + 1 (global image)
    input_shape = input_array.shape
    if len(input_shape)==2 or len(input_shape)==3:
        input_height = input_shape[0]
        input_width = input_shape[1]
    else:
        return False
    path_width  = np.floor(input_width/num_patch_width)
    path_height = np.floor(input_height/num_path_height)
    margin_x = np.floor(path_width*over_lapped_ratio)
    margin_y = np.floor(path_height*over_lapped_ratio)
    patches  = []
    num_patches = 0
    for x in range(num_patch_width):
        begin_x = np.int32(x*path_width - margin_x)
        end_x   = np.int32((x+1)*path_width + margin_x)
        if begin_x<0:
            begin_x = 0
        if end_x > input_width:
            end_x = input_width
        for y in range(num_path_height):
            begin_y = np.int32(y*path_height - margin_y)
            end_y   = np.int32((y+1)*path_height + margin_y)
            if begin_y <0:
                begin_y =0
            if end_y > input_height:
                end_y = input_height
            #Take the batch..............
            #print("{} : begin_x = {}, begin_y = {}, end_x ={}, end_y ={}".format(num_patches,begin_x,begin_y,end_x,end_y))
            patch = input_array[begin_y:end_y,begin_x:end_x,:]
            #Resize and append..........................................................................................
            patch = misc.imresize(patch,size=(input_height,input_width))
            patches.append(patch)
            num_patches +=1
    if full_patch:
        patches.append(input_array)
    patches = np.array(patches)
    return patches,num_patches + 1

def ndarray_to_image(input_array):
    #Scale input ndarray to range of [0,255] for image conversion
    _ndim = len(input_array.shape)
    _image = np.zeros(shape=input_array.shape,dtype=np.uint8)
    if _ndim == 2:
        return  ndarray_to_image_2d(input_array)
    elif _ndim==3:
        _image[:, :, 0] = ndarray_to_image_2d(input_array[:, :, 0])
        _image[:, :, 1] = ndarray_to_image_2d(input_array[:, :, 1])
        _image[:, :, 2] = ndarray_to_image_2d(input_array[:, :, 2])
        return _image
    assert (_ndim !=2 or _ndim!=3), "Error input array"

def ndarray_to_image_2d(input_array):
    _image = np.zeros(shape=input_array.shape,dtype=np.uint8)
    _ndim = len(input_array.shape)
    if _ndim==2:
        _min_val = np.amin(input_array)
        _max_val = np.amax(input_array)
        _gap = np.subtract(_max_val,_min_val)+1e-10
        _image = np.multiply(np.divide(np.subtract(input_array, _min_val), _gap), 255)
        """
        _rows = input_array.shape[0]
        _cols = input_array.shape[1]
        for row in range(_rows):
            for col in range(_cols):
                _pixel = input_array[row,col]
                _image[row, col] = np.multiply(np.divide(np.subtract(_pixel,_min_val),_gap),255)
        """
        return _image.astype(np.uint8)
    assert _ndim!=2, "Error input array"

class ImageDataSequence(keras.utils.Sequence):
    def __init__(self,path_to_dataset,mode,class_marker,batch_size,norm_flag,full_patch,mean_image,std_image,num_path_width,num_path_height, overappled_patch_ratio,retinex_filtering_flag):
        self.mode = mode
        self.ext_full_patch = full_patch
        self.retinex_filtering_flag = retinex_filtering_flag
        self.norm_flag = norm_flag
        self.path_to_dataset = path_to_dataset
        self.image_shape = (224,224,3)
        self.class_marker = class_marker
        self.batch_size = batch_size
        self.num_patch_width = num_path_width
        self.num_patch_height = num_path_height
        self.overappled_patch_ratio = overappled_patch_ratio
        self.list_of_file_names, self.list_of_labels = get_list_file(self.mode, self.path_to_dataset, self.class_marker)
        if self.mode == 'train':
            if self.norm_flag:
                self.mean_image,self.std_image = get_mean_std_image(self.path_to_dataset,input_shape=self.image_shape,mode=self.mode,retinex_filtering_flag = self.retinex_filtering_flag)
                plt.imsave('models/mean_image.jpg',np.uint8(self.mean_image))
                plt.imsave('models/std_image.jpg', ndarray_to_image(self.std_image))
            else:
                self.mean_image = np.zeros(shape=self.image_shape,dtype=np.float32)
                self.std_image = np.ones(shape=self.image_shape,dtype=np.float32)
        else:
            if self.norm_flag:
                self.mean_image = mean_image
                self.std_image  = std_image
                plt.imsave('models/mean_image_val.jpg', np.uint8(self.mean_image))
                plt.imsave('models/std_image_val.jpg', ndarray_to_image(self.std_image))
            else:
                self.mean_image = np.zeros(shape=self.image_shape,dtype=np.float32)
                self.std_image = np.ones(shape=self.image_shape,dtype=np.float32)
    def __len__(self):#Return the number of batches in a sequence
        num_image = len(self.list_of_file_names)
        num_batches = np.int32(np.ceil(num_image/self.batch_size))
        return num_batches
    def __getitem__(self, item):#Get the data and label of each batchs
        batch_x = self.list_of_file_names[item*self.batch_size:(item+1)*self.batch_size]
        batch_y = self.list_of_labels[item*self.batch_size:(item+1)*self.batch_size]
        batch_data = []
        batch_labels = batch_y
        #print('Batch-size = {}'.format(len(batch_x)))
        for image_name in batch_x:
            _image_path = os.path.join(self.path_to_dataset, image_name)
            _image_data = image.load_img(_image_path, target_size=self.image_shape)
            if self.retinex_filtering_flag:
                _image_data = retinex_filtering(image.img_to_array(_image_data))
            else:
                _image_data = image.img_to_array(_image_data)
            if self.norm_flag:
                _image_data = np.divide(np.subtract(_image_data,self.mean_image),self.std_image)
            _image_data,num_patches = generate_image_patches(_image_data,
                                                             num_patch_width=self.num_patch_width,
                                                             num_path_height=self.num_patch_height,
                                                             over_lapped_ratio=self.overappled_patch_ratio,
                                                             full_patch=self.ext_full_patch)
            batch_data.append(_image_data)
        batch_data = np.array(batch_data)
        batch_labels = np.array(keras.utils.to_categorical(batch_labels,num_classes=len(self.class_marker)))
        return batch_data,batch_labels
#----------------------------------------------------------------------------------------------------------------------#
class draw_loss_curve(keras.callbacks.Callback):
    def __init__(self):
        self.i = 0
        self.x = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
    def on_train_begin(self, logs=None):
        self.i = 0
        self.x =[0]
        self.train_loss = []
        self.val_loss = []
        self.train_acc = [0]
        self.val_acc = [0]
    def on_epoch_end(self, epoch, logs=None):
        self.i += 1
        if self.i == 1:
            self.train_loss.append(logs.get('loss'))
            self.val_loss.append(logs.get('val_loss'))
        self.x.append(self.i)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        optimizer = self.model.optimizer
        current_lr = keras.backend.eval(optimizer.lr)
        print('Current learning rate = {:10f}...'.format(current_lr))
        print(self.params)
    def on_train_end(self, logs=None):
        self.x.append(self.i)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.grid(color = 'k',linestyle = 'dashdot', linewidth = 0.25)
        ax1.plot(self.x, self.train_loss, label='loss')
        ax1.plot(self.x, self.val_loss, label='val_loss')
        ax1.legend(['loss', 'val_loss'])
        #ax1.legend(['loss','val_loss'],loc='upper center')
        ax2.grid(color='k', linestyle='dashdot', linewidth=0.25)
        ax2.plot(self.x, self.train_acc, label='acc')
        ax2.plot(self.x, self.val_acc, label='val_acc')
        ax2.legend(['acc', 'val_acc'])
        #ax2.legend(['acc','val_acc'],loc='upper center')
        plt.show()

"""******************************************************************************************************************"""
def PIL_Image_To_ndarray(pil_image):
    return np.array(pil_image)

def ndarray_To_PIL_Image(nd_array):
    _input_shape = nd_array.shape
    if len(_input_shape)==3:
        if _input_shape[2]==1 or _input_shape[2]==3:
            return pil_image_utils.fromarray(nd_array)
        else:
            raise Exception('Input array must be in gray or colour RBG mage')
    elif len(_input_shape)==2:
        return pil_image_utils.fromarray(nd_array)
    else:
        raise Exception('Input array must be in gray or colour RBG mage')

def retinex_filtering(input_image,sigma=10.,adjust=3.0):
    _input_shape = input_image.shape
    if len(_input_shape)==2:
        _retinex = retinex_filtering_gray(input_image,sigma=sigma,adjust=adjust)
        return _retinex
    elif len(_input_shape)==3:
        #print('Retinex filtering of RGB COLOR image...................................................................')
        #print('Color image shape is {}'.format(_input_shape))
        _height = _input_shape[0]
        _width  = _input_shape[1]
        _retinex = np.ndarray(shape=_input_shape)
        _retinex[:, :, 0] = retinex_filtering_gray(input_image[:,:,0].reshape(_height,_width), sigma=sigma, adjust=adjust)
        _retinex[:, :, 1] = retinex_filtering_gray(input_image[:,:,1].reshape(_height,_width), sigma=sigma, adjust=adjust)
        _retinex[:, :, 2] = retinex_filtering_gray(input_image[:,:,2].reshape(_height,_width), sigma=sigma, adjust=adjust)
        return _retinex
    else:
        return False

def retinex_filtering_gray(input_image,sigma=10.,adjust=3.0):
    _input_shape = input_image.shape
    if len(_input_shape) == 2:
        _PIL_image = ndarray_To_PIL_Image(input_image)
        _gaussian_blur = _PIL_image.filter(PIL_ImageFilter.GaussianBlur(radius=sigma))
        _gaussian_blur = PIL_Image_To_ndarray(_gaussian_blur)
        _difference_img = np.log(input_image + 1.) - np.log(_gaussian_blur + 1.)
        _mean_val = np.mean(_difference_img)
        _std_val = np.std(_difference_img)
        _min_val = _mean_val - adjust * _std_val
        _max_val = _mean_val + adjust * _std_val
        _mul_factor = 255.0 / (_max_val - _min_val)
        _retinex_image = np.uint8(_mul_factor * (_difference_img - _min_val))
        _retinex_image[_retinex_image < 0] = 0
        _retinex_image[_retinex_image > 255] = 255
        return np.uint8(np.floor(_retinex_image+0.5))
    else:
        print('Error datatype..........................................................................................')
        return False

def contrastive_loss(y_true,y_pred):
    #ref: https://www.kaggle.com/c/quora-question-pairs/discussion/33631
    #     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    margin=1
    return K.mean(y_true*K.square(y_pred)+(1-y_true)*K.square(K.maximum(margin-y_pred,0)))

def triplet_loss_feature(anchor_feature,possitive_feature,negative_feature,margin = 1.):
    d_pos = tf.reduce_sum(tf.square(anchor_feature - possitive_feature),axis=1)
    d_neg = tf.reduce_sum(tf.square(anchor_feature - negative_feature),axis=1)
    loss =  tf.reduce_mean(tf.maximum(0.,margin + d_pos - d_neg))
    return loss

def triplet_loss(y_true,y_pred):
    return True
