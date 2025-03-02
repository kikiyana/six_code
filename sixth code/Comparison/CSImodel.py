"""
The Codes in this file are used to classify Human Activity using Channel State Information.
The deep learning architecture used here is BidirectioCSImodel.pyCnal LSTM stacked with One Attention Layer.
Author: https://github.com/ludlows
2019-12

Paper "A Survey on Behaviour Recognition Using WiFi Channel State Information"
目前没导入自己的数据集，还不能运行。
"""

import numpy as np
import tensorflow as tf
import glob
import os
import csv
from scipy.io import loadmat, savemat

def merge_csi_DataAndLabel(path):

    # 文件地址类型：/bedroom/1,2,...
    listDir = []  # 存储子文件夹列表
    csiData = [] # 存储CSI数据
    activity = []  # 存储标签
    labelList = ['lying', 'sitting', 'running', 'walking', 'pick up', 'wave hand', 'jump', 'squat', 'site down', 'stand up', 'fall down'] # 标签列表

    # 获取数据集子文件夹列表，并转为整型进行排序(文件名为1，2，3...)
    for root, dirs, files in os.walk(path, topdown=False):
        listDir = dirs
    listDir = sorted(list(map(int, listDir)))

    # 获取原始CSI数据
    for i in range(len(listDir)):
        subpath = path + str(listDir[i])#构建子路径，即path+子文件夹名
        whole_file = [os.path.join(subpath, file) for file in os.listdir(subpath)]
        # 对于每一个路径，将其打开之后，使用readlines获取全部内容
        for w in whole_file:
            # 使用loadmat(路径)方法读取mat文件中的csi数据，获取csi数据
            data = loadmat(w)['csi']
            csiData.append(data)
    # 将csiData转为numpy数组，通过dtype=complex参数指定数据类型为复数类型
    csiData = np.array(csiData, dtype=complex)

    # 构造标签
    labelNum = []
    for i in range(len(listDir)):
        if i != len(listDir)-1:
            # 使用extend扩展数组长度
            activity.extend([labelList[i]] * 100)
            labelNum.extend([i] * 100)
        else:
            # 如果是最后一个文件夹，则重复50次(因为数据集中只有50个样本)
            activity.extend([labelList[i]] * 50)
            labelNum.extend([i] * 50)
    labelNum = np.array(labelNum)

    return csiData, activity, labelNum

def merge_csi_label(csifile, labelfile, win_len=1000, thrshd=0.6, step=200):
    """
    Merge CSV files into a Numpy Array  X,  csi amplitude feature
    Returns Numpy Array X, Shape(Num, Win_Len, 90)，
    合并CSV文件到一个Numpy数组X，csi振幅特征；
    返回Numpy数组X，形状（Num，Win_Len，90），
    Args:
        csifile  :  str, csv file containing CSI data，csv文件，包含CSI数据
        labelfile:  str, csv file with activity label，带有活动标签的CSI文件
        win_len  :  integer, window length，窗口长度
        thrshd   :  float,  determine if an activity is strong enough inside a window，活动存在阈值
        step     :  integer, sliding window by step
    """
    activity = []
    with open(labelfile, 'r') as labelf:
        # 逐行读取标签文件信息，并将其存储到reader中
        reader = csv.reader(labelf)
        # 逐行读取刚刚存储在reader中的CSV信息
        for line in reader:
            # 把当前行的第一个元素赋值给label，假设CSV文件第一列为标签数据
            label = line[0]
            if label == 'NoActivity':
                activity.append(0)
            else:
                activity.append(1)
    # activity变为数组，长度为csv数据段的数量
    activity = np.array(activity)
    csi = []
    with open(csifile, 'r') as csif:
        # 逐行读取CSI文件信息，并将其存储到reader中
        reader = csv.reader(csif)
        for line in reader:
            # 将第一行的每个元素转换为浮点数
            line_array = np.array([float(v) for v in line])
            # 提取振幅信息第一行的1-91号元素，共90个元素
            line_array = line_array[1:91]
            # np.newaxis 用于在数组的维度上增加一个新的维度， ... 表示保持原有的维度。将一维数组转换为二维数组
            csi.append(line_array[np.newaxis, ...])
    # 将数组沿着垂直方向(axis=0)进行拼接
    csi = np.concatenate(csi, axis=0)
    # 确保csi的第一个维度(行数)是否和activity的第一个维度长度相等
    assert (csi.shape[0] == activity.shape[0])
    #窗口滑动来来提取特征， screen the data with a window
    index = 0
    feature = []
    while index + win_len <= csi.shape[0]:
        # 从index开始提取一个win_len长的子数组
        cur_activity = activity[index:index + win_len]
        # 如果子数组中活动的数量小于阈值thrshd，说明该活动不存在，跳过该子数组
        if np.sum(cur_activity) < thrshd * win_len:
            index += step
            continue
        # 否则，将子数组的振幅信息赋值给cur_feature，1*win_len*90
        cur_feature = np.zeros((1, win_len, 90))
        #  将csi数组中从 index 到 index + win_len-1 行的数据赋值给 cur_feature 数组的第一个维度。
        cur_feature[0] = csi[index:index + win_len, :]
        # 符合条件的特征存储到feature中
        feature.append(cur_feature)
        index += step
    # 将feature中的特征数组沿着垂直方向（axis=0）进行拼接
    return np.concatenate(feature, axis=0)


def extract_csi_by_label(raw_folder, label, labels, save=False, win_len=1000, thrshd=0.6, step=200):
    """
    Returns all the samples (X,y) of "label" in the entire dataset
    Args:
        raw_foler: The path of Dataset folder
        label    : str, could be one of labels
        labels   : list of str, ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
        save     : boolean, choose whether save the numpy array
        win_len  :  integer, window length
        thrshd   :  float,  determine if an activity is strong enough inside a window
        step     :  integer, sliding window by step
    """
    print('Starting Extract CSI for Label {}'.format(label))
    # 将所有字符转为小写字符
    label = label.lower()
    # 检查label是否在labels中
    if not label in labels:
        raise ValueError(
            "The label {} should be among 'bed','fall','pickup','run','sitdown','standup','walk'".format(labels))
    # 拼接文件名模式
    data_path_pattern = os.path.join(raw_folder, 'input_*' + label + '*.csv')
    # 根据构造的模式去获取匹配列表并对列表进行排序
    input_csv_files = sorted(glob.glob(data_path_pattern))
    # 将基本名称input替换为annotation
    annot_csv_files = [os.path.basename(fname).replace('input_', 'annotation_') for fname in input_csv_files]
    # 拼接完整路径
    annot_csv_files = [os.path.join(raw_folder, fname) for fname in annot_csv_files]
    feature = []
    index = 0
    for csi_file, label_file in zip(input_csv_files, annot_csv_files):
        index += 1
        if not os.path.exists(label_file):
            # 将format中的值插入占位符中
            print('Warning! Label File {} doesn\'t exist.'.format(label_file))
            continue
        feature.append(merge_csi_label(csi_file, label_file, win_len=win_len, thrshd=thrshd, step=step))
        # 打印进度
        print('Finished {:.2f}% for Label {}'.format(index / len(input_csv_files) * 100, label))

    feat_arr = np.concatenate(feature, axis=0)
    if save:
        # 生成了一个带有标签、窗口长度、阈值百分比和步长的文件名的压缩文件
        np.savez_compressed("X_{}_win_{}_thrshd_{}percent_step_{}.npz".format(
            label, win_len, int(thrshd * 100), step), feat_arr)
    # 创建独热编码，one hot，独热编码将分类变量转为机器学习可以处理的形式，例如将['bed']转换为[0,1,0,0,0,0,0,0,0,0,0]
    feat_label = np.zeros((feat_arr.shape[0], len(labels)))
    feat_label[:, labels.index(label)] = 1
    return feat_arr, feat_label


def train_valid_split(numpy_tuple, train_portion=0.9, seed=379):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    # 设置随机种子固定随机值，好复现
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    # 数据集划分和标签处理
    for i, x_arr in enumerate(numpy_tuple):
        # 随机打乱数组的索引
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        # 划分训练集和验证集，总长*train_portion为训练集的长度
        split_len = int(train_portion * x_arr.shape[0])
# 创建x_train存储数据
        # 将随机化后的前 split_len 个样本添加到训练集 x_train 中
        x_train.append(x_arr[index[:split_len], ...])
# 创建y_train存储对应标签
        # 创建一个大小为 (split_len, 11) 的零矩阵 tmpy ，
        tmpy = np.zeros((split_len, 11))
        # 然后将第 i 列设置为1，表示对应的标签。
        tmpy[:, i] = 1
        # 将 tmpy 添加到训练集的标签列表 y_train 中。
        y_train.append(tmpy)
# 创建x_valid存储数据
        # 将剩余的样本添加到验证集 x_valid 中。
        x_valid.append(x_arr[index[split_len:], ...])
# 创建y_valid存储对应标签
        tmpy = np.zeros((x_arr.shape[0] - split_len, 11))
        tmpy[:, i] = 1
        y_valid.append(tmpy)
    # 将数组按行连接起来,大小
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    # 对训练集的数据和标签进行随机化处理
    index = np.random.permutation([i for i in range(x_train.shape[0])])
    x_train = x_train[index, ...]
    y_train = y_train[index, ...]
    return x_train, y_train, x_valid, y_valid


def extract_csi(raw_folder, labels, save=False, win_len=1000, thrshd=0.6, step=200):
    """
    Return List of Array in the format of [X_label1, y_label1, X_label2, y_label2, .... X_Label7, y_label7]
    Args:
        raw_folder: the folder path of raw CSI csv files, input_* annotation_*
        labels    : all the labels existing in the folder
        save      : boolean, choose whether save the numpy array
        win_len   :  integer, window length
        thrshd    :  float,  determine if an activity is strong enough inside a window
        step      :  integer, sliding window by step
    """
    ans = []
    # 生成一个特征+标签的数组
    for label in labels:
        feature_arr, label_arr = extract_csi_by_label(raw_folder, label, labels, save, win_len, thrshd, step)
        ans.append(feature_arr)
        ans.append(label_arr)
    # 将列表转换为元数组
    return tuple(ans)


class AttenLayer(tf.keras.layers.Layer):
    """
    Attention Layers used to Compute Weighted Features along Time axis
    Args:
        num_state :  number of hidden Attention state

    2019-12, https://github.com/ludlows
    """

    def __init__(self, num_state, **kw):
        super(AttenLayer, self).__init__(**kw)
        self.num_state = num_state

    def build(self, input_shape):
        # 计算模型的输出，添加'kernel'形状的权重，input_shape[-1] 表示输入张量的最后一个维度的大小， self.num_state 表示状态的数量
        self.kernel = self.add_weight('kernel', shape=[input_shape[-1], self.num_state])
        # 对模型的输出进行偏置，形状是 [self.num_state]
        self.bias = self.add_weight('bias', shape=[self.num_state])
        # 计算模型输出的概率，形状是 [self.num_state]
        self.prob_kernel = self.add_weight('prob_kernel', shape=[self.num_state])

    # 定义前向传播过程
    def call(self, input_tensor):
        # 将输入张量进行线性变换并进行非线性处理，以得到一个新的特征表示
        atten_state = tf.tanh(tf.tensordot(input_tensor, self.kernel, axes=1) + self.bias)
        # 将特征表示映射到输出空间，并得到各个输出类别的原始分数。
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
        # 计算各个类别的概率
        prob = tf.nn.softmax(logits)
        # 计算加权特征
        weighted_feature = tf.reduce_sum(tf.multiply(input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature

    # for saving the model
    # 获取类的配置信息
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state, })
        return config


class CSIModelConfig:
    """
    class for Human Activity Recognition ("bed", "fall", "pickup", "run", "sitdown", "standup", "walk")
    Using CSI (Channel State Information)
    Specifically, the author here wants to classify Human Activity using Channel State Information.
    The deep learning architecture used here is Bidirectional LSTM(双向LSTM网络) stacked with One Attention Layer.
       2019-12, https://github.com/ludlows
    Args:
        win_len   :  integer (1000 default) window length for batching sequence
        step      :  integer (200  default) sliding window by this step
        thrshd    :  float   (0.6  default) used to check if the activity is intensive inside a window
        downsample:  integer >=1 (2 default) downsample along the time axis
    """

    def __init__(self, win_len=1000, step=200, thrshd=0.6, downsample=2):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        # self._labels = ("bed", "fall", "pickup", "run", "sitdown", "standup", "walk")
        self._labels = ('lying', 'sitting', 'running', 'walking', 'pick up', 'wave hand', 'jump', 'squat', 'site down',
        'stand up', 'fall down')
        self._downsample = downsample

        # labelList = ['lying', 'sitting', 'running', 'walking', 'pick up', 'wave hand', 'jump', 'squat', 'site down',
        #          'stand up', 'fall down']

    def preprocessing(self, raw_folder, save=False):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            raw_folder: the folder containing raw CSI
            save      : choose if save the numpy array
        """
        numpy_tuple = extract_csi(raw_folder, self._labels, save, self._win_len, self._thrshd, self._step)
        # downsapmle下采样，降低计算成本，加快速度
        if self._downsample > 1:
            # 下采样操作，对偶数索引进行间隔采样，每隔 self._downsample 个列取一个数据点。
            return tuple([v[:, ::self._downsample, ...] if i % 2 == 0 else v for i, v in enumerate(numpy_tuple)])
        return numpy_tuple

    def load_csi_data_from_files(self, np_files):
        """
        Returns the Numpy Array for training within the format of (X_lable1, y_label1, ...., X_label7, y_label7)
        Args:
            np_files: ('x_bed.npz', 'x_fall.npz', 'x_pickup.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz')
        """
        if len(np_files) != 7:
            raise ValueError('There should be 7 numpy files for bed, fall, pickup, run, sitdown, standup, walk.')
        x = [np.load(f)['arr_0'] for f in np_files]
        # 下采样处理
        if self._downsample > 1:
            x = [arr[:, ::self._downsample, :] for arr in x]
        y = [np.zeros((arr.shape[0], len(self._labels))) for arr in x]
        numpy_list = []
        for i in range(len(self._labels)):
            y[i][:, i] = 1
            numpy_list.append(x[i])
            numpy_list.append(y[i])
        return tuple(numpy_list)

    def build_model(self, n_unit_lstm=200, n_unit_atten=400):
        """
        Returns the Tensorflow Model which uses AttenLayer
        """
        # 下采样处理
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
            # 修改输入层的形状，使其能够接受下采样后的输入
            x_in = tf.keras.Input(shape=(length, 90))
        else:
            x_in = tf.keras.Input(shape=(self._win_len, 90))

        # 双向LSTM层接受输入，并返回一个包含前向和后向LSTM输出的张量。
        # units=n_unit_lstm 指定了LSTM层中的单元数， return_sequences=True 表示输出序列中的每个时间步都会有一个输出。
        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True))(x_in)
        # 添加注意力层
        x_tensor = AttenLayer(n_unit_atten)(x_tensor)
        # 全连接层，将注意力机制处理后的张量映射到输出类别的维度，激活函数用softmax
        pred = tf.keras.layers.Dense(len(self._labels), activation='softmax')(x_tensor)
        # 构建模型
        model = tf.keras.Model(inputs=x_in, outputs=pred)
        return model

    @staticmethod
    def load_model(hdf5path):
        """
        Returns the Tensorflow Model for AttenLayer
        Args:
            hdf5path: str, the model file path
        """
        # 从hdf5路径中加载模型，并将加载的模型赋值给变量 model 。如果模型中包含自定义的层或函数，需要通过 custom_objects 参数提供对这些自定义对象的映射。
        model = tf.keras.models.load_model(hdf5path, custom_objects={'AttenLayer': AttenLayer})
        return model


if __name__ == "__main__":
    import sys
    from sklearn.model_selection import train_test_split
    # if len(sys.argv) != 2:
    #     print("Error! Correct Command: python3 csimodel.py Dataset_folder_path")
    # raw_data_foler = sys.argv[1]

    # preprocessing
    cfg = CSIModelConfig(win_len=1000, step=200, thrshd=0.6, downsample=2)
    # numpy_tuple = cfg.preprocessing('Dataset/Data/', save=True)
    # load previous saved numpy files, ignore this if you haven't saved numpy array to files before
    # numpy_tuple = cfg.load_csi_data_from_files(('x_bed.npz', 'x_fall.npz', 'x_pickup.npz', 'x_run.npz', 'x_sitdown.npz', 'x_standup.npz', 'x_walk.npz'))
    # x_bed, y_bed, x_fall, y_fall, x_pickup, y_pickup, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = numpy_tuple
    # x_train, y_train, x_valid, y_valid = train_valid_split(
    #     (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk),
    #     train_portion=0.9, seed=379)
    # parameters for Deep Learning Model

    path1 = r'D:/pycharm/files/OurActivityDataset/OurActivityDataset/Processed Data/bedroom/'
    path2 = r'D:/pycharm/files/OurActivityDataset/OurActivityDataset/Processed Data/meetingroom/'
    csiData, csiLabel, labelNum = merge_csi_DataAndLabel(path1)

    # 将数据集按照指定的比例拆分为训练集和测试集
    x_train, y_train, x_valid, y_valid = train_test_split(csiData, labelNum, test_size=0.2, random_state=20)

    # 存储构建的模型
    model = cfg.build_model(n_unit_lstm=200, n_unit_atten=400)
    # train，配置模型训练参数
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    # 显示模型的层结构、参数数量以及每一层的输出形状。
    model.summary()
    # 训练模型
    model.fit(
        x_train,
        y_train,
        batch_size=128, epochs=5, #60
        validation_data=(x_valid, y_valid),
        # 使用回调函数保存最好的模型
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint('best_atten.hdf5',
                                               monitor='val_accuracy',
                                               save_best_only=True,
                                               save_weights_only=False)
        ])
    # load the best model
    model = cfg.load_model('best_atten.hdf5')
    y_pred = model.predict(x_valid)

    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(np.argmax(y_valid, axis=1), np.argmax(y_pred, axis=1)))
