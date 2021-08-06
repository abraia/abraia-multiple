import os
import wget
import math
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


def load_dataset(dataset, split=0.9):
    """Load one of the available hyperspectral datasets (IP, PU, SA, KSC)."""
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    
    if dataset == 'IP':
        if not os.path.exists('datasets/Indian_pines_corrected.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                          'datasets/Indian_pines_corrected.mat')
        if not os.path.exists('datasets/Indian_pines_gt.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
                          'datasets/Indian_pines_gt.mat')
        data_hsi = sio.loadmat(
            'datasets/Indian_pines_corrected.mat')['indian_pines_corrected']
        gt_hsi = sio.loadmat('datasets/Indian_pines_gt.mat')['indian_pines_gt']

    if dataset == 'PU':
        if not os.path.exists('datasets/PaviaU.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                          'datasets/PaviaU.mat')
        if not os.path.exists('datasets/PaviaU_gt.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                          'datasets/PaviaU_gt.mat')
        data_hsi = sio.loadmat('datasets/PaviaU.mat')['paviaU']
        gt_hsi = sio.loadmat('datasets/PaviaU_gt.mat')['paviaU_gt']

    if dataset == 'SA':
        if not os.path.exists('datasets/Salinas_corrected.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
                          'datasets/Salinas_corrected.mat')
        if not os.path.exists('datasets/Salinas_gt.mat'):
            wget.download('http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
                          'datasets/Salinas_gt.mat')
        data_hsi = sio.loadmat('datasets/Salinas_corrected.mat')['salinas_corrected']
        gt_hsi = sio.loadmat('datasets/Salinas_gt.mat')['salinas_gt']

    if dataset == 'KSC':
        if not os.path.exists('datasets/KSC.mat'):
            wget.download('http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                          'datasets/KSC.mat')
        if not os.path.exists('datasets/KSC_gt.mat'):
            wget.download('http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat',
                          'datasets/KSC_gt.mat')
        data_hsi = sio.loadmat('datasets/KSC.mat')['KSC']
        gt_hsi = sio.loadmat('datasets/KSC_gt.mat')['KSC_gt']

    K = data_hsi.shape[2]
    TOTAL_SIZE = np.sum(gt_hsi != 0)
    TRAIN_SIZE = math.ceil(TOTAL_SIZE * split)

    shapeor = data_hsi.shape
    data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
    data_hsi = PCA(n_components=K).fit_transform(data_hsi)
    shapeor = np.array(shapeor)
    shapeor[-1] = K
    data_hsi = data_hsi.reshape(shapeor)
    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE


if __name__ == '__main__':
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE = load_dataset('IP')
    image_x, image_y, BAND = data_hsi.shape
    data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
    print('The class numbers of the HSI data is:', max(gt))
