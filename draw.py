from tools.data_loader import dataLoader
from tqdm import tqdm
from tools.det_utils import tensor2bgr_numpy
import cv2


data_root = './data/INRIAPerson/Train/pos'
data_sampler = None
data_loader = dataLoader(data_root, [416, 416], is_augment=True,
                             batch_size=1, sampler=data_sampler)

for index, img_tensor_batch in tqdm(enumerate(data_loader)):
    bgr_im_int8 = tensor2bgr_numpy(img_tensor_batch[0])
    cv2.imwrite(f'./data/test/aug/{index}.png', bgr_im_int8)
