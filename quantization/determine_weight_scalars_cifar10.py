import numpy as np
import os

read_folder_name = 'extracted_params/CIFAR10/old_1_511/'

write_folder_name = 'scalars/CIFAR10/old_1_511/'
if not os.path.isdir(write_folder_name):
    os.mkdir(write_folder_name)


for l in ['1','2','3','4','5','6']:
    name = 'linear_layers'+l+'.weight.npy'
    np.save(write_folder_name+name, np.power(2.0,np.ceil(np.log2(np.amax(np.absolute(np.load(read_folder_name+name)))))))
    print('done with '+name)
    print(np.load(write_folder_name+name))

    name = 'linear_layers'+l+'.bias.npy'
    np.save(write_folder_name+name, np.power(2.0,np.ceil(np.log2(np.amax(np.absolute(np.load(read_folder_name+name)))))))
    print('done with '+name)
    print(np.load(write_folder_name+name))


