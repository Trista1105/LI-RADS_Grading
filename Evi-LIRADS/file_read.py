import SimpleITK as sitk
import numpy as np

def img_path_read_ZhongShan(name,name_path, data_site):
    # read image, tumor mask, and liver mask accordingly
    path = name_path + '/' + name

    dyn1_path=path + '/' + 'img_pre.nii.gz'
    dyn2_path=path + '/' + 'img_A.nii.gz'
    dyn3_path=path + '/' + 'img_V.nii.gz'
    dyn4_path=path + '/' + 'img_D.nii.gz'


    t_pre_path=path+'/'+'C_pre.nii.gz'
    t_A_path = path + '/' + 'C_A.nii.gz'
    t_V_path = path + '/' + 'C_V.nii.gz'
    t_D_path = path + '/' + 'C_D.nii.gz'

    l_pre_path = path + '/' + 'liver_mask_dyn1.nii.gz'
    l_A_path = path + '/' + 'liver_mask_dyn2.nii.gz'
    l_V_path = path + '/' + 'liver_mask_dyn3.nii.gz'
    l_D_path = path + '/' + 'liver_mask_dyn4.nii.gz'


    name_dic={}
    name_dic['pre']=dyn1_path
    name_dic['A']=dyn2_path
    name_dic['V']=dyn3_path
    name_dic['D']=dyn4_path

    name_dic['t_pre']=t_pre_path
    name_dic['t_A'] = t_A_path
    name_dic['t_V'] = t_V_path
    name_dic['t_D'] = t_D_path

    name_dic['l_pre']=l_pre_path
    name_dic['l_A'] = l_A_path
    name_dic['l_V'] = l_V_path
    name_dic['l_D'] = l_D_path

    return name_dic


def img_array_read(img_path):
    img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(img)

    spacing = np.array(img.GetSpacing())
    return img_array, spacing




    







