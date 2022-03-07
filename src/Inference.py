import os
import warnings
import argparse
from DataSet_custom import Dataset, ModelConfig
from Main import Test_Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgPath', type=str, default='./data')
    parser.add_argument('--modelPath', type=str, default='./models/Detection.hdf5')
    parser.add_argument('--clsmodelPath', type=str, default='./models/Classifier.hdf5')
    parser.add_argument('--savePath', type=str, default='./results')
    parser.add_argument('--gpu_num', type=str, default='0')
    args = parser.parse_args()

    Dirs = {'Aortic_bio': 'Aortic_bio', 'Aortic_mechanical': 'Aortic_mechanical', 'Mitral_bio': 'Mitral_bio', 'Mitral_mechanical': 'Mitral_mechanical', 'Mitral_ring': 'Mitral_ring', 
            'Pulmonary_bio': 'Pulmonary_bio', 'Pulmonary_mechanical': 'Pulmonary_mechanical','Tricuspid_bio': 'Tricuspid_bio','Tricuspid_mechanical': 'Tricuspid_mechanical','Tricuspid_ring': 'Tricuspid_ring'}
    Classes = {'Aortic_bio': 1, 'Aortic_mechanical': 2, 'Mitral_bio': 3, 'Mitral_mechanical': 4, 'Mitral_ring': 5, 
            'Pulmonary_bio': 6, 'Pulmonary_mechanical': 7, 'Tricuspid_bio': 8, 'Tricuspid_mechanical': 9, 'Tricuspid_ring': 10}
    
    modelConfig = ModelConfig(len(Classes))
    modelConfig.DETECTION_MIN_CONFIDENCE = 0.4 

    dataset_test = Dataset()
    dataset_test.set_dataset(imagePath=args.imgPath, maskPath=None, dirDicts=Dirs, classDicts=Classes)

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_num

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Test_Dataset(args.savePath, modelConfig, args.modelPath, dataset_test, maskFp=None, clsmodelPath = args.clsmodelPath, 
                    saveFig = True, Cls_thr=0.3)
        

if __name__ == '__main__':
    main()