
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from BuildDataSet import BuildDataSet
from hdf5datasetgenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import os


class EmotionNetworkTest:
    def __init__(self, base_path, num_classes):
        print("\n Testing trained model for emotion recognition of FER 2013 database (TEST SPLIT)")
        self.base_path = base_path
        self.num_classes = num_classes

    def testNetwork(self, trained_model_path, test_dataset_path):
        test_augmentation = ImageDataGenerator(rescale=1/255)
        iap = ImageToArrayPreprocessor()

        # get file details
        config = BuildDataSet(base_path=self.base_path, num_classes=self.num_classes)

        test_generation = HDF5DatasetGenerator(test_dataset_path, 64,
                                               aug=test_augmentation, preprocessors=[iap], classes=config.num_classes)

        # load pre-trained model to test accuracy
        print("\n Loading model: {0}".format(trained_model_path))

        trained_model = load_model(trained_model_path)

        # evaluate model against test set
        print("Evaluate model against test set")
        (test_loss, test_acc) = trained_model.evaluate_generator(test_generation.generator(),
                                                       steps=test_generation.numImages // config.batch_size,
                                                       max_queue_size=config.batch_size * 2)

        print("\n \n FINAL MODEL ACCURACY: {:.2f} %".format(test_acc*100))

        print("\n \n *********************Testing Complete*********************\n")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path",
                        help="The base directory path of the dataset. \n Please ensure that you are"
                             " following the directory structure outlined in the documentation.\n "
                             "Also ensure that the fer2013.csv file is place in the fer2013 folder",
                        type=str, required=True)

    parser.add_argument("-n", "--num_emotions", help="Number of emotions same as dataset build file. \n"
                                                     "If num_emotions = 6 (we merge anger and disgust)\n"
                                                     "if num_emotions = 7 (we use all 7 defined emotions)\n"
                                                     "Default value = 7",
                        type=int, required=True)

    parser.add_argument("-im", "--input_model", help="The name of the pretrained input model to be loaded for test.",
                        type=str, required=True)

    args = parser.parse_args()

    emo_test = EmotionNetworkTest(base_path=args.base_path, num_classes=args.num_emotions)
    emo_test.testNetwork(trained_model_path=os.path.join(args.base_path, 'output', args.input_model),
                         test_dataset_path=os.path.join(args.base_path, 'hdf5', "test.hdf5"))

