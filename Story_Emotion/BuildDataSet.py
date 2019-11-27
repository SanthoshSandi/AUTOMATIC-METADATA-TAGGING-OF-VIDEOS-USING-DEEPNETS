
import os
import numpy as np
import argparse
from hdf5datasetwriter import HDF5DatasetWriter


class BuildDataSet:
    def __init__(self, base_path, num_classes):
        print("\n Base Path: {0}".format(base_path))

        self.input_path = os.path.join(base_path, 'fer2013')

        # directory structure check
        if not os.path.exists(self.input_path):
            print("\n Input directory structure does not exist. Manually create the directory structure "
                  "following documentation")
            exit(-1)

        # directory structure check
        self.hdf5_path = os.path.join(base_path, 'hdf5')
        if not os.path.exists(self.hdf5_path):
            print("\n Uncompressed data directory structure does not exist. Manually create the HDF5 directory" 
                  "following documentation")
            exit(-1)

        # directory structure check
        self.output_path = os.path.join(base_path, 'output')
        if not os.path.exists(self.output_path):
            print("\n Output directory structure does not exist. Manually create the output directory" 
                  "following documentation")
            exit(-1)

        # define number of classes
        self.num_classes = num_classes  # set to 6 if you are ignoring the 'disgust' class

        # define the batch size
        self.batch_size = 64

        print("\n Input path: {0}\n Intermediate HDF5 path: {1}\n Output HDF5 path: {2}\n # of Emotions: {3}"
              .format(self.input_path, self.hdf5_path, self.output_path, self.num_classes))

    def config_dataset(self):
        input_csv_file = os.path.join(self.input_path, 'fer2013.csv')
        train_HDF5 = os.path.join(self.hdf5_path, 'train.hdf5')
        val_HDF5 = os.path.join(self.hdf5_path, 'val.hdf5')
        test_HDF5 = os.path.join(self.hdf5_path, 'test.hdf5')

        # check if the csv file is properly placed or not
        if not os.path.isfile(input_csv_file):
            print("\nThe FER2013 dataset in .csv format was not found. Please manually place that file in the directory")
            exit(-1)

        print('\n Input dataset file: {0}\n Train dataset: {1}\n Validate dataset: {2}\n Test dataset: {3}'
              .format(input_csv_file, train_HDF5, val_HDF5, test_HDF5))

        return input_csv_file, train_HDF5, val_HDF5, test_HDF5

    def build_dataset(self, input_csv_file, train_HDF5, val_HDF5, test_HDF5):
        print("\n [STATUS: ] Loading data.... Please wait")

        # Open the Kaggle dataset input CSV file
        input_file = open(input_csv_file)
        input_file.__next__()

        # initiate the training, validation and test data sets (empty)
        (trainImages, trainLabels) = ([], [])
        (valImages, valLabels) = ([], [])
        (testImages, testLabels) = ([], [])

        # loop over each of the input file
        for row in input_file:
            # extract the label, image, and usage from the row
            (label, image, usage) = row.strip().split(",")
            label = int(label)

            # We are going to ignore the disgust label and merge them with angry (refer Memong paper)
            if self.num_classes == 6:
                # merge together the "anger" and "disgust classes
                if label == 1:
                    label = 0

                # if label has a value greater than zero, subtract one from
                # it to make all labels sequential (not required, but helps
                # when interpreting results)
                if label > 0:
                    label -= 1

            # reshape the flattened pixel list into a 48x48 (grayscale) image
            image = np.array(image.split(" "), dtype=np.uint8)
            image = image.reshape((48, 48))

            """
            ===============================================================================================
            Splitting the data into train, validation and test set based on the usage given in the CSV file
            NOTE: Validation is noted as PrivateTest in the CSV file 
            ===============================================================================================
            """

            # check if usage = Training
            if usage == "Training":
                trainImages.append(image)
                trainLabels.append(label)

            # check if usage = Validation
            elif usage == "PrivateTest":
                valImages.append(image)
                valLabels.append(label)

            # check if usage = "Test"
            elif usage == "PublicTest":
                testImages.append(image)
                testLabels.append(label)

        # list pair for training, validation and test sets along with their corresponding files
        datasets = [(trainImages, trainLabels, train_HDF5), (valImages, valLabels, val_HDF5),
                    (testImages, testLabels, test_HDF5)]

        for (images, labels, dataset_path) in datasets:
            # check if file exists
            if os.path.isfile(dataset_path):
                print('File {0} already exists. Skipping...'.format(dataset_path))
                continue
            else:
                # create HDF5 writer
                print("\n [STATUS: ] Building and Writing {0}...".format(dataset_path))
                writer = HDF5DatasetWriter((len(images), 48, 48), dataset_path)

                # loop over each image and add them to each of the dataset files
                for (image, label) in zip(images, labels):
                    writer.add([image], [label])

                writer.close()

        input_file.close()
        return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path",
                        type=str, required=True)
    parser.add_argument("-n", "--num_emotions", 
                        type=int, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    if not os.path.exists(base_path):
        print("\n Base path does not exist.")
        exit(0)

    num_classes = args.num_emotions
    bds = BuildDataSet(base_path, num_classes)
    (input_csv_file, train_HDF5, val_HDF5, test_HDF5) = bds.config_dataset()
    bds.build_dataset(input_csv_file, train_HDF5, val_HDF5, test_HDF5)
