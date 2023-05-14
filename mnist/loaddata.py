#! usr/bin/env python
#-*- encoding: utf-8 -*-
import os
import numpy as np
import glob
import tqdm

class LoadData():
    def __init__(self, logging = None, lib='pil'):
        self.X_train   = None
        self.y_train   = None
        self.X_test    = None
        self.y_test    = None
        self.X_valid   = None
        self.y_valid   = None
        self.path_type = ""
        self.num_data  = 0
        self.path_log  = ""
        self.lib       = lib


    def load(self, path: str, extension='png', fit_size=True, cv2_RGB=True):
        """
        load image data
        supported extensions: pil, png, jpg
        """

        if os.path.isdir(path):
            print("[INFO] The input 'path' indicates directory")
        else:
            print("[INFO] The input 'path' indicates file")

        # check dir
        if os.path.isdir(path):
            self.data_type = "image"

            list_file = os.listdir(path)
            list_file_img = [os.path.join(path, x) for x in list_file if x in extension]

            # check image file extension
            if self.lib.lower() == 'pil':
                imgs = self.load_img_pil(path, extension)
            elif self.lib.lower() == 'cv2':
                imgs = self.load_img_cv2(path, extension, cv2_RGB)
            elif self.lib.lower() == 'plt':
                imgs = self.load_img_plt(path, extension)
            elif self.lib.lower() == 'sklearn':
                imgs = self.load_img_sklearn(path, extension)
            else:
                print("[ERROR] Check the lib arg to load images")

            # Check if image is successfully loaded
            try:
                print(f"[INFO] img.shape = {imgs.shape}")
            except:
                print(f"[ERROR] Failed loading image data")

            return imgs

        # csv
        elif path.endswith(".csv"):
            self.data_type = "csv"

    

    def load_img_pil(self, dir_img, extension='png', fit_size=True):
        """
        load image files using PIL library
        """
        from PIL import Image
        
        print('[INFO] Loading imgs using PIL')

        list_np_img = []
        for path_img in glob.glob(dir_img + '/*.' + extension):
            img = np.array(Image.open(path_img))

            if len(img.shape) == 2:
                # 2-D image (NO RGB dimension)
                list_np_img.append(img)
            elif len(img.shape) == 3:
                # 3-D image (RGB dimension) -> only get RGB
                list_np_img.append(img[:,:,:3])
        np_imgs = np.array(list_np_img)

        print('[INFO] Completed Loading imgs using PIL')
        return np_imgs


    def load_img_cv2(self, dir_img, extension='png', fit_size=True, cvt2RGB=True):
        """
        load image files using opencv2 library
        """
        import cv2

        print('[INFO] Loading imgs using cv2')

        list_np_img = []

        list_path_img = [x for x in glob.glob(dir_img + '/*.' + extension)]
        if len(list_path_img) == 0:
            raise FileExistsError('[ERROR] No image is found')

        for path_img in list_path_img:
            list_np_img.append(np.array(cv2.imread(path_img)))
        np_imgs = np.array(list_np_img)
        if cvt2RGB:
            np_imgs = np_imgs[:, :, :, ::-1]

        print('[INFO] Completed Loading imgs using cv2')
        return np_imgs


    def load_img_plt(self, dir_img, extension='png', fit_size=True):
        """
        load image files using matplotlib library
        """
        import matplotlib.pyplot as plt

        print('[INFO] Loading imgs using matplotlib')

        list_np_img = []
        for path_img in glob.glob(dir_img + '/*.' + extension):
            list_np_img.append(np.array(plt.imread(path_img)))
        np_imgs = np.array(list_np_img)

        print('[INFO] Completed Loading imgs using matplotlib')
        return np_imgs


    def load_img_sklearn(self, dir_img, extension='png', fit_size=True):
        """
        load image files using sklearn library
        """
        
        from skimage import io

        print('[INFO] Loading imgs using sklearn')

        list_np_img = []
        for path_img in glob.glob(dir_img + '/*.' + extension):
            list_np_img.append(np.array(io.imread(path_img)))
        np_imgs = np.array(list_np_img)

        print('[INFO] Completed Loading imgs using sklearn')
        return np_imgs



    def prep_data_mnist(self, load=True, dir_now=""):
        """
        Load prepared MNIST image data

        load=False: save .npy
        load=True:  load .npy

        dir_now: 
        """

        if (not load) and (dir_now == ""):
            raise("Enter dir_now if load=False")
        
        path_X_train = r"F:/project/coding_test/MLDL/data/img/mnist_X_train.npy"
        path_X_test = r"F:/project/coding_test/MLDL/data/img/mnist_X_test.npy"
        path_y_train = r"F:/project/coding_test/MLDL/data/img/mnist_y_train.npy"
        path_y_test = r"F:/project/coding_test/MLDL/data/img/mnist_y_test.npy"

        # load .npy
        if load:
            X_train = np.load(r"F:/project/coding_test/MLDL/data/img/mnist_X_train.npy")
            X_test  = np.load(r"F:/project/coding_test/MLDL/data/img/mnist_X_test.npy")
            y_train = np.load(r"F:/project/coding_test/MLDL/data/img/mnist_y_train.npy")
            y_test  = np.load(r"F:/project/coding_test/MLDL/data/img/mnist_y_test.npy")

        # save .npy
        else:
            

            list_y_train = []
            list_y_test  = []

            for i in tqdm(range(0, 10)):
                # X train
                dir_mnist_train = os.path.join(dir_now, "data", "img", "MNIST_JPG_train", str(i))
                np_img_train_tmp = self.load(path=dir_mnist_train, extension='jpg')

                # X test
                dir_mnist_test = os.path.join(dir_now, "data", "img", "MNIST_JPG_test", str(i))
                np_img_test_tmp = self.load(path=dir_mnist_test, extension='jpg')
            
                # y
                list_y_train += [i] * np_img_train_tmp.shape[0]
                list_y_test +=  [i] * np_img_test_tmp.shape[0]
            
                if i == 0:
                    X_train_np = np_img_train_tmp
                    X_test_np  = np_img_test_tmp
                else:
                    X_train_np = np.vstack((X_train_np, np_img_train_tmp))
                    X_test_np  = np.vstack((X_test_np, np_img_test_tmp))

            sz_X_train = X_train_np.shape
            sz_X_test  = X_test_np.shape

            X_train = np.reshape(X_train_np, (sz_X_train[0], 1, sz_X_train[1], sz_X_train[2]))
            X_test  = np.reshape(X_test_np,  (sz_X_test[0],  1, sz_X_test[1],  sz_X_test[2]))

            y_train = np.array(list_y_train)
            y_test  = np.array(list_y_test)

            np.save(path_X_train, X_train)
            np.save(path_X_test, X_test)
            np.save(path_y_train, y_train)
            np.save(path_y_test, y_test)

        return X_train, X_test, y_train, y_test



    def load_num_data(self, X_dim=1):
        """
        load numeric data for regression
          - lin reg
          - tree, xgboost
        """
        if X_dim > 3 or X_dim < 1:
            raise("[ERROR] No more usable dimension in the iris data")
        
        import pandas as pd
        df = pd.read_csv(r"F:/project/coding_test/python/iris.csv")
        
        y = df['petalwidth'].to_numpy().reshape((-1,1))
        
        X_names = ""
        if X_dim == 1:
            X_names = ['sepallength']
            X = df['sepallength'].to_numpy().reshape((-1, X_dim))
        if X_dim == 2:
            X_names = ['sepallength', 'petallength']
            X = df[X_names].to_numpy().reshape((-1, X_dim))
        if X_dim == 3:
            X_names = ['sepallength', 'petalwidth', 'sepalwidth']
            X = df[X_names].to_numpy().reshape((-1, X_dim))

        return X, y, X_names