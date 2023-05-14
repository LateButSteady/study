#! usr/bin/env python
#-*- encoding: utf-8 -*-

import os
import torch
import numpy as np
from loaddata import LoadData
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from tqdm import tqdm

from mnist import Mnist

dir_now = os.path.dirname(__file__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


##### random seed #####
seed = 4444
np.random.seed(seed)
torch.manual_seed(seed)


def main():
    #####################################
    ############# prep data #############
    #####################################

    X_train, X_test, y_train, y_test = prep_data(load=True)
    
    X_train_tensor = torch.Tensor(X_train).to(device)
    X_test_tensor  = torch.Tensor(X_test).to(device)
    y_train_tensor = torch.Tensor(y_train).to(device)
    y_test_tensor  = torch.Tensor(y_test).to(device)
    
    
    lr = 1e-3
    num_epoch = 5
    batch_size = 100

    #####  mnist data ref  #####
    # mnist_train = dsets.MNIST(root=r"F:\project\coding_test\MLDL\data\img\MNIST_JPG_train", # 다운로드 경로 지정
    #                         train=True, # True를 지정하면 훈련 데이터로 다운로드
    #                         transform=transforms.ToTensor(), # 텐서로 변환
    #                         download=True)
    # mnist_test = dsets.MNIST(root=r"F:\project\coding_test\MLDL\data\img\MNIST_JPG_test", # 다운로드 경로 지정
    #                      train=False, # False를 지정하면 테스트 데이터로 다운로드
    #                      transform=transforms.ToTensor(), # 텐서로 변환
    #                      download=True)
    # data_loader = DataLoader(dataset=mnist_train,
    #                               batch_size=batch_size,
    #                               shuffle=True,
    #                               drop_last=False)

    data_train = TensorDataset(X_train_tensor, y_train_tensor)
    
    data_loader = DataLoader(dataset=data_train,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False)

    total_batch = len(data_loader)
    print('총 배치의 수 : {}'.format(total_batch))

    model = Mnist().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters() , lr=lr)



    ##### Training #####
    for epoch in range(num_epoch):
        avg_cost = 0

        for batch_ind, samples in enumerate(data_loader):
            X, y = samples

            X = X.to(device)
            y = y.to(torch.int64).to(device)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            avg_cost += loss / total_batch
        
        print(f"epoch = {epoch} // avg_cost = {avg_cost:.4f}")
        
    ##### save the model #####
    ##### model path #####
    path_model_state_dict = r"F:\project\coding_test\MLDL\data\img\mnist_model_state_dict.pth"
    path_model = r"F:\project\coding_test\MLDL\data\img\mnist_model.pth"
    path_model_torchscript = r"F:\project\coding_test\MLDL\data\img\mnist_model_torchscript.pth"

    # 1. state_dict로 학습 para만 저장
    torch.save(model.state_dict(), path_model_state_dict)

    # 2-1. 모델 자체 저장
    # torch.save(model, path_model) 

    # 2-2. TorchScript로 저장
    # model_scripted = torch.jit.script(model)
    # model_scripted.save(path_model_torchscript)


    ##### load the model #####
    # 1. 학습 para만 로딩
    model = Mnist().to(device)  # 구조 먼저 선언
    model_state_dict = torch.load(path_model_state_dict, map_location=device)  # state_dict 먼저 로딩
    model.load_state_dict(model_state_dict)  # model에 state_dict 입히기
    model.eval()  # dropout, batch norm 를 잠금

    # 2-1. 모델 자체 로딩
    # model = torch.load(path_model)
    # model.eval()  # dropout, batch norm 를 잠금

    # 2-2. TorchScript로 로딩
    # model = torch.jit.load(path_model_torchscript)
    # model.eval()  # dropout, batch norm 를 잠금

    ##### Test #####
    with torch.no_grad():   # NO update grad
        pred = model(X_test_tensor)
        pred_correct = torch.argmax(pred, dim=1) == y_test_tensor
        accuracy = pred_correct.float().mean()
        print("Acc = ", accuracy.item())
        
    return 0



def prep_data(load: bool):
    # load=False: save .npy
    # load=True:  load .npy

    path_X_train = r"F:\project\coding_test\MLDL\data\img\mnist_X_train.npy"
    path_X_test = r"F:\project\coding_test\MLDL\data\img\mnist_X_test.npy"
    path_y_train = r"F:\project\coding_test\MLDL\data\img\mnist_y_train.npy"
    path_y_test = r"F:\project\coding_test\MLDL\data\img\mnist_y_test.npy"

    # load .npy
    if load:
        X_train = np.load(r"F:\project\coding_test\MLDL\data\img\mnist_X_train.npy")
        X_test  = np.load(r"F:\project\coding_test\MLDL\data\img\mnist_X_test.npy")
        y_train = np.load(r"F:\project\coding_test\MLDL\data\img\mnist_y_train.npy")
        y_test  = np.load(r"F:\project\coding_test\MLDL\data\img\mnist_y_test.npy")

    # save .npy
    else:
        ld = LoadData()

        list_y_train = []
        list_y_test  = []

        for i in tqdm(range(0, 10)):
            # X train
            dir_mnist_train = os.path.join(dir_now, "data", "img", "MNIST_JPG_train", str(i))
            np_img_train_tmp = ld.load(path=dir_mnist_train, extension='jpg')

            # X test
            dir_mnist_test = os.path.join(dir_now, "data", "img", "MNIST_JPG_test", str(i))
            np_img_test_tmp = ld.load(path=dir_mnist_test, extension='jpg')
        
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



if __name__ == "__main__":
    main()