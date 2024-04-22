import os
from ConvLSTM.encoder import Encoder
from ConvLSTM.decoder import Decoder
from ConvLSTM.model import ED
from ConvLSTM.net_params import convgru_encoder_params, convgru_decoder_params
from data.CLDataLoader import weatherDataset
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from earlystoppingCL import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from ER import *
from buffer import *

MODELSAVE = "ER-MIR"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=32,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-3, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=4,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=15,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=100000, type=int, help='sum of epochs')
args = parser.parse_args()


LR = args.lr
random_seed = 41
np.random.seed(random_seed) #之后调用np.random保证一致
torch.manual_seed(random_seed)  #同上，为cpu设置
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)  #为所有gpu设置
else:
    torch.cuda.manual_seed(random_seed) #为当前gpu设置
torch.backends.cudnn.deterministic = True #输入不变
torch.backends.cudnn.benchmark = False


testFolder = weatherDataset(n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                             numpy_folder = 'data/Xiannorm')
testLoader = torch.utils.data.DataLoader(testFolder,
                                          batch_size=2,
                                          shuffle=True)


encoder_params = convgru_encoder_params
decoder_params = convgru_decoder_params

def getDateloader(bi):
    trainFolder = weatherDataset(is_train=True,
                                 begin_index=bi,
                                 n_frames_input=args.frames_input,
                                 n_frames_output=args.frames_output,
                                 numpy_folder='data/Xiannorm')
    validFolder = weatherDataset(is_val=True,
                                 begin_index=bi,
                                 n_frames_input=args.frames_input,
                                 n_frames_output=args.frames_output,
                                 numpy_folder='data/Xiannorm')
    tl = torch.utils.data.DataLoader(trainFolder,
                                              batch_size=args.batch_size,
                                              shuffle=True)
    vl = torch.utils.data.DataLoader(validFolder,
                                              batch_size=2,
                                              shuffle=True)
    index = []
    for i in range(50):
        index.append(bi + (i // 5) * 6 + i % 5)
    return tl, vl, index

def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    save_dir = './save_model/' + MODELSAVE
    print('==> loading existing model')
    model_info = torch.load(os.path.join(save_dir, 'base_model.pth.tar'))
    net.load_state_dict(model_info['state_dict'])
    lossfunction = nn.MSELoss().cuda()
    conti_model_test_loss = []
    conti_model_val_loss = []
    task_num = 20
    last_best_model = ''
    args.cuda = True
    bf = Buffer('data/Xiannorm', [i for i in range(3000)])
    for task_id in range(task_num):
        cur_epoch = 0
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        # mini_val_loss = np.inf
        if task_id > 0:
            model_info = torch.load(os.path.join(last_best_model))
            net.load_state_dict(model_info['state_dict'])
            with torch.no_grad():
                net.eval()
                t = tqdm(testLoader, leave=False, total=len(testLoader))
                loss_all = []
                for i, (idx, targetVar, inputVar) in enumerate(t):
                    if i == 3000:
                        break
                    inputs = inputVar.to(device)
                    label = targetVar.to(device)
                    pred = net(inputs) # B,S,C,H,W
                    B, S, C, H, W = label.shape
                    label = label.reshape(B, S, C, H, W)
                    pred = pred.reshape(B, S, C, H, W)
                    pred = pred[:,:,:,16:48,16:48]
                    label = label[:,:,:,16:48,16:48]
                    loss = lossfunction(pred, label)
                    # loss_aver = loss.item()
                    loss_aver = loss.item()
                    loss_all.append(loss_aver)
                print("validloss: {:.6f},  task : {:02d}".format(np.average(loss_all), task_id),end = '\r', flush=True)
                conti_model_test_loss.append(np.average(loss_all))
        run_dir = os.path.join('./runs/', MODELSAVE, str(task_id))
        save_dir = os.path.join('./save_model/', MODELSAVE, str(task_id))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        tb = SummaryWriter(run_dir)
        print('****************************************************')
        early_stopping = EarlyStopping(patience=40, verbose=True)
        optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
        pla_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25, verbose=True)
        bi = task_id * 60 + 3000
        args.buffer_batch_size = 20
        args.method = 'mir_replay'
        if args.method == 'season_replay':
            bf.update_season(bi)
        trainLoader, validLoader, x_index = getDateloader(bi)
        for epoch in range(cur_epoch, args.epochs + 1):
            ###################
            # train the model #
            ###################
            t = tqdm(trainLoader, leave=False, total=len(trainLoader))
            for i, (idx, targetVar, inputVar) in enumerate(t):
                inputs = inputVar.to(device)  # B,S,C,H,W
                label = targetVar.to(device)  # B,S,C,H,W
                optimizer.zero_grad()
                net.train() #进入训练模式
                net, loss_aver = retrieve_replay_update(args, begin_index=bi, model=net, opt=optimizer, input_x=inputs, input_y=label, x_index=x_index, loss_fuction=lossfunction, buffer=bf, amt=100,rehearse=True)
                # pred = net(inputs)  # B,S,C,H,W
                # B, S, C, H, W = label.shape
                # label = label.reshape(B, S, C, H, W)
                # pred = pred.reshape(B, S, C, H, W)
                #
                # pred = pred[:,:,:,16:48,16:48]
                # label = label[:,:,:,16:48,16:48]
                # loss = lossfunction(pred, label)
                # loss_aver = loss.item() / args.batch_size
                # train_losses.append(loss_aver)
                # loss.backward() #反向传播，计算梯度
                # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0) #设置梯度不超过10
                # optimizer.step()
                t.set_postfix({
                    'trainloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
            tb.add_scalar('TrainLoss', loss_aver, epoch)
            ######################
            # validate the model #
            ######################
            with torch.no_grad():
                net.eval()
                t = tqdm(validLoader, leave=False, total=len(validLoader))
                for i, (idx, targetVar, inputVar) in enumerate(t):
                    if i == 3000:
                        break
                    inputs = inputVar.to(device)
                    label = targetVar.to(device)
                    # label = label[:,0,...]
                    # label = label.unsqueeze(1)
                    pred = net(inputs) # B,S,C,H,W
                    B, S, C, H, W = label.shape
                    label = label.reshape(B, S, C, H, W)
                    pred = pred.reshape(B, S, C, H, W)

                    pred = pred[:,:,:,16:48,16:48]
                    label = label[:,:,:,16:48,16:48]
                    loss = lossfunction(pred, label)
                    # loss_aver = loss.item()
                    loss_aver = loss.item()
                    # record validation loss
                    valid_losses.append(loss_aver)
                    print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                    t.set_postfix({
                        'validloss': '{:.6f}'.format(loss_aver),
                        'epoch': '{:02d}'.format(epoch)
                    })
            pla_lr_scheduler.step()  # lr_scheduler

            tb.add_scalar('ValidLoss', loss_aver, epoch)
            torch.cuda.empty_cache()
            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            epoch_len = len(str(args.epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.6f} ' +
                         f'valid_loss: {valid_loss:.6f}')

            print(print_msg)
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            # pla_lr_scheduler.step(valid_loss)  # lr_scheduler
            model_dict = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            temp_model = early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
            print("model",temp_model)
            if temp_model != '':
                last_best_model = temp_model
            if early_stopping.early_stop:
                print("Early stopping")
                break
        bf.append_set(x_index)
        # del optimizer
        # del pla_lr_scheduler
        conti_model_val_loss.append(min(avg_valid_losses))

    with open(os.path.join('save_model', MODELSAVE,"conti_test_losses.txt"), 'wt') as f:
        for i in conti_model_test_loss:
            print(i, file=f)

    with open(os.path.join('save_model', MODELSAVE,"conti_val_losses.txt"), 'wt') as f:
        for i in conti_model_val_loss:
            print(i, file=f)



if __name__ == "__main__":
    train()

