import sys
import os
# 将 ConVAE 目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.dataset import SmilesDataset
from util.tokens import Tokenizer
import util.utils as utils
import torch
# import rnn
# import vae
import model.convae as cvae
import argparse
import os
import multiprocessing

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--model', type=str)
# parser.add_argument('-i', '--printInterval', type=int)
# parser.add_argument('--gpu', action='store_true', default=False)
# args = parser.parse_args()
# model_type = args.model
# printInterval = args.printInterval
# useGPU = args.gpu and torch.cuda.is_available()
def main():
    model_type = 'cvae'
    printInterval = 10
    useGPU = True
    device = torch.device('cuda' if useGPU else 'cpu')
    print(f'useGPU = {useGPU}')

    tokenizer = utils.get_tokenizer()
    print(tokenizer.tokensDict)
    smilesDataset = SmilesDataset(
        utils.config['fname_dataset'], tokenizer, utils.config['maxLength'])
    
    # if model_type == 'cvae':
    smilesDataloader = torch.utils.data.DataLoader(
        smilesDataset, batch_size=utils.config['batch_size'], 
        shuffle=True, num_workers=4, 
        collate_fn=smilesDataset.one_hot_collate_fn)
    
    vae_model = cvae.VAE(**utils.config['vae_param'],
                        encoder_state_fname=utils.config['fname_vae_encoder_parameters'],
                        decoder_state_fname=utils.config['fname_vae_decoder_parameters'],
                        device=device)

    for name, layer in vae_model.encoder.named_parameters():
        print(name, layer.shape, layer.dtype,
                layer.requires_grad, layer.device)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    for name, layer in vae_model.decoder.named_parameters():
        print(name, layer.shape, layer.dtype,
                layer.requires_grad, layer.device)

    encoderOptimizer = torch.optim.Adam(
        vae_model.encoder.parameters(), 
        lr=utils.config['lr'], 
        weight_decay=1.0e-4,
        eps=1e-8
    )
    decoderOptimizer = torch.optim.Adam(
        vae_model.decoder.parameters(), 
        lr=utils.config['lr'], 
        weight_decay=1.0e-4,
        eps=1e-8
    )
    encoderScheduler = torch.optim.lr_scheduler.StepLR(
        encoderOptimizer, step_size=1, gamma=0.95)
    decoderScheduler = torch.optim.lr_scheduler.StepLR(
        decoderOptimizer, step_size=1, gamma=0.95)
    
    vae_model.trainModel(smilesDataloader, encoderOptimizer, decoderOptimizer, 
                        encoderScheduler, decoderScheduler, 1.0, 
                        utils.config['num_epoch'], tokenizer, printInterval)

    # elif model_type == 'rnn':
    #     smilesDataloader = torch.utils.data.DataLoader(
    #         smilesDataset, batch_size=utils.config['batch_size'], 
    #         shuffle=True, num_workers=4, 
    #         collate_fn=smilesDataset.collate_fn)
        
    #     rnn_model = rnn.RNN(
    #         **utils.config['rnn_param'], 
    #         state_fname=utils.config['fname_rnn_parameters'], 
    #         device=device)
        
    #     for name, layer in rnn_model.named_parameters():
    #         print(name, layer.shape, layer.dtype,
    #               layer.requires_grad, layer.device)
        
    #     optimizer = torch.optim.Adam(
    #         rnn_model.parameters(), lr=utils.config['lr'], weight_decay=1.0e-4)
    #     scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, step_size=1, gamma=0.95)
        
    #     rnn_model.trainModel(smilesDataloader, optimizer, scheduler,
    #                         utils.config['num_epoch'], utils.config['maxLength'], 
    #                         tokenizer, printInterval)

if __name__ == '__main__':
    # 如果要冻结成exe还需要加这行
    # multiprocessing.freeze_support()
    main()
