import click
import numpy as np 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 
import torch.optim as optim 

from torch.utils.data import DataLoader as DTL 
from optimization.dataset import Source 

import torch.distributed as td 
import torch.multiprocessing as mp 

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DSP

from modelization.extractor import Extractor 
from modelization.generator import Generator 
from modelization.discriminator import Discriminator

from libraries.strategies import *
from loguru import logger 

def train(gpu_idx, node_rank, nb_gpus_per_node, world_size, server_config, root_path, nb_epochs, bt_size):
	worker_rank = node_rank * nb_gpus_per_node + gpu_idx
	td.init_process_group(backend='nccl', world_size=world_size, rank=worker_rank, init_method=server_config)
	
	th.manual_seed(0)
	th.cuda.set_device(gpu_idx)

	source = Source(root_path=root_path)
	picker = DSP(dataset=source, num_replicas=world_size, rank=worker_rank)
	loader = DTL(dataset=source, shuffle=True, batch_size=bt_size, sampler=picker)

	E = DDP(Extractor().eval().cuda(gpu_idx), device_ids=[gpu_idx])  # vgg19 network
	G = DDP(Generator(i_channels=3, o_channels=64, num_blocks=16, num_scaling_block=2).cuda(gpu_idx), device_ids=[gpu_idx])
	D = DDP(Discriminator(i_channels=3, o_channels=64, num_super_blocks=4, num_neurons=1024).cuda(gpu_idx), device_ids=[gpu_idx])

	pixel_loss = nn.L1Loss().cuda(gpu_idx)
	content_loss = nn.L1Loss().cuda(gpu_idx)
	adversarial_loss = nn.BCEWithLogitsLoss().cuda(gpu_idx)

	solver_generator = optim.Adam(params=G.parameters(), lr=0.0002, betas=(0.9, 0.999))
	solver_discriminator = optim.Adam(params=D.parameters(), lr=0.0002, betas=(0.9, 0.999))

	message_fmt = '[%03d/%03d]:%05d | ED => %07.3f | EG => %07.3f'
	epoch_counter = 0
	while epoch_counter < nb_epochs:
		for index, (low_resolution_img, high_resolution_img) in enumerate(loader):
			# create labels (real and fake)
			real_labels = th.ones(low_resolution_img.shape[0])[:, None].cuda(gpu_idx)
			fake_labels = th.zeros(low_resolution_img.shape[0])[:, None].cuda(gpu_idx)

			# train generator
			solver_generator.zero_grad()
		
			super_resolved_img = G(low_resolution_img) # G(I_LR) = I_SR ~ I_HR
			hr_is_real_or_fake = D(high_resolution_img).detach()
			sr_is_real_or_fake = D(super_resolved_img)
			
			loss_g0 = pixel_loss(super_resolved_img, high_resolution_img)
			loss_g1 = content_loss(E(super_resolved_img), E(high_resolution_img).detach())
			loss_g2 = adversarial_loss(sr_is_real_or_fake - th.mean(hr_is_real_or_fake), real_labels)
			loss_g3 = loss_g1 + 0.01 * loss_g0 + 0.005 * loss_g2
			
			loss_g3.backward()
			
			solver_generator.step()

			# train discriminator
			solver_discriminator.zero_grad()

			hr_is_real_or_fake = D(high_resolution_img)
			sr_is_real_or_fake = D(super_resolved_img.detach())

			loss_d0 = adversarial_loss(hr_is_real_or_fake - th.mean(sr_is_real_or_fake), real_labels)
			loss_d1 = adversarial_loss(sr_is_real_or_fake - th.mean(hr_is_real_or_fake), fake_labels)
			loss_d2 = 0.5 * (loss_d0 + loss_d1)

			loss_d2.backward()

			solver_discriminator.step()

			logger.debug(message_fmt % (epoch_counter, nb_epochs, index, loss_d2.item(), loss_g3.item()))
			if index % 100 and gpu_idx == 0:
				I_LR = to_grid(nn.functional.interpolate(low_resolution_img.cpu(), scale_factor=4), nb_rows=1)
				I_SR = to_grid(super_resolved_img.cpu(), nb_rows=1)
				I_HR = to_grid(high_resolution_img.cpu(), nb_rows=1)
				I_LS = th2cv(th.cat((I_LR, I_SR, I_HR), -1)) * 255
				cv2.imwrite(f'storage/img_{epoch_counter:02d}_{index:03d}.jpg', I_LS)
				logger.success('An image was saved...!')
				
		#end for loop ...!

		epoch_counter += 1

	logger.debug('End of training... save the models(G, D)')
	th.save(generator, 'generator.pt')
	th.save(discriminator, 'discriminator.pt')

@click.command()
@click.option('--nb_nodes', help='number of nodes', type=int)
@click.option('--nb_gpus_per_node', help='number of gpus core per nodes', type=int)
@click.option('--node_rank', help='rank of current node', type=int)
@click.option('--source_path', help='path to source data', type=str)
@click.option('--nb_epochs', help='number of epochs during training', type=int)
@click.option('--bt_size', help='size of batched data', type=int)
@click.option('--server_config', help='tcp://address:port', type=str)
def main_loop(nb_nodes, nb_gpus_per_node, node_rank, source_path, nb_epochs, bt_size, server_config):
    if th.cuda.is_available():
        logger.debug('The training mode will be on GPU')
        logger.debug(f'{th.cuda.device_count()} were detected ...!')
        mp.spawn(
            train, 
            nprocs=nb_gpus_per_node,
            args=(node_rank, nb_gpus_per_node, nb_nodes * nb_gpus_per_node, server_config, source_path, nb_epochs, bt_size)
        )
    else:
        logger.debug('No GPU was detected ...!')

if __name__ == '__main__':
	main_loop()
