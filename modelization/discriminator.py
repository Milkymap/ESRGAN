import torch as th 
import torch.nn as nn 

import numpy as np 
import operator as op 
import functools as ft, itertools as it 


class Discriminator(nn.Module):
	def __init__(self, i_channels=3, o_channels=64, num_super_blocks=4, num_neurons=1024):
		super(Discriminator, self).__init__()
		self.shapes = np.repeat([ o_channels * 2 ** idx for idx in range(num_super_blocks)], 2)
		self.shapes = list(zip(self.shapes[:-1], self.shapes[1:]))
		self.head = nn.Sequential(
			nn.Conv2d(i_channels, o_channels, 3, 1, 1),
			nn.LeakyReLU(0.2, inplace=True)
		)
		self.body = []
		for idx, (m,n) in enumerate(self.shapes):
			s = 2 if idx % 2 == 0 else 1  
			self.body.append(nn.Conv2d(m, n, 3, s, 1))
			self.body.append(nn.BatchNorm2d(n))
			self.body.append(nn.LeakyReLU(0.2, inplace=True))
		self.body = nn.Sequential(*self.body)
		self.tail = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(o_channels * 2 ** (num_super_blocks - 1), num_neurons, 1),
			nn.Conv2d(num_neurons, 1, 1),
			nn.Flatten()
		)

	def forward(self, X0):
		return self.tail(self.body(self.head(X0))) 


if __name__ == '__main__':
	D = Discriminator()
	print(D)
	X = th.randn((3, 3, 256, 256))
	Y = D(X)
	print(Y)