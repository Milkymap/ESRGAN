import torch as th 
import torch.nn as nn 


import operator as op 
import functools as ft, itertools as it 

class RDB(nn.Module):
	# residual dense block 
	def __init__(self, filters, residual_scale, num_layers):
		super(RDB, self).__init__() 
		def make_body(num_layers, i_channels, o_channels):
			accumulator = []
			for idx in range(num_layers - 1):
				head = nn.Sequential(
					nn.Conv2d((idx + 1) * i_channels, o_channels, 3, 1, 1),
					nn.LeakyReLU()
				)
				accumulator.append(head)	
			return accumulator

		self.residual_scale = residual_scale
		self.body = nn.ModuleList(make_body(num_layers, filters, filters))
		self.tail = nn.Conv2d(num_layers * filters, filters, 3, 1, 1)

	
	def forward(self, X0):
		XN = ft.reduce(
			lambda X_i, F_i: th.cat([X_i, F_i(X_i)], dim=1), 
			self.body, 
			X0 
		)
		return X0 + self.residual_scale * self.tail(XN) 

class RIRDB(nn.Module):
	# residual in residual dense block  
	def __init__(self, residual_scale, num_rdb):
		super(RIRDB, self).__init__()
		self.residual_scale = residual_scale
		self.body = nn.Sequential(*[
			RDB(filters=64, residual_scale=0.2, num_layers=5)
			for idx in range(num_rdb)
		])

	def forward(self, X0):
		XN = self.body(X0)
		return X0 + self.residual_scale * XN  


class UPSCB(nn.Module):
	# up_scale block : 4C,H,W => C, 2H, 2W
	def __init__(self, i_channels, upscale_factor=2):
		super(UPSCB, self).__init__()
		self.head = nn.Conv2d(i_channels, upscale_factor ** 2 * i_channels, 3, 1, 1)
		self.body = nn.PixelShuffle(upscale_factor=upscale_factor)
		self.tail = nn.LeakyReLU()

	def forward(self, X0):
		return self.tail(self.body(self.head(X0)))


class Generator(nn.Module):
	# I_LR => Head => Body => botl + Head(I_LR) => tail => term => I_SR 
	def __init__(self, i_channels, o_channels, num_blocks, num_scaling_block=2):
		super(Generator, self).__init__()
		self.head = nn.Conv2d(i_channels, o_channels, 3, 1, 1)
		self.body = nn.Sequential(*[ RIRDB(residual_scale=0.2, num_rdb=3) for idx in range(num_blocks)]) 
		self.botl = nn.Conv2d(o_channels, o_channels, 3, 1, 1)
		self.tail = nn.Sequential(*[ UPSCB(o_channels) for idx in range(num_scaling_block)])
		self.term = nn.Sequential(
			nn.Conv2d(o_channels, i_channels, 3, 1, 1),
			nn.Tanh()
		)

	def forward(self, X0):
		X1 = self.head(X0)
		X2 = self.body(X1)
		X3 = self.botl(X2)
		X4 = self.tail(X1 + X3)
		X5 = self.term(X4)
		return X5 

if __name__ == '__main__':
	G = Generator(i_channels=3, o_channels=64, num_blocks=8, num_scaling_block=3)
	print(G)
	X = th.randn((3, 3, 64, 64))
	Y = G(X)
	print(Y.shape)
