import torch
from torch.nn import Linear
from .Strategy import Strategy
from openke.module.loss import MarginLoss
from openke.module.model.LinkPredictor import LinkPredictor
import torch.nn.functional as F
import pdb

class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, sf_loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.sf_loss = sf_loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		self.lin = Linear(2,1)

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		score = self.model(data)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		#pdb.set_trace()
		return loss_res
	
	def predict(self, data):
		score_score = self.model.forward(data).unsqueeze(dim = 1)
		#pdb.set_trace()
		return score.cpu().data.numpy()
