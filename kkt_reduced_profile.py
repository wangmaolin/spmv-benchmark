import time
import torch
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate
import argparse

from app_pat_gen import huber_fitting_pat, lasso_pat, svm_pat

def scipycsc_to_torchcsr(scipy_csc, device):
	scipy_coo = coo_matrix(scipy_csc.todense())
	indices = np.vstack((scipy_coo.row, scipy_coo.col))
	i = torch.LongTensor(indices)
	v = torch.FloatTensor(scipy_coo.data)
	torch_coo = torch.sparse.FloatTensor(i, v, torch.Size(scipy_coo.shape)).to(device)
	torch_csr = torch_coo.to_sparse_csr()
	return torch_csr

def profile_spmv(mat_csc, iter_steps, device):
	torch_csr = scipycsc_to_torchcsr(mat_csc, device)
	N = mat_csc.shape[1]

	x_batch = []
	for i in range(iter_steps):
		x_tensor = torch.rand(N, 1, dtype=torch.float32, device=device)
		x_batch.append(x_tensor)

	# torch.cuda.synchronize()
	tic = time.perf_counter()

	for i in range(iter_steps):
		result = torch.sparse.mm(torch_csr, x_batch[i])

	# torch.cuda.synchronize()
	toc = time.perf_counter()

	return toc - tic
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--pattern', type=str, default='lasso')
	parser.add_argument('-m', '--matrix', type=str, default='A')
	parser.add_argument('-d', '--device', type=str, default='cpu')
	parser.add_argument('-i', '--iteration', type=int, default=10)
	args = parser.parse_args()
	if args.pattern == 'lasso':
		A_csc, P_csc = lasso_pat(n=12, m=1000)
	elif args.pattern == 'svm':
		A_csc, P_csc = svm_pat(n=16, m=1024)
	elif args.pattern == 'huber':
		A_csc, P_csc = huber_fitting_pat(n=16, m=256)

	if args.matrix== 'A':
		mat_csc = A_csc.astype(np.float32)
	elif args.matrix== 'P':
		mat_csc = P_csc.astype(np.float32)

	device = args.device
	iter_steps = args.iteration

	""" Warm up CUDA context of torch"""
	profile_spmv(mat_csc, 1, device)

	run_time = profile_spmv(mat_csc, iter_steps, device)

	mat_info = [[mat_csc.shape,
				len(mat_csc.data)*100.0/(mat_csc.shape[0]*mat_csc.shape[1]),
				len(mat_csc.data),
				device,
				iter_steps, 
				run_time * 1000]]

	print(tabulate(mat_info, headers=["shape", "density", "nnz", "device", "iter", "time(ms)"]))

if __name__ == '__main__':
    main()