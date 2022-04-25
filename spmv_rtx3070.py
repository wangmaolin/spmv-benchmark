import time
import torch
import argparse
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def profile_spmv(N, acc_pack, device, batch_size, format):
    A_batch = []
    x_batch = []

    """ The masks are the same for all matrices"""
    nnz_data = N * 16 * acc_pack
    A_cols = []
    for i in range(N):
        row_sample = np.sort(np.random.choice(N, size=16 * acc_pack, replace=False))
        A_cols = np.append(A_cols, row_sample)
    A_crows = np.arange(N+1)*16*acc_pack

    for i in range(batch_size):

        A_data = np.random.random(nnz_data).astype(np.float32)
        scipy_csr = csr_matrix((A_data, A_cols, A_crows), shape=[N, N])
        scipy_coo = coo_matrix(scipy_csr.todense())

        values = scipy_coo.data
        indices = np.vstack((scipy_coo.row, scipy_coo.col))
        i = torch.LongTensor(indices)
        shape = scipy_coo.shape
        v = torch.FloatTensor(values)
        torch_coo = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)

        if format =='coo':
            A_batch.append(torch_coo)
        elif format =='csr':
            torch_csr = torch_coo.to_sparse_csr()
            A_batch.append(torch_csr)

        x_tensor = torch.rand(N, 1, dtype=torch.float32, device=device)
        x_batch.append(x_tensor)

    torch.cuda.synchronize()
    tic = time.perf_counter()

    for i in range(batch_size):
        result = torch.sparse.mm(A_batch[i], x_batch[i])

    torch.cuda.synchronize()
    toc = time.perf_counter()

    return toc - tic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-r', '--repeat', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-f', '--format', type=str, default='csr')
    args = parser.parse_args()

    result_table = np.zeros((4,4))
    for idx_n, N in enumerate([4096]):
        # for idx_d, acc_pack in enumerate([1, 2, 4, 8]):
        for idx_d, acc_pack in enumerate([16, 32, 64]):
            """ warm up cuda context setup """ 
            profile_spmv(N=N, 
                         acc_pack=acc_pack, 
                         device=args.device, 
                         batch_size=1,
                         format=args.format)

            stats = np.zeros(args.repeat)
            for r in range(args.repeat):
                stats[r] = profile_spmv(N=N, 
                                        acc_pack=acc_pack, 
                                        device=args.device, 
                                        batch_size=args.batch_size,
                                        format=args.format)

            average_time = np.average(stats) * 1000
            print(f"Device={args.device} Repeat={args.repeat} N={N} acc_pack={acc_pack} Time {average_time:0.4f} ms")

            result_table[idx_n, idx_d] = average_time

    # np.savetxt(args.device+'-'+args.precision+'.csv', result_table, fmt='%3.6f', delimiter=',')

if __name__ == '__main__':
    main()
