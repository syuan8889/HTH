import torch
from retrieval import misc
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


@torch.no_grad()
def evaluate(data_loader, model, device, topK=-1, query_num=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    # model.train()
    bs, clses = [], []
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 100, header):
            images = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)
            # compute output
            with torch.amp.autocast('cuda'):
                output = model(images)

            output = output.sign()
            clses.append(target.cpu())
            bs.append(output.cpu())

    
            del images, target, output
         
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tst_binary = torch.cat(bs).to(device)
    tst_label = torch.cat(clses).to(device)

    tst_binary = tst_binary.sign()
    mAP, ind_mtx = CalcTopMap_CUDA(tst_binary, tst_binary[:query_num], tst_label, tst_label[:query_num], topK)

    return mAP


# def CalcHammingDist_CUDA(B1, B2):
#     q = B2.shape[1]
#     distH = 0.5 * (q - torch.matmul(B1, B2.t()))
#     return distH

def CalcL2Dist_CUDA(d1, d2, batch_size=1024):
    """
    Calculate the L2 distance between two matrices using CUDA with batch processing.

    Parameters:
    - d1: torch.Tensor, shape (N, D)
    - d2: torch.Tensor, shape (M, D)
    - batch_size: int, batch size for processing

    Returns:
    - distances: torch.Tensor, shape (N, M)
    """
    N, D = d1.shape
    M = d2.shape[0]

    distances = torch.empty((N, M), device=d1.device)

    for i in range(0, N, batch_size):
        d1_batch = d1[i:i + batch_size]
        for j in range(0, M, batch_size):
            d2_batch = d2[j:j + batch_size]
            diff = d1_batch.unsqueeze(1) - d2_batch.unsqueeze(0)
            dist_batch = torch.sqrt(torch.sum(diff ** 2, dim=-1))
            distances[i:i + batch_size, j:j + batch_size] = dist_batch

    return distances

def CalcTopMap_CUDA(rB, qB, retrievalL, queryL, topk, ret_mtx=False, same_train_test=False):
    num_query = queryL.shape[0]
    dist_mtx = CalcL2Dist_CUDA(qB, rB, batch_size=256).cpu()
    queryL = queryL.float()
    retrievalL = retrievalL.float()
    tmp = torch.matmul(queryL, retrievalL.t()).cpu()
    queryL = queryL.cpu()
    retrievalL = retrievalL.cpu()
    gnd_mtx = tmp.gt(0).float()

    _, ind_mtx = torch.sort(dist_mtx, dim=1,descending=False)
    topkmap = 0
    for iter in tqdm(range(num_query)):
        ind = ind_mtx[iter, :]
        gnd = gnd_mtx[iter, ind]

        if same_train_test:
            gnd[iter] = 0

        tgnd = gnd[0:topk]
        tsum = torch.sum(tgnd).int()
        if tsum.item() == 0:
            continue

        count = torch.linspace(1, end=tsum.item(), steps=tsum.item()) 
        tindex = torch.nonzero(tgnd.eq(1)).add(1.0).squeeze()

        count = count.to(queryL.device)
        tindex = tindex.to(queryL.device)

        topkmap_ = torch.mean(torch.div(count, tindex))
        topkmap = topkmap + topkmap_.item()
    topkmap = topkmap / num_query
    if not ret_mtx:
        return topkmap, ind_mtx
    else:
        return topkmap, dist_mtx, gnd_mtx, ind_mtx