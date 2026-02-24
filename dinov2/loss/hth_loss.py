import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

class HTHLoss(nn.Module):
    def __init__(self,
                 do_loss_hierarchical_align=False,
                 do_loss_hash_reg=False,
                 do_loss_fine_align=False,
                 ):
        super().__init__()
        self.do_loss_hierarchical_align = do_loss_hierarchical_align
        self.do_loss_hash_reg = do_loss_hash_reg
        self.do_loss_fine_align = do_loss_fine_align
        # dim: 512 for text embedding of clip-b, 256 for shared space
        self.fc = nn.ModuleList([nn.Linear(512, 256) for _ in range(4)])
        self.fc_word = nn.Linear(512, 256)

    def forward_loss_hierarchical_align(self, u_h, t_e):
        # bs = 64, num_global_crops = 2 * bs = 128
        bs = u_h.shape[0] // 2
        t_h_list = torch.split(t_e, 1, dim=1)  # [64,1,512]
        t_per_h_list = [t.squeeze(1) for t in t_h_list]  # [64,512], [64,512]....
        for i in range(4):
            t_per_h_list[i] = self.fc[i](t_per_h_list[i])

        u_h = u_h[0:bs, :].transpose(0, 1)  # [4, 64, 256]
        loss_total = 0
        for i in range(4):
            u_per_h = F.normalize(u_h[i], p=2, dim=-1)
            t_per_h = F.normalize(t_per_h_list[i], p=2, dim=-1)  # [64, 256]
            sim = torch.matmul(u_per_h, t_per_h.T) / 0.1
            labels = torch.arange(u_per_h.size(0)).cuda()
            loss = F.cross_entropy(sim, labels)
            loss_total += loss
        return loss_total / 4

    def forward_loss_fine_align(self, u_patch_pool, word_embed, cap_lens):
        """
        cap_lens:    [bs] 每条文本有效 token 数
        """
        bs = u_patch_pool.size(0) // 2
        u_patch_pool = u_patch_pool[:bs]
        t = 0.1

        # normalize
        word_embed = F.normalize(self.fc_word(word_embed), dim=-1)  # [bs, 77, 256]
        u_patch_pool = F.normalize(u_patch_pool, dim=-1)  # [bs, 256]

        # transpose to [bs, 256, 77]
        word_embed = word_embed.transpose(1, 2)

        # 计算所有 pairwise patch-to-word
        sim = torch.einsum('bd,jdk->bjk', u_patch_pool, word_embed)
        # 构造 mask，把 padding token 去掉
        device = sim.device
        max_len = word_embed.size(-1)
        mask = torch.arange(max_len, device=device)[None, :] < cap_lens[:, None]
        # mask: [bs, 77]
        # 扩展到 pairwise: [1, bs, 77]
        mask = mask[None, :, :]

        # 把无效 token 设为 0
        sim = sim * mask

        denom = cap_lens[None, :].clamp(min=1)  # 防除0
        sim = sim.sum(-1) / denom

    
        labels = torch.arange(bs, device=device)  # [0..bs-1]
        loss = F.cross_entropy(sim / t, labels)

        return loss

    def forward(self, u_h, u_fused, text_embeddings, word_embeddings, cap_lens, u_patch_pool):
        hth_loss_dict = {}
        if self.do_loss_hierarchical_align:
            loss_hierarchical_align = self.forward_loss_hierarchical_align(u_h, text_embeddings)
            hth_loss_dict['loss_hierarchical_align'] = loss_hierarchical_align
        if self.do_loss_hash_reg:
            loss_hash_reg = (u_fused.abs() - 1).abs().mean()
            hth_loss_dict['loss_hash_reg'] = loss_hash_reg
        if self.do_loss_fine_align:
            loss_fine_align = self.forward_loss_fine_align(u_patch_pool, word_embeddings, cap_lens)
            hth_loss_dict['loss_fine_align'] = loss_fine_align
        return hth_loss_dict

if __name__ == '__main__':
    u_patch_pool = torch.randn(128, 256)
    word_embeddings = torch.randn(64, 77, 512)
    cap_lens = torch.randint(low=1, high=77, size=([64]))
    l = HTHLoss().forward_loss_fine_align(u_patch_pool, word_embeddings, cap_lens)




