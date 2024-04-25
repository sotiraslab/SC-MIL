import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=True),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

    def forward(self, x, isNorm=True):
        ## x: N x L
        A = self.attention(x)  ## N x K
        A = torch.transpose(A, 1, 0)  # KxN
        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N
        return A  ### K x N


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        if isNorm:
            A = F.softmax(A, dim=1)  # softmax over N

        return A  ### K x N


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)

    def forward(self, x): ## x: N x L
        AA = self.attention(x)  ## K x N
        afeat = torch.mm(AA, x) ## K x L
        pred = self.classifier(afeat) ## K x num_cls
        return pred


def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps


class DTFDMIL(nn.Module):
    def __init__(self, in_features, m_dim=512, num_cls=2, numGroup=4, total_instance=4, distill='AFS', numLayer_Res=0, droprate=0., sc=None):
        super().__init__()
        self.numGroup = numGroup
        self.total_instance = total_instance
        self.distill = distill

        if sc is None:
            sc = nn.Identity()
            self.m_dim = m_dim
        else:
            sc = sc
            self.m_dim = sc.n_atoms

        self.instance_per_group = self.total_instance // self.numGroup

        self.tier1 = nn.ModuleDict({
            'dimRedunction': DimReduction(in_features, m_dim, numLayer_Res),
            'sc': sc,
            'attention': Attention_Gated(self.m_dim),
            'classifier': Classifier_1fc(self.m_dim, num_cls, droprate)
        })

        self.tier2 = Attention_with_Classifier(self.m_dim, num_cls=num_cls, droprate=droprate)

    def get_param_group(self):
        tier1_params = self.tier1.parameters()
        tier2_params = self.tier2.parameters()

        return tier1_params, tier2_params

    def feature_distallation(self, tmidFeat, sort_idx, tattFeat_tensor):
        ############  Different feature fusion mode ##############
        if self.distill == 'MaxMinS':
            topk_idx_max = sort_idx[:self.instance_per_group].long()
            topk_idx_min = sort_idx[-self.instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx) 
            return MaxMin_inst_feat
        elif self.distill == 'MaxS':
            topk_idx_max = sort_idx[:self.instance_per_group].long()
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            return max_inst_feat
        elif self.distill == 'AFS':
            return tattFeat_tensor

    def forward_train(self, feat):
        device = feat.device
        # instances have already been shuffled in the dataloader
        # no need to shuffle again
        feat_index = list(range(feat.shape[0]))
        index_chunk_list = np.array_split(np.array(feat_index), self.numGroup)
        index_chunk_list = [sst.tolist() for sst in index_chunk_list]

        slide_pseudo_feat = []
        slide_sub_preds = []
        for tindex in index_chunk_list:
            subFeat_tensor = torch.index_select(feat, dim=0, index=torch.LongTensor(tindex).to(device))
            tmidFeat = self.tier1['dimRedunction'](subFeat_tensor)
            tmidFeat = self.tier1['sc'](tmidFeat)
            tAA = self.tier1['attention'](tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = self.tier1['classifier'](tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)
            # print(tPredict.shape)

            patch_pred_logits = get_cam_1d(self.tier1['classifier'], tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

            # topk
            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            slide_pseudo_feat.append(self.feature_distallation(tmidFeat, sort_idx, tattFeat_tensor))

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs
        ############# first tier output #############
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x num_cls
        ### The slide_sub_labels in https://github.com/hrzhang1123/DTFD-MIL/blob/main/Main_DTFD_MIL.py ### 
        ### just repeat the labels at the first dim with the number of psuedo bags (i.e., numGroup x num_cls)
        # print(slide_pseudo_feat.shape)

        ############# second tier output #############
        gSlidePred = self.tier2(slide_pseudo_feat) ### 1 x num_cls

        return slide_sub_preds, gSlidePred

    def forward_inference(self, feat, num_MeanInference=1):
        device = feat.device
        midFeat = self.tier1['dimRedunction'](feat)
        midFeat = self.tier1['sc'](midFeat)
        AA = self.tier1['attention'](midFeat, isNorm=False).squeeze(0)  ## N
        allSlide_pred_softmax = []

        for _ in range(num_MeanInference):
            feat_index = list(range(feat.shape[0]))
            index_chunk_list = np.array_split(np.array(feat_index), self.numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            slide_d_feat = []
            slide_sub_preds = []

            for tindex in index_chunk_list:
                # slide_sub_labels.append(tslideLabel)
                idx_tensor = torch.LongTensor(tindex).to(device)
                tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                tAA = AA.index_select(dim=0, index=idx_tensor)
                tAA = torch.softmax(tAA, dim=0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                tPredict = self.tier1['classifier'](tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                patch_pred_logits = get_cam_1d(self.tier1['classifier'], tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.sigmoid(patch_pred_logits)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                slide_d_feat.append(self.feature_distallation(tmidFeat, sort_idx, tattFeat_tensor))

            slide_d_feat = torch.cat(slide_d_feat, dim=0)
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
            
            gSlidePred = self.tier2(slide_d_feat)
            allSlide_pred_softmax.append(torch.sigmoid(gSlidePred))

        allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
        allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)

        return allSlide_pred_softmax
    
    def forward(self, x, num_MeanInference=1):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_inference(x, num_MeanInference=num_MeanInference)

if __name__ == "__main__":
    x = torch.randn(100, 512)
    model = DTFDMIL(512, 512)
    tier1_params, tier2_params = model.get_param_group()
    # print(model)

    model.train()

    model(x)