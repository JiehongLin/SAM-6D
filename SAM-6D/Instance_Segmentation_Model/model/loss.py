from torch import nn
import torch
from utils.poses.pose_utils import load_rotation_transform, convert_openCV_to_openGL_torch
import torch.nn.functional as F
from model.utils import BatchedData


class Similarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(Similarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query, reference):
        query = F.normalize(query, dim=-1)
        reference = F.normalize(reference, dim=-1)
        similarity = F.cosine_similarity(query, reference, dim=-1)
        return similarity.clamp(min=0.0, max=1.0)


class PairwiseSimilarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(PairwiseSimilarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query, reference):
        N_query = query.shape[0]
        N_objects, N_templates = reference.shape[0], reference.shape[1]
        references = reference.clone().unsqueeze(0).repeat(N_query, 1, 1, 1)
        queries = query.clone().unsqueeze(1).repeat(1, N_templates, 1)
        queries = F.normalize(queries, dim=-1)
        references = F.normalize(references, dim=-1)

        similarity = BatchedData(batch_size=None)
        for idx_obj in range(N_objects):
            sim = F.cosine_similarity(
                queries, references[:, idx_obj], dim=-1
            )  # N_query x N_templates
            similarity.append(sim)
        similarity.stack()
        similarity = similarity.data
        similarity = similarity.permute(1, 0, 2)  # N_query x N_objects x N_templates
        return similarity.clamp(min=0.0, max=1.0)

class MaskedPatch_MatrixSimilarity(nn.Module):
    def __init__(self, metric="cosine", chunk_size=64):
        super(MaskedPatch_MatrixSimilarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def compute_straight(self, query, reference):
        (N_query, N_patch, N_features) = query.shape 
        sim_matrix = torch.matmul(query, reference.permute(0, 2, 1)) # N_query x N_query_mask x N_refer_mask

        # N2_ref score max
        max_ref_patch_score = torch.max(sim_matrix, dim=-1).values
        # N1_query score average
        factor = torch.count_nonzero(query.sum(dim=-1), dim=-1) + 1e-6
        scores = torch.sum(max_ref_patch_score, dim=-1) / factor # N_query x N_objects x N_templates

        return scores.clamp(min=0.0, max=1.0)

    def compute_visible_ratio(self, query, reference, thred=0.5):

        sim_matrix = torch.matmul(query, reference.permute(0, 2, 1)) # N_query x N_query_mask x N_refer_mask
        sim_matrix = sim_matrix.max(1)[0] # N_query x N_refer_mask
        valid_patches = torch.count_nonzero(sim_matrix, dim=(1, )) + 1e-6

        # fliter correspendence by thred
        flitered_matrix = sim_matrix * (sim_matrix > thred)
        sim_patches = torch.count_nonzero(flitered_matrix, dim=(1,))

        visible_ratio = sim_patches / valid_patches

        return visible_ratio

    def compute_similarity(self, query, reference):
        # all template computation
        N_query = query.shape[0]
        N_objects, N_templates = reference.shape[0], reference.shape[1]
        references = reference.unsqueeze(0).repeat(N_query, 1, 1, 1, 1)
        queries = query.unsqueeze(1).repeat(1, N_templates, 1, 1)

        similarity = BatchedData(batch_size=None)
        for idx_obj in range(N_objects):
            sim_matrix =  torch.matmul(queries, references[:, idx_obj].permute(0, 1, 3, 2)) # N_query x N_templates x N_query_mask x N_refer_mask
            similarity.append(sim_matrix)
        similarity.stack()
        similarity = similarity.data
        similarity = similarity.permute(1, 0, 2, 3, 4)  # N_query x N_objects x N_templates x N1_query x N2_ref

        # N2_ref score max
        max_ref_patch_score = torch.max(similarity, dim=-1).values
        # N1_query score average
        factor = torch.count_nonzero(query.sum(dim=-1), dim=-1)[:, None, None]
        scores = torch.sum(max_ref_patch_score, dim=-1) / factor # N_query x N_objects x N_templates

        return scores.clamp(min=0.0, max=1.0)
    
    def forward_by_chunk(self, query, reference):
        # divide by N_query
        batch_query = BatchedData(batch_size=self.chunk_size, data=query)
        del query
        scores = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_query)):
            score = self.compute_similarity(batch_query[idx_batch], reference)
            scores.cat(score)
        return scores.data
    
    def forward(self, qurey, reference):
        if qurey.shape[0] > self.chunk_size:
            scores = self.forward_by_chunk(qurey, reference)
        else:
            scores = self.compute_similarity(qurey, reference)
        return scores
