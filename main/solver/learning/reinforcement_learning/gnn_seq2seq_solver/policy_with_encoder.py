# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_dense_batch
from main.solver.learning.neural_network import GCNConvNet, DeepEdgeFeatureGAT, ResNetBlock, MLPNet, get_gnn_class
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, \
    MultiheadAttention


class GnnSeq2SeqActorCritic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, gnn_type='gcn'):
        super(GnnSeq2SeqActorCritic, self).__init__()
        self.encoder = Encoder(v_net_x_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob,
                               batch_norm=batch_norm, gnn_type=gnn_type)
        self.actor = Actor(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim,
                           dropout_prob=dropout_prob, batch_norm=batch_norm, gnn_type=gnn_type)
        self.critic = Critic(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim,
                             dropout_prob=dropout_prob, batch_norm=batch_norm, gnn_type=gnn_type)
        self._last_hidden_state = None

    def encode(self, obs):
        x = obs['v_net_x']
        outputs, hidden_state = self.encoder(x)
        self._last_hidden_state = hidden_state
        return outputs

    def act(self, obs):
        logits, outputs, hidden_state = self.actor(obs)
        self._last_hidden_state = hidden_state
        return logits

    def evaluate(self, obs):
        value = self.critic(obs)
        return value

    def get_last_rnn_state(self):
        return self._last_hidden_state

    def set_last_rnn_hidden(self, hidden_state):
        self._last_hidden_state = hidden_state


class GcnSeq2SeqActorCritic(GnnSeq2SeqActorCritic):

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, **kwargs):
        super(GcnSeq2SeqActorCritic, self).__init__(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim,
                                                    embedding_dim=embedding_dim, dropout_prob=dropout_prob,
                                                    batch_norm=batch_norm, gnn_type='gcn')


class GATSeq2SeqActorCritic(GnnSeq2SeqActorCritic):
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, **kwargs):
        super(GATSeq2SeqActorCritic, self).__init__(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim,
                                                    embedding_dim=embedding_dim, dropout_prob=dropout_prob,
                                                    batch_norm=batch_norm, gnn_type='gat')


class GATSeq2SeqNoGATActorCritic(GnnSeq2SeqActorCritic):
    """Ablation variant: w/o M-GAT.

    Lightweight fusion ablation. It no longer uses the PPO-MGT Seq2Seq
    Transformer decoder. The physical network side removes graph attention and
    maps each physical node independently with an MLP, while the SFC side keeps
    the Transformer encoder. The final action logits are produced by an MLP
    fusion of physical-node embeddings and the current VNF context.
    """

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, **kwargs):
        AblationFusionActorCritic.__init__(
            self,
            p_net_num_nodes,
            p_net_x_dim,
            p_net_edge_dim,
            v_net_x_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            use_gat=False,
            use_transformer=True,
            num_gnn_layers=int(kwargs.get('num_gnn_layers', 3)),
        )


class GATSeq2SeqNoTransformerActorCritic(GnnSeq2SeqActorCritic):
    """Ablation variant: w/o Transformer.

    Lightweight fusion ablation. It keeps M-GAT for physical-network encoding,
    removes the Transformer SFC encoder/decoder, and uses only a linear mapping
    of the current VNF features. The final action logits are produced by an MLP
    fusion module.
    """

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, **kwargs):
        AblationFusionActorCritic.__init__(
            self,
            p_net_num_nodes,
            p_net_x_dim,
            p_net_edge_dim,
            v_net_x_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            use_gat=True,
            use_transformer=False,
            num_gnn_layers=int(kwargs.get('num_gnn_layers', 3)),
        )


class GCNSeq2SeqNoTransformerActorCritic(GnnSeq2SeqActorCritic):
    """Ablation variant: w/o M-GAT and w/o Transformer.

    Lightweight fusion ablation. It removes both key modules: the physical side
    keeps only local node-feature MLP encoding, and the SFC side keeps only the
    current VNF linear feature encoding. The class name is kept for backward
    compatibility with old runs.
    """

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, **kwargs):
        AblationFusionActorCritic.__init__(
            self,
            p_net_num_nodes,
            p_net_x_dim,
            p_net_edge_dim,
            v_net_x_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            use_gat=False,
            use_transformer=False,
            num_gnn_layers=1,
        )


class AblationFusionActorCritic(nn.Module):
    """Lightweight ablation policy used by the three PPO-MGT variants.

    This class intentionally avoids the original PPO-MGT Transformer decoder.
    It keeps a common fusion-style decision head so that the three ablations
    differ only in whether the physical-network M-GAT encoder and the SFC
    Transformer encoder are used.
    """

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, use_gat=True, use_transformer=True, num_gnn_layers=3):
        nn.Module.__init__(self)
        self.encoder = Encoder(v_net_x_dim, embedding_dim=embedding_dim, dropout_prob=dropout_prob) \
            if use_transformer else EncoderNoTransformer(v_net_x_dim, embedding_dim=embedding_dim,
                                                         dropout_prob=dropout_prob)
        self.actor = FusionActor(
            p_net_num_nodes,
            p_net_x_dim,
            p_net_edge_dim,
            embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            use_gat=use_gat,
            num_gnn_layers=num_gnn_layers,
        )
        self.critic = FusionCritic(
            p_net_num_nodes,
            p_net_x_dim,
            p_net_edge_dim,
            embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            use_gat=use_gat,
            num_gnn_layers=num_gnn_layers,
        )
        self._last_hidden_state = None


class FusionActor(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, use_gat=True, num_gnn_layers=3):
        super().__init__()
        self.decoder = FusionDecoder(
            p_net_num_nodes,
            p_net_x_dim,
            p_net_edge_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            use_gat=use_gat,
            num_gnn_layers=num_gnn_layers,
            output_value=False,
        )

    def forward(self, obs):
        logits, outputs, hidden_state = self.decoder(obs)
        return logits, outputs, hidden_state


class FusionCritic(nn.Module):
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, use_gat=True, num_gnn_layers=3):
        super().__init__()
        self.decoder = FusionDecoder(
            p_net_num_nodes,
            p_net_x_dim,
            p_net_edge_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            use_gat=use_gat,
            num_gnn_layers=num_gnn_layers,
            output_value=True,
        )

    def forward(self, obs):
        value, outputs, hidden_state = self.decoder(obs)
        return value


class FusionDecoder(nn.Module):
    """MLP fusion decoder for ablation policies.

    For each physical node, it concatenates the physical-node embedding and the
    current VNF context, then predicts the action logit with an MLP.  No
    Transformer decoder, recurrent decoder state, or cross-attention is used.
    """

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim=None, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, use_gat=True, num_gnn_layers=3, output_value=False):
        super().__init__()
        self.output_value = output_value
        self.p_net_num_nodes = max(p_net_num_nodes, 100)
        PNetEncoder = get_gnn_class('gat') if use_gat else BasicPNetFeatureEncoder
        self.p_net_encoder = PNetEncoder(
            p_net_x_dim,
            embedding_dim,
            edge_dim=p_net_edge_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            return_batch=True,
            num_layers=num_gnn_layers,
        )
        # The original PPO-MGT decoder conditions the current decision on the
        # previously selected physical node.  The first lightweight fusion
        # version accidentally ignored p_node_id, which made the ablation
        # policies almost memoryless and caused many routing failures.  Keep a
        # cheap learned previous-node context here: it is not a Transformer/GAT
        # module, but it restores the necessary placement-continuity signal.
        self.prev_p_node_embedding = nn.Embedding(self.p_net_num_nodes + 1, embedding_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.Linear(embedding_dim, 1),
        )

    def _get_curr_v_context(self, obs, batch_size):
        encoder_outputs = obs['encoder_outputs']  # [batch, seq_len, embedding_dim]
        curr_v_node_id = obs.get('curr_v_node_id', None)
        if curr_v_node_id is None:
            return encoder_outputs[:, -1, :]
        curr_v_node_id = curr_v_node_id.long().view(-1)
        seq_len = encoder_outputs.size(1)
        curr_v_node_id = torch.clamp(curr_v_node_id, min=0, max=seq_len - 1)
        batch_indices = torch.arange(batch_size, device=encoder_outputs.device)
        return encoder_outputs[batch_indices, curr_v_node_id, :]

    def _get_prev_p_context(self, obs, batch_size, device):
        p_node_id = obs.get('p_node_id', None)
        if p_node_id is None:
            p_node_id = torch.full((batch_size,), self.p_net_num_nodes, dtype=torch.long, device=device)
        else:
            p_node_id = p_node_id.long().view(-1).to(device)
            if p_node_id.numel() == 1 and batch_size > 1:
                p_node_id = p_node_id.expand(batch_size)
            elif p_node_id.numel() != batch_size:
                p_node_id = p_node_id[-batch_size:]
        p_node_id = torch.clamp(p_node_id, min=0, max=self.p_net_num_nodes)
        return self.prev_p_node_embedding(p_node_id)

    def forward(self, obs):
        p_node_embeddings = self.p_net_encoder(obs['p_net'])  # [batch, num_nodes, embedding_dim]
        batch_size, num_nodes, _ = p_node_embeddings.size()
        curr_v_context = self._get_curr_v_context(obs, batch_size)
        curr_v_context = curr_v_context.unsqueeze(1).expand(-1, num_nodes, -1)
        prev_p_context = self._get_prev_p_context(obs, batch_size, p_node_embeddings.device)
        prev_p_context = prev_p_context.unsqueeze(1).expand(-1, num_nodes, -1)
        fused = torch.cat([p_node_embeddings, curr_v_context, prev_p_context], dim=-1)
        node_scores = self.fusion_mlp(fused).squeeze(-1)
        hidden_state = (curr_v_context[:, :1, :] + prev_p_context[:, :1, :]).permute(1, 0, 2).detach()
        outputs = hidden_state
        if self.output_value:
            return node_scores.mean(dim=-1, keepdim=True), outputs, hidden_state
        return node_scores, outputs, hidden_state


class DeepEdgeGnnSeq2SeqActorCritic(GnnSeq2SeqActorCritic):
    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, **kwargs):
        super(DeepEdgeGnnSeq2SeqActorCritic, self).__init__(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim,
                                                            embedding_dim=embedding_dim, dropout_prob=dropout_prob,
                                                            batch_norm=batch_norm, gnn_type='deep_edge_gat')


class Actor(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, gnn_type='gcn'):
        super(Actor, self).__init__()
        self.decoder = Decoder(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, embedding_dim=embedding_dim,
                               dropout_prob=dropout_prob, batch_norm=batch_norm, gnn_type=gnn_type)

    def forward(self, obs):
        """Return logits of actions"""
        logits, outputs, hidden_state = self.decoder(obs)
        return logits, outputs, hidden_state


class Critic(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, v_net_x_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, gnn_type='gcn'):
        super(Critic, self).__init__()
        self.decoder = Decoder(p_net_num_nodes, p_net_x_dim, p_net_edge_dim, embedding_dim=embedding_dim,
                               dropout_prob=dropout_prob, batch_norm=batch_norm, gnn_type=gnn_type)

    def forward(self, obs):
        """Return logits of actions"""
        logits, outputs, hidden_state = self.decoder(obs)
        value = torch.mean(logits, dim=-1, keepdim=True)
        return value


class ActorNoTransformer(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, gnn_type='gat', num_gnn_layers=3):
        super(ActorNoTransformer, self).__init__()
        self.decoder = DecoderNoTransformer(p_net_num_nodes, p_net_x_dim, p_net_edge_dim,
                                            embedding_dim=embedding_dim,
                                            dropout_prob=dropout_prob,
                                            batch_norm=batch_norm,
                                            gnn_type=gnn_type,
                                            num_gnn_layers=num_gnn_layers)

    def forward(self, obs):
        logits, outputs, hidden_state = self.decoder(obs)
        return logits, outputs, hidden_state


class CriticNoTransformer(nn.Module):

    def __init__(self, p_net_num_nodes, p_net_x_dim, p_net_edge_dim, embedding_dim=128, dropout_prob=0.,
                 batch_norm=False, gnn_type='gat', num_gnn_layers=3):
        super(CriticNoTransformer, self).__init__()
        self.decoder = DecoderNoTransformer(p_net_num_nodes, p_net_x_dim, p_net_edge_dim,
                                            embedding_dim=embedding_dim,
                                            dropout_prob=dropout_prob,
                                            batch_norm=batch_norm,
                                            gnn_type=gnn_type,
                                            num_gnn_layers=num_gnn_layers)

    def forward(self, obs):
        logits, outputs, hidden_state = self.decoder(obs)
        value = torch.mean(logits, dim=-1, keepdim=True)
        return value


# class Encoder(nn.Module):
#
#     def __init__(self, v_net_x_dim, embedding_dim=128, dropout_prob=0., batch_norm=False, gnn_type='gcn'):
#         super(Encoder, self).__init__()
#         self.emb = nn.Linear(v_net_x_dim, embedding_dim)
#         self.gru = nn.GRU(embedding_dim, embedding_dim)
#
#     def forward(self, x):
#         x = x.permute(1, 0, 2)
#         embeddings = F.relu(self.emb(x))
#         outputs, hidden_state = self.gru(embeddings)
#         a = outputs.shape
#         b = hidden_state.shape
#         return outputs, hidden_state


# class Decoder(nn.Module):
#
#     def __init__(self, p_net_num_nodes, feature_dim, edge_dim=None, embedding_dim=128, dropout_prob=0.,
#                  batch_norm=False, gnn_type='gcn'):
#         super(Decoder, self).__init__()
#         if p_net_num_nodes <= 100:
#             self.p_net_num_nodes = 100
#         else:
#             self.p_net_num_nodes = p_net_num_nodes
#         self.emb = nn.Embedding(p_net_num_nodes + 1, embedding_dim)
#         self.att = Attention(embedding_dim)
#         GnnNet = get_gnn_class(gnn_type)
#         self.gcn = GnnNet(feature_dim, embedding_dim, edge_dim=edge_dim, embedding_dim=embedding_dim,
#                           dropout_prob=dropout_prob, batch_norm=batch_norm, return_batch=True)
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim, 1),
#             nn.Flatten()
#         )
#         self.gru = nn.GRU(embedding_dim, embedding_dim)
#         self._last_hidden_state = None
#
#     def forward(self, obs):
#         batch_p_net = obs['p_net']
#         hidden_state = obs['hidden_state']
#         p_node_embeddings = self.gcn(batch_p_net)
#         p_node_embeddings = p_node_embeddings.reshape(batch_p_net.num_graphs, -1, p_node_embeddings.shape[-1])
#         p_node_embeddings = p_node_embeddings + hidden_state
#         logits = self.mlp(p_node_embeddings)
#         p_node_id = obs['p_node_id']
#         hidden_state = hidden_state.permute(1, 0, 2)
#         encoder_outputs = obs['encoder_outputs']
#         mask = obs['mask']
#         # p_node_id is in batch model [batch_size, p_net_num_nodes]
#         # if any p_node_id large than 100, please use 100 as the index
#         p_node_id = torch.clamp(p_node_id, max=self.p_net_num_nodes)
#         p_node_emb = self.emb(p_node_id).unsqueeze(0)
#         context, attention = self.att(hidden_state, encoder_outputs, mask)
#         context = context.unsqueeze(0)
#         # TODO 铻嶅叆涓婁笅鏂囧悜閲?
#         p_node_emb = p_node_emb + context
#         outputs, hidden_state = self.gru(p_node_emb, hidden_state)
#
#         return logits, outputs, hidden_state


class BasicPNetFeatureEncoder(nn.Module):
    """Physical-node encoder without GAT/GCN topology aggregation.

    It independently maps each physical node feature vector into the embedding
    space, so it keeps only basic node attributes and intentionally discards
    edge/topology message passing for the w/o M-GAT ablation.
    """

    def __init__(self, input_dim, output_dim, dropout_prob=0.1, return_batch=True, **kwargs):
        super().__init__()
        self.return_batch = return_batch
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, input):
        x = self.net(input['x'])
        if self.return_batch:
            x, _ = to_dense_batch(x, input.batch)
        return x


class EncoderNoTransformer(nn.Module):
    """Simple SFC sequence encoder without Transformer.

    It keeps only linear feature embedding; no positional encoding,
    no self-attention, no Transformer layer, and no GRU recurrence is used.
    """

    def __init__(self, v_net_x_dim, embedding_dim=128, dropout_prob=0.1, max_len=100):
        super().__init__()
        self.embedding = nn.Linear(v_net_x_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x):
        # x: [batch_size, seq_len, v_net_x_dim]
        embeddings = F.relu(self.embedding(x))
        embeddings = self.dropout(embeddings)
        outputs = embeddings.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        hidden_state = outputs[-1:, :, :]      # [1, batch_size, embedding_dim]
        return outputs, hidden_state


class Encoder(nn.Module):
    def __init__(self, v_net_x_dim, embedding_dim=128, dropout_prob=0.1, batch_norm=False, gnn_type='gcn', max_len=100):
        super().__init__()
        self.embedding = nn.Linear(v_net_x_dim, embedding_dim)
        # TODO
        self.special_token_embedding = nn.Embedding(2, embedding_dim)  # 0: start, 1: end
        # 鍙涔犵殑浣嶇疆缂栫爜
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dropout=dropout_prob,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

    def forward(self, x):
        # x: [batch_size, seq_len, v_net_x_dim]
        batch_size, seq_len, _ = x.size()
        embeddings = F.relu(self.embedding(x))  # [batch_size, seq_len, embedding_dim]
        embeddings = embeddings.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        # TODO
        # 娣诲姞鐗规畩 token embedding锛堣捣鐐?缁堢偣锛?
        start_vnf_index = 0
        end_vnf_index = seq_len-1
        special_flags = torch.zeros(seq_len, batch_size, embeddings.size(-1), device=x.device)
        special_flags[start_vnf_index, :, :] = self.special_token_embedding(torch.tensor(0, device=x.device))
        special_flags[end_vnf_index, :, :] = self.special_token_embedding(torch.tensor(1, device=x.device))
        embeddings = embeddings + special_flags

        # 娣诲姞浣嶇疆缂栫爜
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(seq_len, batch_size)  # [seq_len, batch_size]
        position_encodings = self.position_embedding(positions)  # [seq_len, batch_size, embedding_dim]
        embeddings = embeddings + position_encodings  # 鍔犱笂浣嶇疆缂栫爜
        transformer_outputs = self.transformer_encoder(embeddings)  # [seq_len, batch_size, embedding_dim]
        hidden_state = transformer_outputs[-1, :, :].unsqueeze(0)   # [1, batch_size, embedding_dim]
        return transformer_outputs, hidden_state


class DecoderNoTransformer(nn.Module):
    """Ablation decoder without Transformer.

    It keeps the selected physical-network encoder and uses a plain GRU instead
    of Transformer/attention for SFC sequence decoding.
    """

    def __init__(self, p_net_num_nodes, feature_dim, edge_dim=None, embedding_dim=128, dropout_prob=0.1,
                 batch_norm=False, gnn_type='gat', num_gnn_layers=3):
        super().__init__()
        self.p_net_num_nodes = max(p_net_num_nodes, 100)
        self.emb = nn.Embedding(self.p_net_num_nodes + 1, embedding_dim)
        GnnNet = BasicPNetFeatureEncoder if gnn_type == 'mlp' else get_gnn_class(gnn_type)
        self.gcn = GnnNet(
            feature_dim,
            embedding_dim,
            edge_dim=edge_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            return_batch=True,
            num_layers=num_gnn_layers
        )
        self.context_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()
        )

    def _format_hidden_state(self, hidden_state, batch_size, embedding_dim, device):
        if hidden_state is None:
            return torch.zeros(1, batch_size, embedding_dim, device=device)
        if hidden_state.dim() == 2:
            # [batch, dim] -> [1, batch, dim]
            hidden_state = hidden_state.unsqueeze(0)
        elif hidden_state.dim() == 3:
            # Stored observations usually have [batch, 1, dim].
            if hidden_state.size(0) == batch_size and hidden_state.size(1) == 1:
                hidden_state = hidden_state.permute(1, 0, 2)
        return hidden_state

    def forward(self, obs):
        batch_p_net = obs['p_net']
        encoder_outputs = obs['encoder_outputs']  # [batch, seq_len, embedding_dim]
        p_node_id = obs['p_node_id']
        batch_size = encoder_outputs.size(0)
        embedding_dim = encoder_outputs.size(-1)

        if p_node_id.dim() == 1:
            p_node_id = p_node_id.view(1, batch_size)
        elif p_node_id.dim() == 0:
            p_node_id = p_node_id.unsqueeze(0).unsqueeze(0)
        p_node_id = torch.clamp(p_node_id, max=self.p_net_num_nodes)
        p_node_emb = self.emb(p_node_id)  # [1, batch, embedding_dim]

        hidden_state = self._format_hidden_state(
            obs.get('hidden_state', None),
            batch_size,
            embedding_dim,
            encoder_outputs.device
        )
        # Strict ablation: no Transformer/GRU attention. The context is a
        # simple projection of the previous physical-node embedding and the
        # simple SFC sequence hidden vector.
        hidden_last = hidden_state[-1]  # [batch, embedding_dim]
        prev_node = p_node_emb.squeeze(0)  # [batch, embedding_dim]
        decoder_context = torch.tanh(self.context_proj(torch.cat([prev_node, hidden_last], dim=-1)))

        p_node_embeddings = self.gcn(batch_p_net)  # [batch, num_nodes, embedding_dim]
        fused_node_embeddings = p_node_embeddings + decoder_context.unsqueeze(1)
        logits = self.mlp(fused_node_embeddings)
        outputs = decoder_context.unsqueeze(0)
        return logits, outputs, hidden_state


class Decoder(nn.Module):
    def __init__(self, p_net_num_nodes, feature_dim, edge_dim=None, embedding_dim=128, dropout_prob=0.1,
                 batch_norm=False, gnn_type='gcn'):
        super().__init__()
        self.p_net_num_nodes = max(p_net_num_nodes, 100)
        self.emb = nn.Embedding(self.p_net_num_nodes + 1, embedding_dim)
        # TODO
        self.vnf_type_embedding = nn.Embedding(3, embedding_dim)
        # Transformer Decoder Layer
        decoder_layer = TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dropout=dropout_prob,
            batch_first=False
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=2)
        # GNN for physical network embedding
        GnnNet = BasicPNetFeatureEncoder if gnn_type == 'mlp' else get_gnn_class(gnn_type)
        self.gcn = GnnNet(
            feature_dim,
            embedding_dim,
            edge_dim=edge_dim,
            dropout_prob=dropout_prob,
            batch_norm=batch_norm,
            return_batch=True
        )
        # Attention for hidden_state aggregation
        self.hidden_att = MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout_prob,
            batch_first=False
        )
        # Final MLP for action logits
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Flatten()
        )

    def forward(self, obs):
        batch_p_net = obs['p_net']
        encoder_outputs = obs['encoder_outputs']  # [batch_size, seq_len,embedding_dim]
        mask = obs.get('mask', None)  # [batch_size, seq_len]
        p_node_id = obs['p_node_id']  # [batch_size] or [1]

        # Ensure p_node_id is [batch_size, 1]
        batch_size = encoder_outputs.size(0)
        if p_node_id.dim() == 1:
            p_node_id = p_node_id.view(1, batch_size)  # [1, batch]
        elif p_node_id.dim() == 0:
            p_node_id = p_node_id.unsqueeze(0).unsqueeze(0)  # [1, 1]

        # Clamp and embed p_node_id
        p_node_id = torch.clamp(p_node_id, max=self.p_net_num_nodes)
        p_node_emb = self.emb(p_node_id)  # [1, batch_size, embedding_dim]
        # GNN embeddings for physical network nodes
        p_node_embeddings = self.gcn(batch_p_net)  # [batch_size, num_nodes, embedding_dim]
        p_node_embeddings = p_node_embeddings.permute(1, 0, 2)  # [num_nodes, batch, d]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # Transformer Decoder with mask
        decoder_outputs = self.transformer_decoder(
            tgt=p_node_emb,
            memory=encoder_outputs,
            memory_key_padding_mask=mask
        )  # [batch_size, 1, embedding_dim]

        # TODO
        # src = obs['src']
        # dst = obs['dst']
        # 灏嗘暣鏁拌浆鎹负涓€涓紶閲?
        decoder_context = decoder_outputs.repeat(p_node_embeddings.size(0), 1, 1)

        # Fuse decoder context with GNN embeddings
        fused_node_embeddings = p_node_embeddings + decoder_context
        # fused_node_embeddings = p_node_embeddings + decoder_context + src_emb + dst_emb
        fused_node_embeddings = fused_node_embeddings.permute(1, 0, 2)

        # Action logits
        logits = self.mlp(fused_node_embeddings)  # [batch_size, num_nodes]

        hidden_state = decoder_outputs[-1:].detach()  # [1, batch_size, embedding_dim]
        return logits, decoder_outputs, hidden_state


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, hidden, encoder_outputs, mask=None):
        if hidden.dim() == 3:
            hidden = hidden[-1]  # (batch_size, hidden_dim)
        elif hidden.dim() == 4:
            hidden = hidden[-1, 0]  # (hidden[灞傦紝batch锛?,dim])

        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        attn_scores = torch.bmm(hidden, encoder_outputs.transpose(1, 2))
        attn_scores = attn_scores.squeeze(1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e10)

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        return context, attn_weights
# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.v = nn.Linear(hidden_dim, 1, bias=False)
#
#     def forward(self, hidden, encoder_outputs, mask=None):
#         # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
#         # encoder_outputs shape: (batch_size, seq_len, hidden_dim * num_directions)
#         batch_size = encoder_outputs.size(0)
#         seq_len = encoder_outputs.size(1)
#         hidden = hidden.transpose(0, 1).repeat(1, seq_len, 1)  # shape: (batch_size, seq_len, hidden_dim)
#         energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))  # shape: (batch_size, seq_len, hidden_dim)
#         attn_scores = self.v(energy).squeeze(2)  # (batch_size, seq_len)
#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
#         attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_len)
#         context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim * num_directions)
#         return context, attn_weights

