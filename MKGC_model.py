import torch
import torch.nn as nn
import pickle

class M_REFT(nn.Module):
    """完整重构的多模态知识图谱模型"""

    # 模型常量定义
    SEGMENT_LENGTH = 6
    SEMANTIC_RELATED_ENTITIES = 2
    NEIGHBOR_COUNT = 8

    def __init__(self, num_entities, num_relations, embedding_dim,
                 num_heads, hidden_dim, num_encoder_layers,
                 num_decoder_layers, dropout=0.01, **kwargs):
        super().__init__()

        # 初始化模型参数
        self._init_hyperparameters(num_entities, num_relations, embedding_dim,
                                 num_heads, hidden_dim, num_encoder_layers,
                                 num_decoder_layers, dropout, **kwargs)

        # 初始化嵌入层
        self._init_embeddings()

        # 初始化编码器-解码器结构
        self._init_transformer_layers()

        # 初始化卷积模块
        self._init_convolutional_layers()

        # 加载预训练特征
        self._load_pretrained_features()

        # 初始化位置编码
        self._init_positional_embeddings(self.embedding_dim)

        # 权重初始化
        self._init_weights()

    def _init_hyperparameters(self, num_entities, num_relations, embedding_dim,
                            num_heads, hidden_dim, num_encoder_layers,
                            num_decoder_layers, dropout, **kwargs):
        """初始化模型超参数"""
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

        # 评分函数类型
        self.score_function = kwargs.get('score_function', "DB15K")

        # 位置索引
        self.rel_index = kwargs.get('rel_index')
        self.source_ent_index = kwargs.get('source_ent_index')
        self.SREs_index = kwargs.get('SREs_index')
        self.neighbors_index = kwargs.get('neighbors_index')

        # Dropout率
        self.emb_dropout = kwargs.get('emb_dropout', 0.6)
        self.vis_dropout = kwargs.get('vis_dropout', 0.4)
        self.txt_dropout = kwargs.get('txt_dropout', 0.1)

    def _init_embeddings(self):
        """初始化各种嵌入表示"""
        dim = self.embedding_dim

        # 实体和关系嵌入
        self.entity_embed = nn.Embedding(self.num_entities + 1, dim)
        self.relation_embed = nn.Embedding(self.num_relations * 2, dim)

        # 特殊token
        self.entity_token = nn.Parameter(torch.Tensor(1, 1, dim))
        self.decoder_head_token = nn.Parameter(torch.Tensor(1, 1, dim))
        self.decoder_tail_token = nn.Parameter(torch.Tensor(1, 1, dim))
        self.link_prediction_token = nn.Parameter(torch.Tensor(1, dim))

        # LayerNorm层
        self.str_ent_ln = nn.LayerNorm(dim)
        self.str_rel_ln = nn.LayerNorm(dim)
        self.vis_ln = nn.LayerNorm(dim)
        self.txt_ln = nn.LayerNorm(dim)
        self.conv_ln = nn.LayerNorm(dim)
        self.de_conv_ln = nn.LayerNorm(dim)
        self.bnm = nn.LayerNorm(dim)

        # Dropout层
        self.emb_drop = nn.Dropout(p=self.emb_dropout)
        self.vis_drop = nn.Dropout(p=self.vis_dropout)
        self.txt_drop = nn.Dropout(p=self.txt_dropout)

    def _init_positional_embeddings(self, dim):
        """初始化位置编码"""
        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, dim))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1, 1, dim))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1, 1, dim))
        self.pos_head = nn.Parameter(torch.Tensor(1, 1, dim))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, dim))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, dim))

    def _init_transformer_layers(self):
        """初始化Transformer编码器/解码器"""
        encoder_layer = nn.TransformerEncoderLayer(
            self.embedding_dim, self.num_heads, self.hidden_dim,
            self.dropout, batch_first=True)

        self.entity_encoder = nn.TransformerEncoder(
            encoder_layer, self.num_encoder_layers)

        self.relation_encoder = nn.TransformerEncoder(
            encoder_layer, self.num_encoder_layers)

        decoder_layer = nn.TransformerEncoderLayer(
            self.embedding_dim, self.num_heads, self.hidden_dim,
            self.dropout, batch_first=True)

        self.decoder = nn.TransformerEncoder(
            decoder_layer, self.num_decoder_layers)

    def _init_convolutional_layers(self):
        """初始化卷积网络层"""
        # 特征融合卷积
        self.fus_conv1 = nn.Conv2d(1, 16, (4, 4), (4, 4))
        self.fus_fc1 = nn.Linear(64, 64)
        self.fus_drop1 = nn.Dropout(0.3)

        self.fus_conv2 = nn.Conv2d(16, 16, (3, 3), (4, 4))
        self.fus_fc2 = nn.Linear(256, self.embedding_dim)
        self.fus_drop2 = nn.Dropout(0.3)

        # 解码器卷积
        self.de_conv1 = nn.Conv2d(1, 16, (2, 2), (1, 1))
        self.de_fc1 = nn.Linear(7440, 512)
        self.de_drop1 = nn.Dropout(0.5)

        self.de_conv2 = nn.Conv2d(1, 32, (4, 4), (2, 2))
        self.de_fc2 = nn.Linear(3360, self.num_entities)
        self.de_drop2 = nn.Dropout(0.4)

    def _load_pretrained_features(self):
        """加载预训练的多模态特征"""
        with open(f'data/{self.score_function}/segment_img_feature.pickle', 'rb') as f:
            img_feat = pickle.load(f)
        self.segment_img_feature = torch.tensor(img_feat).float()[:, :self.SEGMENT_LENGTH, :].cuda()
        self.seg_img_ln = nn.Linear(1000, self.embedding_dim).cuda()

        with open(f'data/{self.score_function}/segment_txt_feature.pickle', 'rb') as f:
            txt_feat = pickle.load(f)
        self.segment_txt_feature = torch.tensor(txt_feat).float()[:, :self.SEGMENT_LENGTH, :].cuda()
        self.seg_txt_ln = nn.Linear(768, self.embedding_dim).cuda()

    def _init_weights(self):
        """初始化模型权重"""
        nn.init.xavier_uniform_(self.fus_conv1.weight)
        nn.init.xavier_uniform_(self.fus_conv2.weight)
        nn.init.xavier_uniform_(self.fus_fc1.weight)
        nn.init.xavier_uniform_(self.seg_img_ln.weight)
        nn.init.xavier_uniform_(self.seg_txt_ln.weight)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.entity_token)
        nn.init.xavier_uniform_(self.decoder_head_token)
        nn.init.xavier_uniform_(self.decoder_tail_token)
        nn.init.xavier_uniform_(self.link_prediction_token)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

    def _fuse_multimodal_features(self):
        """多模态特征融合"""
        # 视觉特征处理
        vis_embs = self.vis_drop(self.bnm(self.seg_img_ln(self.segment_img_feature)))
        vis_embs_primary = self.vis_drop(vis_embs[:, 0, :].unsqueeze(1))

        # 文本特征处理
        txt_embs = self.txt_drop(self.bnm(self.seg_txt_ln(self.segment_txt_feature)))
        txt_embs_primary = self.txt_drop(txt_embs[:, 0, :].unsqueeze(1))

        # 结构特征处理
        str_embs = self.emb_drop(self.str_ent_ln(
            self.entity_embed(self.source_ent_index).unsqueeze(1)))

        # 多模态融合
        fused_features = torch.cat([vis_embs, txt_embs], dim=1).unsqueeze(1)
        conv1_out = self.fus_drop1(self.fus_fc1(self.fus_conv1(fused_features)))
        conv2_out = self.fus_drop2(self.conv_ln(
            self.fus_fc2(self.fus_conv2(conv1_out).view(self.num_entities, -1)))).unsqueeze(1)

        # 实体编码
        entity_seq = torch.cat([str_embs, vis_embs_primary, txt_embs_primary, conv2_out], dim=1)
        return self.entity_encoder(entity_seq)[:, 0, :]

    def _encode_entities(self, fused_entities):
        """编码实体及其语义相关实体"""
        entity_tokens = self.entity_token.expand(self.num_entities, -1, -1)
        sre_index = self.SREs_index[:, :self.SEMANTIC_RELATED_ENTITIES]
        sre_entities = self.emb_drop(self.str_ent_ln(fused_entities[sre_index]))

        entity_seq = torch.cat([entity_tokens, fused_entities.unsqueeze(1), sre_entities], dim=1)
        return self.relation_encoder(entity_seq)[:, 0]

    def _encode_relations(self):
        """编码关系特征"""
        rel_emb = self.emb_drop(self.str_rel_ln(
            self.relation_embed(self.rel_index).unsqueeze(1)))
        return rel_emb.squeeze(1)

    def forward(self):
        """完整的前向传播流程"""
        fused_entities = self._fuse_multimodal_features()
        encoded_entities = self._encode_entities(fused_entities)
        encoded_relations = self._encode_relations()
        return torch.cat([encoded_entities, self.link_prediction_token], dim=0), encoded_relations

    def _score_mkgw(self, entity_emb, relation_emb, triplets):
        """MKG-W评分函数实现"""
        # 准备解码序列
        h_seq = entity_emb[triplets[:, 0]].unsqueeze(1) + self.pos_head
        r_seq = relation_emb[triplets[:, 1]].unsqueeze(1) + self.pos_rel
        neighbors = entity_emb[self.neighbors_index[triplets[:, 0]][:, :self.NEIGHBOR_COUNT]] + self.pos_tail
        decoder_tokens = self.decoder_head_token.expand(len(triplets), -1, -1)

        # 解码过程
        dec_seq = torch.cat([decoder_tokens, h_seq, r_seq, neighbors], dim=1)
        dec_output = self.decoder(dec_seq)

        # 卷积评分
        rel_emb = dec_output[:, 2, :]
        ctx_emb = dec_output[:, 0, :]
        conv_input = torch.stack([ctx_emb, rel_emb], dim=1).view(len(triplets), 1, 16, -1)

        conv1_out = self.de_drop1(self.de_fc1(self.de_conv1(conv_input).view(len(triplets), -1)))
        conv2_out = self.de_drop2(self.de_fc2(self.de_conv2(conv1_out.view(len(triplets), 1, 16, -1)).view(len(triplets), -1)))

        return torch.mm(conv2_out, entity_emb[:-1].t())

    def _score_db15k(self, entity_emb, relation_emb, triplets):
        """DB15K评分函数实现"""
        # 准备解码序列
        h_seq = entity_emb[triplets[:, 0]].unsqueeze(1) + self.pos_head
        r_seq = relation_emb[triplets[:, 1]].unsqueeze(1) + self.pos_rel
        neighbors = entity_emb[self.neighbors_index[triplets[:, 0]][:, :self.NEIGHBOR_COUNT]] + self.pos_tail
        decoder_tokens = self.decoder_head_token.expand(len(triplets), -1, -1)

        # 解码过程
        dec_seq = torch.cat([decoder_tokens, h_seq, r_seq, neighbors], dim=1)
        dec_output = self.decoder(dec_seq)

        # 卷积评分
        rel_emb = dec_output[:, 1, :]
        ctx_emb = dec_output[:, 0, :]
        conv_input = torch.stack([ctx_emb, rel_emb], dim=1).view(len(triplets), 1, 16, -1)

        conv1_out = self.de_drop1(self.de_fc1(self.de_conv1(conv_input).view(len(triplets), -1)))
        return self.de_drop2(self.de_fc2(self.de_conv2(conv1_out.view(len(triplets), 1, 16, -1)).view(len(triplets), -1)))

    def _score_mkgy(self, entity_emb, triplets):
        """MKG-Y评分函数实现"""
        # 准备解码序列
        h_seq = entity_emb[triplets[:, 0]].unsqueeze(1) + self.pos_head
        r_seq = self.relation_embed(triplets[:, 1]).unsqueeze(1) + self.pos_rel
        neighbors = entity_emb[self.neighbors_index[triplets[:, 0]][:, :self.NEIGHBOR_COUNT]] + self.pos_tail
        decoder_tokens = self.decoder_head_token.expand(len(triplets), -1, -1)

        # 解码过程
        dec_seq = torch.cat([decoder_tokens, h_seq, r_seq, neighbors], dim=1)
        dec_output = self.decoder(dec_seq)

        # 直接使用上下文向量评分
        ctx_emb = dec_output[:, 0, :]
        return torch.mm(ctx_emb, entity_emb[:-1].t())

    def score(self, entity_emb, relation_emb, triplets):
        """统一的评分函数接口"""
        if self.score_function == "MKG-W":
            return self._score_mkgw(entity_emb, relation_emb, triplets)
        elif self.score_function == "DB15K":
            return self._score_db15k(entity_emb, relation_emb, triplets)
        elif self.score_function == "MKG-Y":
            return self._score_mkgy(entity_emb, triplets)
        else:
            raise ValueError(f"未知的评分函数: {self.score_function}")
