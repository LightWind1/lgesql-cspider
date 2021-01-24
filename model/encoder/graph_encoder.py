#coding=utf8
import torch
import torch.nn as nn
from model.encoder.graph_input import *
from model.encoder.lgnn import LGNN
from model.encoder.rat import RAT
from model.encoder.lgnn_plus_rat import LGNNPlusRAT
from model.encoder.graph_output import *
from model.model_utils import Registrable

@Registrable.register('encoder_hetgnn')
class HetGNNEncoder(nn.Module):

    def __init__(self, args):
        super(HetGNNEncoder, self).__init__()
        lazy_load = args.lazy_load if hasattr(args, 'lazy_load') else False
        self.input_layer = GraphInputLayer(args.embed_size, args.gnn_hidden_size, args.word_vocab, dropout=args.dropout,
            schema_aggregation=args.schema_aggregation, add_cls=args.add_cls) \
            if args.ptm is None else GraphInputLayerPTM(args.ptm, args.gnn_hidden_size, dropout=args.dropout, add_cls=args.add_cls,
                subword_aggregation=args.subword_aggregation, schema_aggregation=args.schema_aggregation, lazy_load=lazy_load)
        self.hidden_layer = Registrable.by_name(args.model)(args)
        self.output_layer = Registrable.by_name(args.output_model)(args)

    def forward(self, batch):
        outputs = self.input_layer(batch)
        outputs, lg_outputs = self.hidden_layer(outputs, batch)
        return self.output_layer(outputs, lg_outputs, batch)