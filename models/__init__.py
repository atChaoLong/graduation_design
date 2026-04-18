from .textcnn import TextCNNForMultiLabel
from .bilstm import BiLSTMForMultiLabel
from .bert_baseline import BERTForMultiLabel
from .bert_gcn import BertGCNForMultiLabel

__all__ = [
    'TextCNNForMultiLabel',
    'BiLSTMForMultiLabel',
    'BERTForMultiLabel',
    'BertGCNForMultiLabel',
]
