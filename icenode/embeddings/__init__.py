from .gram import MatrixEmbeddings, GRAM, AbstractEmbeddingsLayer
from .glove import glove_representation

embedding_cls = AbstractEmbeddingsLayer.embedding_cls
short_tag = AbstractEmbeddingsLayer.short_tag
