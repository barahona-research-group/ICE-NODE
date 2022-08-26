from .gram import MatrixEmbeddings, GRAM, AbstractEmbeddingsLayer, CachedGRAM
from .glove import glove_representation

embedding_cls = AbstractEmbeddingsLayer.embedding_cls
short_tag = AbstractEmbeddingsLayer.short_tag
