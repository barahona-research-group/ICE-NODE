from .gram import MatrixEmbeddings, GRAM, AbstractEmbeddingsLayer, CachedGRAM
from .glove import train_glove

embedding_cls = AbstractEmbeddingsLayer.embedding_cls
short_tag = AbstractEmbeddingsLayer.short_tag
