from models.autorec import autorec
from models.init_feature_embed_ae import initfeatureembedae
from models.alter_feature_embed_ae import alterfeatureembedae
from models.concat_feature_embed_ae import concatfeatureembedae
from models.union_sample_ae import unionsampleae
from models.wr_sample_ae import wrsampleae
from models.batch_sample_ae import batchsampleae
from models.bridge_label_ae import bridgelabelae
from models.refine_label_ae import refinelabelae
from models.deep_autorec import deepautorec
from models.soft_label_ae import softlabelae
from models.hint_ae import hintae


models = {
    "AutoRec": autorec,
    "InitFeatureEmbedAE": initfeatureembedae,
    "AlterFeatureEmbedAE": alterfeatureembedae,
    "ConcatFeatureEmbedAE": concatfeatureembedae,
    "UnionSampleAE": unionsampleae,
    "WRSampleAE": wrsampleae,
    "BatchSampleAE": batchsampleae,
    "BridgeLabelAE": bridgelabelae,
    "RefineLabelAE": refinelabelae,
    "DeepAutoRec": deepautorec,
    "SoftLabelAE": softlabelae,
    "HintAE": hintae,
}
