from models.biased_mf import biasedmf
from models.propensity_mf import propensitymf
from models.init_feature_embed_mf import initfeatureembedmf
from models.alter_feature_embed_mf import alterfeatureembedmf
from models.concat_feature_embed_mf import concatfeatureembedmf
from models.causal_sample_mf import causalsamplemf
from models.union_sample_mf import unionsamplemf
from models.wr_sample_mf import wrsamplemf
from models.batch_sample_mf import batchsamplemf
from models.bridge_label_mf import bridgelabelmf
from models.refine_label_mf import refinelabelmf
from models.restricted_batch_sample_mf import restrictedbatchsamplemf


models = {
    "BiasedMF": biasedmf,
    "PropensityMF": propensitymf,
    "InitFeatureEmbedMF": initfeatureembedmf,
    "AlterFeatureEmbedMF": alterfeatureembedmf,
    "ConcatFeatureEmbedMF": concatfeatureembedmf,
    "CausalSampleMF": causalsamplemf,
    "UnionSampleMF": unionsamplemf,
    "WRSampleMF": wrsamplemf,
    "BatchSampleMF": batchsamplemf,
    "BridgeLabelMF": bridgelabelmf,
    "RefineLabelMF": refinelabelmf,
    "RestrictedBatchSampleMF": restrictedbatchsamplemf,
}
