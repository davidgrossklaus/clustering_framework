from model.DCRN.model import DCRN
import torch
import numpy as np
from utils.data_processor import normalize_adj, numpy_to_torch, diffusion_adj
from sklearn.decomposition import PCA

dataset_name = "internetz"
clusters = 15

param_dict = {
        "acm": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 5e-5, "n_input": 100},
        "dblp": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-4, "n_input": 50},
        "cite": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-5, "n_input": 100},
        "amap": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-3, "n_input": 100},
        "internetz": {"alpha_value": 0.2, "lambda_value": 10, "gamma_value": 1e3, "lr": 1e-3, "n_input": 100}
}
max_epoch = 400
alpha_value = param_dict[dataset_name]["alpha_value"]
lambda_value = param_dict[dataset_name]["lambda_value"]
gamma_value = param_dict[dataset_name]["gamma_value"]
lr = param_dict[dataset_name]["lr"]
n_input = param_dict[dataset_name]["n_input"]
embedding_dim = 20
ae_n_enc_1 = 128
ae_n_enc_2 = 256
ae_n_enc_3 = 512
ae_n_dec_1 = 512
ae_n_dec_2 = 256
ae_n_dec_3 = 128
gae_n_enc_1 = 128
gae_n_enc_2 = 256
gae_n_enc_3 = 20
gae_n_dec_1 = 20
gae_n_dec_2 = 256
gae_n_dec_3 = 128

n_node = 1000 
device = "cuda" if torch.cuda.is_available() else "cpu"

pretrain_ae_filename = "pretrain/pretrain_ae/DCRN/acm/acm.pkl"
pretrain_igae_filename = "pretrain/pretrain_igae/DCRN/acm/acm.pkl"

feature = np.load("dataset/internetz/features.npy")
adjMatrix = np.load("dataset/internetz/adjMatrix.npy")

model = DCRN(dataset_name, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, embedding_dim, clusters, n_node).to(device)

#model.ae.load_state_dict(torch.load(pretrain_ae_filename, map_location='cpu'))
#model.igae.load_state_dict(torch.load(pretrain_igae_filename, map_location='cpu'))

print(model.ae, model.igae)

pca = PCA(n_components=n_input)
X_pca = pca.fit_transform(feature)
X_pca = numpy_to_torch(X_pca).to(device).float()
adj = adjMatrix + np.eye(adjMatrix.shape[0])
print(adjMatrix)
adj_norm = normalize_adj(adjMatrix, symmetry=False)
adj_norm = numpy_to_torch(adj_norm).to(device).float()
Ad = diffusion_adj(adjMatrix, mode='ppr', transport_rate=alpha_value)
Ad = numpy_to_torch(Ad).to(device).float()
#label = data.label
adj = numpy_to_torch(adj)

X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Q, embedding, AZ_all, Z_all = model(X_pca, adj_norm, X_pca, adj_norm)

