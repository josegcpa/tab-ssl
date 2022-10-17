from src.data import load_firewall
from src.data_generator import EmbeddingGenerator, get_cat_info

X, y = load_firewall()

print('X shape: ', X.shape)

cat_idxs, cat_dims = get_cat_info(X)

print('cat_idxs: ', cat_idxs)
print('cat_dims: ', cat_dims)

embedings = EmbeddingGenerator(X.shape[-1], cat_idxs, cat_dims)

print(embedings)

pos_embedings = embedings.post_embed_dim

print('pos_embedings: ', pos_embedings)

