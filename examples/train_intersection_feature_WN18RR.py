import openke
from openke.config import Trainer, Tester
from openke.module.model import LinkPredictor
from openke.module.loss import *
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from openke.utils import set_seeds_all

# dataloader for training
hash_hops = 1
set_seeds_all(seed=23)
train_dataloader = TrainDataLoader(
    in_path="./benchmarks/WN18RR/",
    nbatches=100,
    threads=8,
    sampling_mode="normal",
    bern_flag=1,
    filter_flag=1,
    neg_ent=25,
    neg_rel=0,
    device="cuda",
    max_hash_hops = hash_hops
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link", device="cuda", max_hash_hops = hash_hops)

lr = 0.005
epochs = 3000
hidden_dim = 128
opt = "Adam"
loss = "margin_loss"

print(f"Training config: lr={lr}, optmizer={opt}, loss = {loss}, epochs = {epochs}, hash_hops = {hash_hops}")
predictor = LinkPredictor(
    ent_tot=train_dataloader.get_ent_tot(),
    rel_tot=train_dataloader.get_rel_tot(),
    hidden_dim=hidden_dim,
    max_hash_hops = hash_hops
)
# define the loss function
model = NegativeSampling(
    model=predictor,
    loss=MarginLoss(margin=5.0),
    batch_size=train_dataloader.get_batch_size()
)

trainer = Trainer(
    model=model,
    data_loader=train_dataloader,
    train_times=epochs,
    opt_method=opt,
    alpha=lr,
    use_gpu=True,
)

trainer.run()
predictor.save_checkpoint(f"./checkpoint/intersection_feature_WN18RR_{lr}_{opt}_{loss}_{epochs}.ckpt")

# test the model
predictor.load_checkpoint(f"./checkpoint/intersection_feature_WN18RR_{lr}_{opt}_{loss}_{epochs}.ckpt")
predictor.eval()
tester = Tester(model=predictor, data_loader=test_dataloader, use_gpu=True)
tester.run_link_prediction(type_constrain=False)
