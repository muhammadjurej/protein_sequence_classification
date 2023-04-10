from dataset import DatasetProtein
from config import Confiq
from tools import Tools
from torch.utils.data import DataLoader
from torchsummary import summary
from model_builder import BILSTM
import engine
import torch

cfg = Confiq()
tool = Tools()
common_family = tool.find_common_family()

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds= DatasetProtein(cfg.ROOT_PATH, 'train', cfg.AMINO, common_family=common_family)
test_ds= DatasetProtein(cfg.ROOT_PATH, 'test', cfg.AMINO, common_family=common_family)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = BILSTM(num_embedding=len(cfg.AMINO)+1, output_size=cfg.number_family, embedding_dim=cfg.number_family, hidden_dim=512, n_layers=2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

engine.train(
    model=model,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=5,
    device=device
)

tool.save_model(
    model=model,
    target_dir="model_results",
    model_name="usk_protein_LSTM_model.pth"
)