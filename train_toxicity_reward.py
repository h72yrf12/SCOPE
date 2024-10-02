from reward_model.sentence_toxicity_model import SentenceToxicRegressionModel, SentenceToxicClassifierModel
from torch.utils.data import DataLoader
from reward_model.toxicity_data import ToxicityData
import pickle
import torch

with open('data/toxicity_data_classification/sexual_train.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

with open('data/toxicity_data_classification/sexual_validation.pkl', 'rb') as f:
    val_dataset = pickle.load(f)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

from transition_models.deterministic_transition_model import DeterministicTransitionModel

device = torch.device('cuda:4')
#model = SentenceToxicRegressionModel(cuda=device)
model = SentenceToxicClassifierModel(cuda=device)
t,v = model.train(train_loader, val_loader, path_to_save="reward_model/sentence_sexual", lr=0.001, num_epochs=200)
