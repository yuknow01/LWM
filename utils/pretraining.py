#%% PACKAGES & MODULES
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from inference import prepare_for_lwm
from input_preprocess import tokenizer
from lwm_model import lwm
import numpy as np

#%% PARAMETERS
n_epochs = 100
n_layers = 12
n_heads = 12
d_model = 64
d_ff = d_model * 4
d_k = d_model // n_heads
d_v = d_model // n_heads
dropout = 0.1
max_len = 129
element_length = 16
batch_size = 64
train_ratio = 0.7
val_ratio = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% PRE-TRAINING DATA GENERATION
# The following DeepMIMO scenarios are not enough for pre-training a 
# Transformer-based foundation model like LWM. Add more scenarios for 
# more effective pre-training. The instruction for reproducing the actual 
# dataset used for pre-training LWM can be found in the Huggingface forum.
scenario_names = np.array([
    "city_18_denver", "city_15_indianapolis", "city_19_oklahoma", 
    "city_12_fortworth", "city_11_santaclara", "city_7_sandiego"
])

scenario_idxs = np.array([0, 1, 2, 3, 4, 5])  
selected_scenario_names = scenario_names[scenario_idxs]

preprocessed_chs = tokenizer(
    selected_scenario_names=selected_scenario_names, 
    manual_data=None, 
    gen_raw=False) 

#%% DATALOADER
train_size = int(train_ratio * len(preprocessed_chs))
val_size = int(val_ratio * len(preprocessed_chs))
test_size = len(preprocessed_chs) - val_size - train_size

train_data, val_data, test_data = torch.utils.data.random_split(
    preprocessed_chs, [train_size, val_size, test_size]
)

train_loader = prepare_for_lwm(train_data, device, batch_size=batch_size, shuffle=True)
val_loader = prepare_for_lwm(val_data, device, batch_size=batch_size, shuffle=True)
test_loader = prepare_for_lwm(test_data, device, batch_size=batch_size, shuffle=True)

# %% Model
load_model = False

model = lwm()
model.to(device)

if load_model:
    model_name = 'models/pretrained_model.pth'
    model.load_state_dict(torch.load(model_name))
    print(f"Model loaded from {model_name}")
    
# Loss function
criterionMLM = nn.MSELoss()

# %% Optimizer and Scheduler
adaptive_lr = False

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = (
    optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    if adaptive_lr
    else StepLR(optimizer, step_size=10, gamma=0.9)
)

# %% Training
training_loss = []
validation_loss = []

def train(model, dataloader, optimizer, scheduler=None, device="cuda"):

    model.train()
    running_loss = 0.0
    criterionMCM = nn.MSELoss()

    for idx, batch in enumerate(dataloader):
        input_ids = batch[0].to(device)
        masked_tokens = batch[1].to(device)
        masked_pos = batch[2].to(device)
        
        optimizer.zero_grad()
        
        logits_lm, _ = model(input_ids, masked_pos)
        loss_lm = criterionMCM(logits_lm, masked_tokens)
        loss = loss_lm / torch.var(masked_tokens) 
        
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)

    return average_loss

def validate(model, dataloader, device="cuda"):
    model.eval()
    running_loss = 0.0
    criterionMCM = nn.MSELoss()

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            input_ids = batch[0].to(device)
            masked_tokens = batch[1].to(device)
            masked_pos = batch[2].to(device)

            logits_lm, _ = model(input_ids, masked_pos)

            loss_lm = criterionMCM(logits_lm, masked_tokens)
            loss = loss_lm / torch.var(masked_tokens)  

            running_loss += loss.item()

    average_loss = running_loss / len(dataloader)

    return average_loss

# %% Training Loop
for epoch in range(n_epochs):
    print(f"Epoch {epoch + 1}/{n_epochs}")

    # Training step
    train_loss = train(model, train_loader, optimizer, scheduler, device)
    training_loss.append(train_loss)
    print(f"Training Loss: {train_loss:.4f}")

    # Validation step
    if val_loader is not None:
        val_loss = validate(model, val_loader, device)
        validation_loss.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")
