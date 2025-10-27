# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os



# Import raw data
#Defining function to check if directory exists, if not it generates it
def check_and_make_dir(dir):
    if not os.path.isdir(dir):os.mkdir(dir)
#Base directory 
base_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/Second_Generals/'
#File containing temperature values
raw_T_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_T.csv', delimiter=',')
#File containing pressure values
raw_P_data = np.loadtxt(base_dir+'Data/bt-4500k/training_data_P.csv', delimiter=',')
#Path to store model
model_save_path = base_dir+'Model_Storage/RNN/'
check_and_make_dir(model_save_path)
#Path to store plots
plot_save_path = base_dir+'Plots/RNN/'
check_and_make_dir(plot_save_path)

#Last 51 columns are the temperature/pressure values, 
#First 5 are the input values (H2 pressure in bar, CO2 pressure in bar, LoD in hours, Obliquity in deg, H2+Co2 pressure) but we remove the last one since it's not adding info.
raw_inputs = raw_T_data[:, :4]
raw_outputs_T = raw_T_data[:, 5:]
raw_outputs_P = raw_P_data[:, 5:]

#Storing useful quantitites
N = raw_inputs.shape[0] #Number of data points
D = raw_inputs.shape[1] #Number of features
O = raw_outputs_T.shape[1] #Number of outputs






# Potentially shrink the dataset
#Number of samples to shrink our dataset to 
sample_size = 10000

filter = np.random.choice(np.arange(N), size=sample_size, replace=False)

shrink_inputs = torch.tensor(raw_inputs[filter, :], dtype=torch.float32)
shrink_outputs_T = torch.tensor(raw_outputs_T[filter, :], dtype=torch.float32)
shrink_outputs_P = torch.tensor(raw_outputs_P[filter, :], dtype=torch.float32)

N = sample_size






# Define training, validation, and testing dataset
#Defining partition of data used for 1. training 2. validation and 3. testing
data_partitions = [0.7, 0.1, 0.2]

#Defining the noise seed for the random partitioning of the training data
partition_seed = 4

#Splitting the data 
## Setting noise seec
generator = torch.Generator().manual_seed(partition_seed)
## Retrieving indices of data partitions
train_idx, valid_idx, test_idx = torch.utils.data.random_split(range(N), data_partitions, generator=generator)
## Generate the data partitions
### Training
train_inputs = shrink_inputs[train_idx]
train_outputs_T = shrink_outputs_T[train_idx]
train_outputs_P = shrink_outputs_P[train_idx]
### Validation
valid_inputs = shrink_inputs[valid_idx]
valid_outputs_T = shrink_outputs_T[valid_idx]
valid_outputs_P = shrink_outputs_P[valid_idx]
### Testing
test_inputs = shrink_inputs[test_idx]
test_outputs_T = shrink_outputs_T[test_idx]
test_outputs_P = shrink_outputs_P[test_idx]






# Define RNN classes
class RecurrentNeuralNetworkCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input + previous hidden layer
        self.input_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        # Hidden layer n.1
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size, bias=True)
        # Hidden layer n.2
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Hidden layer n.3
        self.hidden_layer3 = nn.Linear(hidden_size, hidden_size, bias=True)
        # Hidden layer n.4
        self.hidden_layer4 = nn.Linear(hidden_size, hidden_size, bias=True)
        # Hidden layer n.5
        self.hidden_layer5 = nn.Linear(hidden_size, hidden_size, bias=True)
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, h_prev):
        # Concatenate input and previous hidden state
        combined = torch.cat((x, h_prev), dim=1)

        # Pass through all hidden layers manually
        h = torch.tanh(self.input_layer(combined))
        h = torch.tanh(self.hidden_layer1(h))
        h = torch.tanh(self.hidden_layer2(h))
        h = torch.tanh(self.hidden_layer3(h))
        h = torch.tanh(self.hidden_layer4(h))
        h = torch.tanh(self.hidden_layer5(h))

        # Output
        y = self.output_layer(h)
        return y, h
    
class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RecurrentNeuralNetworkCell(input_size, hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        Returns:
            y_seq: (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Container for outputs
        outputs = []

        # Loop over sequence
        for t in range(seq_len):
            x_t = x[:, t, :]          # shape: (batch_size, input_size)
            y_t, h = self.cell(x_t, h)
            outputs.append(y_t.unsqueeze(1))  # keep sequence dimension

        # Concatenate along sequence dimension
        y_seq = torch.cat(outputs, dim=1)  # shape: (batch_size, seq_len, output_size)
        return y_seq
    

# Define device and instantiate model
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"
num_threads = 2
torch.set_num_threads(num_threads)
print(f"Using {device} device with {num_threads} threads")

#Define sizes
hidden_size=100
model = DeepRNN(input_size=D, hidden_size=hidden_size, output_size=O).to(device)
print(model)





#Define optimization functions
# --- Loss and optimizer ---
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


# --- Training loop ---
def train_loop(inputs, targets, model, loss_fn, optimizer):
    model.train()
    pred = model(inputs)
    loss = loss_fn(pred, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# --- Evaluation loop ---
def eval_loop(inputs, targets, model, loss_fn):
    model.eval()
    with torch.no_grad():
        pred = model(inputs)
        loss = loss_fn(pred, targets)
    return loss.item()





#Run the optimization
#Define number of epochs 
n_epochs = 100
train_losses = np.zeros(n_epochs, dtype=float)
val_losses = np.zeros(n_epochs, dtype=float)

#Define batch size
batch_size = 1

#Define storage for losses
val_losses = np.zeros(n_epochs, dtype=float)

#Change shapes of training/validation dataset to work with RNN
RNN_train_inputs = train_inputs.reshape(batch_size, train_inputs.shape[0], D)  # shape: (batch size = 1, sequence length = N, input_size)
RNN_valid_inputs = valid_inputs.reshape(batch_size, valid_inputs.shape[0], D)
RNN_train_outputs_T = train_outputs_T.reshape(batch_size, train_inputs.shape[0], O)
RNN_valid_outputs_T = valid_outputs_T.reshape(batch_size, valid_inputs.shape[0], O)

# --- Training ---
for epoch in range(n_epochs):
    train_losses[epoch] = train_loop(RNN_train_inputs, RNN_train_outputs_T, model, loss_fn, optimizer)
    val_losses[epoch] = eval_loop(RNN_valid_inputs, valid_outputs_T, model, loss_fn)

    print(f"Epoch {epoch:03d}: train_loss={train_losses[epoch]:.5f}, val_loss={val_losses[epoch]:.5f}")

#Save model 
torch.save(model.state_dict(), model_save_path + f'{n_epochs}epochs.pth')

#Change shapes of testing dataset to work with RNN
eval_test_inputs = test_inputs.reshape(batch_size, test_inputs.shape[0], D) 
eval_test_outputs_T = test_outputs_T.reshape(batch_size, test_inputs.shape[0], O) 

# --- Testing ---
test_loss = eval_loop(eval_test_inputs, eval_test_outputs_T, model, loss_fn)
print(f"\nFinal test loss: {test_loss:.5f}")





#Diagnostic plots
# Loss curves
plt.figure(figsize=(10, 6))
plt.plot(np.arange(n_epochs), train_losses, label="Train")
plt.plot(np.arange(n_epochs), val_losses, label="Validation")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.savefig(plot_save_path+'/loss.pdf')
plt.close()

#Comparing predicted T-P profiles vs true T-P profiles with residuals
substep = 1000

#Converting tensors to numpy arrays if this isn't already done
if (type(test_outputs_T) != np.ndarray):
    test_outputs_T = test_outputs_T.detach().cpu().numpy()
    test_outputs_P = test_outputs_P.detach().cpu().numpy()

for test_idx, (test_input, test_output_T, test_output_P) in enumerate(zip(test_inputs, test_outputs_T, test_outputs_P)):

    #Retrieve prediction
    pred_output_T = model(test_input.reshape(1, 1, D)).detach().numpy()
    pred_output_T = pred_output_T.reshape(O)

    #Convert to numpy
    test_input = test_input.numpy()

    #Plotting
    if (test_idx % substep == 0):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8, 6], sharey=True, gridspec_kw = {'width_ratios':[3, 1]})
        ax1.plot(test_output_T, np.log(test_output_P/1000), '.', linestyle='-', color='blue', linewidth=2)
        ax1.plot(pred_output_T, np.log(test_output_P/1000), color='green', linewidth=2)
        ax1.invert_yaxis()
        ax1.set_ylabel(r'log$_{10}$ Pressure (bar)')
        ax1.set_xlabel('Temperature (K)')
        ax2.plot(pred_output_T - test_output_T, np.log(test_output_P/1000), '.', linestyle='-', color='green', linewidth=2)
        ax2.set_xlabel('Residuals (K)')
        plt.suptitle(rf'H$_2$O : {test_input[0]} bar, CO$_2$ : {test_input[1]} bar, LoD : {test_input[2]:.0f} days, Obliquity : {test_input[3]} deg')
        plt.tight_layout()
        plt.savefig(plot_save_path+f'/pred_vs_actual_n.{test_idx}.pdf')
        plt.close()
    
#Plotting all residuals 

#Storage
residuals = np.zeros(test_outputs_T.shape,  dtype=object)

#Converting tensors to numpy arrays if this isn't already done
if (type(test_outputs_T) != np.ndarray):
    test_outputs_T = test_outputs_T.numpy()

for test_idx, (test_input, test_output_T) in enumerate(zip(test_inputs, test_outputs_T)):

    #Retrieve prediction
    residuals[test_idx] = model(test_input.reshape(1, 1, D)).detach().numpy().reshape(O) - test_output_T


fig, ax = plt.subplots(figsize=[8, 6])
ax.plot(residuals, color='green', alpha=0.2)
ax.axhline(0, color='black', linestyle='dashed')
plt.xlabel('Output dimension')
plt.ylabel('Temperature (K)')
plt.savefig(plot_save_path+f'/residuals.pdf')
plt.close()
print(f'Median: {np.median(residuals):.3f} K, Standard deviation: {np.std(residuals):.3f} K')
