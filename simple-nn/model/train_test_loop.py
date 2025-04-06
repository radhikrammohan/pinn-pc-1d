from itertools import zip_longest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim import Adam
import copy

# check for gpu
if torch.backends.mps.is_available():
    print("MPS is available")
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

print('Using device:', device)

def training_loop(epochs, model, \
                  optimizer, \
                  train_dataloader, val_dataloader):
    
    # Initialize the lists to store losses
    train_losses = []
    test_losses = []
    
    
    
    model.to(device)  # Move the model to the GPU
    
    for epoch in range(epochs):
        
        model.train()  # Set the model to training mode
        train_loss = 0  # Initialize the training loss for this epoch
        
        
        
        data_loss_t = 0  # Data loss accumulator
        


        # Loop through the training data loaders
        for batch in train_dataloader:
            
            # print(len(train_dataloader))
            # print(len(train_loader_pde))
            # print(len(train_loader_init))
            # print(len(train_loader_bc_l))
            # print(len(train_loader_bc_r))
            if batch is None:
                continue  # Skip this iteration
            # Extract inputs from each batch
            inputs, temp_inp = batch  # Move inputs to GPU
              # Move inputs to GPU
           
            # Move all tensors to the GPU
            inputs = inputs.to(device)
            temp_inp = temp_inp.to(device)
            
            
            optimizer.zero_grad()  # Zero the gradients before backpropagation
            
            # Forward pass for data prediction
            u_pred_d = model(inputs)
            loss_fn_data = nn.MSELoss()
            temp_inp = temp_inp.view(-1, 1)  # Reshape temp_inp to match u_pred_d
            data_loss = loss_fn_data(u_pred_d,temp_inp) # Data loss
        
            loss =  data_loss
            
            # Backpropagation
            loss.backward(retain_graph=True)  # Backpropagate the gradients
            
            optimizer.step()  # Update the weights
            
             
            # Accumulate losses for tracking
            train_loss += loss.item()
        
        train_loss_batch = train_loss / len(train_dataloader)
        # Append losses to respective lists for tracking
        if len(train_dataloader) > 0:
            train_losses.append(train_loss_batch)
            
        # Set model to evaluation mode for testing
        model.eval()
        test_loss = 0
        
        # Evaluate on test data without gradient calculation
        for batch in val_dataloader:
            
            if batch is None:
               continue  # Skip this iteration
            
            inputs, temp_inp = batch  # Move inputs to GPU
              # Move inputs to GPU
            
            # Move all tensors to the GPU
            inputs = inputs.to(device)
            temp_inp = temp_inp.to(device)
            
        
            u_pred = model(inputs)
            loss_fn_data = nn.MSELoss()
            temp_inp = temp_inp.view(-1, 1)  # Reshape temp_inp to match u_pred
            data_loss_t = loss_fn_data(u_pred, temp_inp)
            
            loss_t =  data_loss_t
            # loss_t = w1 * phy_loss_t + w2 * init_loss_t + w3 * bc_loss_t
            
            test_loss += loss_t.item()
            
        test_loss_batch = test_loss / len(val_dataloader)
        # Normalize the test loss by the number of test batches
        if len(val_dataloader) > 0:
            test_losses.append(test_loss_batch)
        
         # Saving the best model in the training loop
        if epoch == 0:
            best_loss = test_loss / len(val_dataloader)
            best_model = copy.deepcopy(model)
        else:
            current_loss = test_loss / len(val_dataloader)
            if current_loss < best_loss:
                best_loss = current_loss
                best_model = copy.deepcopy(model)
        # Empty CUDA cache to free memory
        torch.cuda.empty_cache()

        # Get the train losses in an array  format
        loss_train = {"train-loss": train_losses}
        
        loss_test = {"test-loss": test_losses}

        # save this losses to a csv file

        
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f" ")
            print(f"--"*50)
            print(f"| Epoch {epoch+1},            | Training-Loss {train_loss_batch:.4e},| Test-Loss {test_loss_batch:.4e}   |")
            print(f"--"*50)
            
            print(f" ")

        if epoch == (epochs-1):
            print(f" ")
            print(f"--"*50)
            print(f"| Epoch {epoch+1},            | Training-Loss {train_loss_batch:.4e},| Test-Loss {test_loss_batch:.4e}   |")
            print(f"--"*50)
            
            print(f" ")
    # Return all collected losses for further analysis
    # return train_losses, test_losses, pde_losses, bc_losses, ic_losses, data_losses
    return loss_train, loss_test, best_model


