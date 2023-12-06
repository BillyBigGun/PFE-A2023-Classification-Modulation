from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray import train
import time

class NNModel(nn.Module):
    def __init__(self, input_size, output_size, learning_rate):
        super(NNModel, self).__init__()
        
        self.layers = None

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

    @abstractmethod
    def forward(self, input):
        pass # must be override

    
    def train_model(self, train_loader, epochs, patience=3):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Set the model to training mode
        self.train()

        start_time = time.time()  # Start time of the entire training

        best_val_loss = float('inf')
        epochs_no_improve = 0

        loss_per_epoch = [];

        for epoch in range(epochs):  # loop over the dataset multiple times
            epoch_start_time = time.time()  # Start time of the current epoch
            running_loss = 0.0
            mini_batch_loss = 0.0
            nb_mini_batch_print = 1000

            total = 0
            correct = 0

            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                mini_batch_loss += loss.item()

                # Calcul des prédictions correctes
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                mini_batch_loss = self.print_mini_batch(i, nb_mini_batch_print, epoch, mini_batch_loss)


            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total

            loss_per_epoch.append(epoch_loss)

            # Early stopping logic
            if epoch_loss +0.01< best_val_loss:
                best_val_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break

            epoch_end_time = time.time()  # End time of the current epoch
            print(f'Epoch {epoch + 1} completed in {epoch_end_time - epoch_start_time:.2f} seconds')
            print(f'Loss={epoch_loss}; Accuracy={epoch_accuracy}')

        end_time = time.time()  # End time of the entire training
        print(f'Finished Training in {end_time - start_time:.8f} seconds')

        return loss_per_epoch

    def train_tune_model(self, train_loader, epochs, patience=3):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Set the model to training mode
        self.train()

        start_time = time.time()  # Start time of the entire training

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):  # loop over the dataset multiple times
            epoch_start_time = time.time()  # Start time of the current epoch
            running_loss = 0.0
            mini_batch_loss = 0.0
            nb_mini_batch_print = 100

            total = 0
            correct = 0

            nb_error = 0

            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()
                try:
                    # forward + backward + optimize
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    mini_batch_loss += loss.item()

                    # Calcul des prédictions correctes
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except:
                    nb_error = nb_error +1

                
                # if i % nb_mini_batch == nb_mini_batch-1:    # print every 100 mini-batches
                #     print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, mini_batch_loss/nb_mini_batch))
                #     mini_batch_loss = 0.0
                mini_batch_loss = self.print_mini_batch(i, nb_mini_batch_print, epoch, mini_batch_loss)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total
 
            # Rapportez les métriques à Ray Tune
            train.report({'loss': epoch_loss, 'accuracy':epoch_accuracy})

            # Early stopping logic
            if epoch_loss +0.01< best_val_loss:
                best_val_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break
        
            epoch_end_time = time.time()
            print(f'Epoch {epoch + 1} completed in {epoch_end_time - epoch_start_time:.2f} seconds')

            print(f'Nb errror: {nb_error}')                    

        end_time = time.time()  # End time of the entire training
        print(f'Finished Training in {end_time - start_time:.8f} seconds')

    def print_mini_batch(self, i, nb_mini_batch, epoch, mini_batch_loss):
        if i % nb_mini_batch == nb_mini_batch-1:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, mini_batch_loss/nb_mini_batch))
            mini_batch_loss = 0.0
        return mini_batch_loss


    def predict(self, input):
        # model in evaluation mode 
        self.eval()

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            # Forward pass through the network to get the outputs
            outputs = self(input)
            
            # Convert outputs to probabilities to get the predicted class
            probabilities = torch.softmax(outputs, dim=1) # dim=1: softmax for each row of the ouput. The addition of each row needs to sum to 1
            predicted_classes = torch.max(probabilities, dim=1)[1]

        # Return the predictions
        return predicted_classes   

    def evaluate(self, train_loader):
        """
        Evaluate the model's performance on the test data.

        Args:
        test_data (Tensor): Input features for the test set.
        test_labels (Tensor): True labels for the test set.

        Returns:
        float: Accuracy of the model on the test set.
        """
        self.eval()  # Set the model to evaluation mode.

        start_time = time.time()  # Start time of the entire training
        
        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                test_input, test_labels = data
                outputs = self(test_input)
            
                # The predicted class is the one with the highest value
                _, predicted_classes = torch.max(outputs, dim=1)
                correct_counts = (predicted_classes == test_labels).sum().item()
                accuracy = correct_counts / test_labels.size(0) 

        # Switch back to train mode
        self.train()
        end_time = time.time()  # End time of the entire training
        return accuracy
        print(f'Finished evaluation in {end_time - start_time:.8f} seconds')
        
        return accuracy      