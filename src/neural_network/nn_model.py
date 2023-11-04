from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

class NNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, activation=F.relu):
        super(NNModel, self).__init__()
        
        self.layers = None

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.learning_rate = learning_rate

    @abstractmethod
    def forward(self, input):
        pass # must be override

    def train_model(self, train_loader, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Set the model to training mode
        self.train()

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0

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
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def predict(self, input):
        # model in evaluation mode 
        self.eval()

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            # Forward pass through the network to get the outputs
            outputs = self(input)
            
            # Convert outputs to probabilities to get the predicted class
            probabilities = torch.softmax(outputs, dim=1) # dim=1 softmax for each row of the ouput. It needs to sum to 1
            predicted_classes = torch.max(probabilities, dim=1)[1]

        # Return the predictions
        return predicted_classes   

    def evaluate(self, test_data, test_labels):
        """
        Evaluate the model's performance on the test data.

        Args:
        test_data (Tensor): Input features for the test set.
        test_labels (Tensor): True labels for the test set.

        Returns:
        float: Accuracy of the model on the test set.
        """
        self.eval()  # Set the model to evaluation mode.
        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            outputs = self(test_data)
            
            # The predicted class is the one with the highest value
            _, predicted_classes = torch.max(outputs, dim=1)
            correct_counts = (predicted_classes == test_labels).sum().item()
            accuracy = correct_counts / test_labels.size(0) 

        # Switch back to train mode
        self.train()
        return accuracy      
