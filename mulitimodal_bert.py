import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW,RobertaTokenizer,RobertaForSequenceClassification
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score,f1_score,confusion_matrix
from torch.optim import RMSprop,SGD,Adagrad


# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
classification_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# Define the Multi-Task Emotion Model
class MultiTaskEmotionModel(torch.nn.Module):
    def __init__(self, bert_model, num_classes):
        super(MultiTaskEmotionModel, self).__init__()
        self.bert = bert_model
        self.classification_head = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        self.regression_head = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        # print(f"input_ids: {input_ids}")
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True)

        pooled_output = outputs.hidden_states[-1][:, 0, :]

        # For classification task
        # classification_logits = self.classification_head(outputs.last_hidden_state[:, 0, :])
        # classification_logits = outputs.logits
        classification_logits = self.classification_head(pooled_output)

        # print(f"classification_logits: {classification_logits}")
        # print(f"outputs: {outputs}")
        # classification_logits = self.classification_head(outputs.logits)


        # For regression task
        # regression_output = self.regression_head(outputs.last_hidden_state[:, 0, :])
        # regression_output = self.regression_head(outputs.pooler_output)
        # regression_output = self.regression_head(outputs.hidden_states[:, 0, :])
         # Forward pass for emotion intensity regression
        # sequence_output = outputs.hidden_states[-1]  # Use the last hidden state as input for regression
        # mean_hidden_state = torch.mean(sequence_output, dim=1)  # Calculate mean hidden state
        # regression_output = torch.squeeze(mean_hidden_state)  # Remove extra dimension
        # regression_output = outputs.hidden_states[-1][:, 0, :]
        regression_output = self.regression_head(pooled_output)


        # regression_output = outputs.pooler_output
        # pooled_output = outputs.hidden_states[:, 0]  # Use the [CLS] token's hidden state
        # regression_output = self.regression_head(pooled_output)




        return classification_logits, regression_output

# Function to load and preprocess the data
def load_data(file_path, tokenizer):
    texts = []
    labels_classification = []
    labels_regression = []

    with open(file_path, encoding='utf-8') as f:
        # next(f)  # Skip the header line
        # for line in f:
        #     parts = line.strip().split("\t")
        #     text, emotion, intensity = parts[1], parts[2], float(parts[3])
        #     texts.append(text)
        #     labels_classification.append(emotion)
        #     labels_regression.append(intensity)
        next(f)  # Skip the header line
        for line in f:
            # print(f"Line contents: {line}")  # Add this line to print the contents of each line
            parts = line.strip().split("\t")
            # print(f"parts[1]: {parts[1]}")
            if len(parts) >= 4:
              text, emotion, intensity =  parts[1], parts[2], round(float((parts[3])),1)
              # print(f"text: {text}")
              texts.append(text)
              labels_classification.append(emotion)
              labels_regression.append(intensity)
            # else:
              # print(f"Skipped line: {line}")  # Print if line is skipped due to missing elements


    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Use label encoding for emotion labels
    emotion_label_encoder = LabelEncoder()
    labels_classification_encoded = emotion_label_encoder.fit_transform(labels_classification)

    labels_classification = torch.tensor(labels_classification_encoded)
    # labels_regression = torch.tensor(labels_regression)
    # print(f"lables_regression:{labels_regression}")
    labels_regression = torch.tensor(labels_regression,dtype=torch.float)


    return TensorDataset(inputs.input_ids, inputs.attention_mask, labels_classification, labels_regression)

# # Load the data
train_dataset = load_data('train.txt', tokenizer)
valid_dataset = load_data('dev.txt', tokenizer)
test_dataset = load_data('test.txt', tokenizer)

# Create DataLoader objects
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the multi-task model
NUM_CLASSES = 4  # Number of emotion categories (anger, fear, joy, sadness)
multi_task_model = MultiTaskEmotionModel(classification_model, num_classes=NUM_CLASSES)
multi_task_model.to(device)

# Set up the optimizer and loss functions
optimizer = AdamW(multi_task_model.parameters(), lr=2e-5)
# optimizer = RMSprop(multi_task_model.parameters(), lr=1e-6)
# optimizer = SGD(multi_task_model.parameters(), lr=1e-8,momentum=0)
# optimizer = Adagrad(multi_task_model.parameters(), lr=1e-6)



classification_criterion = torch.nn.CrossEntropyLoss()
regression_criterion = torch.nn.MSELoss()

# Train the multi-task model
NUM_EPOCHS = 1
# Early stopping parameters
early_stopping_patience = 20  # Number of epochs to wait for improvement
best_combined_loss = float('inf')  # Initialize the best validation loss
no_improvement_count = 0  # Initialize the count for the number of epochs with no improvement
for epoch in range(NUM_EPOCHS):
    multi_task_model.train()
    total_classification_loss = 0.0
    total_regression_loss = 0.0
    all_predicted_labels = []
    all_true_labels = []
    all_predicted_outputs = []
    all_true_outputs = []


    for batch in train_loader:
        batch_input_ids, batch_attention_mask, batch_labels_classification, batch_labels_regression = batch
        batch_input_ids, batch_attention_mask, batch_labels_classification, batch_labels_regression = \
            batch_input_ids.to(device), batch_attention_mask.to(device), batch_labels_classification.to(device), batch_labels_regression.to(device)

        optimizer.zero_grad()
        classification_logits, regression_output = multi_task_model(batch_input_ids, attention_mask=batch_attention_mask)

        # Compute losses for classification and regression tasks
        classification_loss = classification_criterion(classification_logits, batch_labels_classification)
        # regression_loss = regression_criterion(regression_output, labels_regression)
        regression_loss = regression_criterion(regression_output, batch_labels_regression)


        # Total loss is a weighted combination of classification and regression losses
        total_loss = classification_loss + regression_loss


        total_loss.backward()
        optimizer.step()


        total_classification_loss += classification_loss.item()
        total_regression_loss += regression_loss.item()

        if epoch > 5:
          # Check for early stopping
          if total_loss < best_combined_loss:
              best_combined_loss = total_loss
              no_improvement_count = 0
          else:
              no_improvement_count += 1

        # If there's no improvement for the specified patience, stop training
        if no_improvement_count >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch + 1} epochs without improvement.')
            break

        ## Classification Accuracy variables
        predicted_labels = classification_logits.argmax(dim=1)
        all_predicted_labels.extend(predicted_labels.cpu().numpy())
        all_true_labels.extend(batch_labels_classification.cpu().numpy())




        # Regression Training Accuracy outputs

        # print('regression training output')
        # print(regression_output.flatten())

        predicted_outputs = regression_output.flatten().cpu()
        true_outputs = batch_labels_regression.cpu()

        all_predicted_outputs.extend(predicted_outputs)
        all_true_outputs.extend(true_outputs)


    ## Calculate Classification Accuracy
    print(f'All True outputs: {all_true_outputs}')
    print(f'All Predicted outputs: {all_predicted_outputs}')
    classification_accuracy = f1_score(all_true_labels, all_predicted_labels,average='weighted')
    ## Calculate Regression Accuracy using R2
    # print(f'All True Outputs :{all_true_outputs.detach().cpu().numpy()}')

    regression_accuracy = r2_score([item.detach().cpu().numpy() for item in all_true_outputs], [item.detach().cpu().numpy() for item in all_predicted_outputs])
    # regression_accuracy = r2_score(all_true_outputs, all_predicted_outputs)

    ## Training confusion Metrics
    # Calculate the confusion matrix
    confusion = confusion_matrix(all_true_labels, all_predicted_labels)

    # # Extract values from the confusion matrix
    # tn, fp, fn, tp = confusion.ravel()

    # # Print false positives and false negatives
    # print(f'False Positives (FP): {fp}')
    # print(f'False Negatives (FN): {fn}')

    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, '
          f'Classification Loss: {total_classification_loss / len(train_loader):.3f}, '
          f'Regression Loss: {total_regression_loss / len(train_loader):.3f}')

    print(f'Clasification Training F1 Score: {classification_accuracy}')
    print(f'Regression Training R2 Score {regression_accuracy}')
    class_loss = total_classification_loss / len(train_loader)
    reg_loss = total_regression_loss / len(train_loader)

# Function for evaluation
def evaluate_model(model, data_loader, classification_criterion, regression_criterion):
    model.eval()
    total_classification_loss = 0.0
    total_regression_loss = 0.0
    all_predicted_labels = []
    all_true_labels = []
    all_predicted_outputs = []
    all_true_outputs = []

    with torch.no_grad():
      for batch in data_loader:
          batch_input_ids, batch_attention_mask, batch_labels_classification, batch_labels_regression = batch
          batch_input_ids, batch_attention_mask, batch_labels_classification, batch_labels_regression = \
              batch_input_ids.to(device), batch_attention_mask.to(device), batch_labels_classification.to(device), batch_labels_regression.to(device)


          # classification_logits, regression_output = model(input_ids, attention_mask=attention_mask)
          classification_logits, regression_output = model(batch_input_ids, attention_mask=batch_attention_mask)

          # print(f"regression output:{regression_output}")

          ## Classification Accuracy
          predicted_labels = classification_logits.argmax(dim=1)

          all_predicted_labels.extend(predicted_labels.cpu().numpy())
          all_true_labels.extend(batch_labels_classification.cpu().numpy())

          ## Regression Accuracy Variables
          # predicted_outputs = regression_output.squeeze().cpu().numpy()
          # predicted_outputs = regression_output.squeeze()

          predicted_outputs = regression_output.flatten().cpu()


          true_outputs = batch_labels_regression.cpu()
          # true_outputs = labels_regression.cpu()

          all_predicted_outputs.extend(predicted_outputs)
          all_true_outputs.extend(true_outputs)



          classification_loss = classification_criterion(classification_logits, batch_labels_classification)
          # regression_loss = regression_criterion(regression_output.squeeze(), labels_regression)
          # regression_loss = regression_criterion(regression_output.squeeze(), labels_regression)
          regression_loss = regression_criterion(regression_output.squeeze(),batch_labels_regression)



          total_classification_loss += classification_loss.item()
          total_regression_loss += regression_loss.item()

    #Classification Accuracy
    classification_accuracy = accuracy_score(all_true_labels, all_predicted_labels)

    # print(f"all_true_labels.shape: {all_true_labels}")
    # print(f"all_predicted_labels.shape: {all_predicted_labels}")


    # print(f"all_true_outputs.shape: {all_true_outputs}")
    # print(f"all_predicted_outputs.shape: {all_predicted_outputs}")

    ## Regression Accuracy
    # regression_accuracy = mean_squared_error(all_true_outputs, all_predicted_outputs)

    # print(f"labels_regression:{labels_regression}")
    # regression_accuracy = mean_squared_error(all_predicted_outputs, all_true_outputs )
    # Calculate the R-squared (R2) score
    regression_accuracy = r2_score(all_true_outputs, all_predicted_outputs)




    avg_classification_loss = total_classification_loss / len(data_loader)
    avg_regression_loss = total_regression_loss / len(data_loader)

    return avg_classification_loss, avg_regression_loss, classification_accuracy,regression_accuracy

# # Evaluate the model on the validation set
valid_classification_loss, valid_regression_loss, valid_classification_accuracy, valid_regression_accuracy = evaluate_model(multi_task_model, valid_loader,
                                                                   classification_criterion, regression_criterion)
print(f'Validation Classification Loss: {valid_classification_loss:.3f}, '
      f'Validation Regression Loss: {valid_regression_loss:.3f}')
print(f'Validation Classification Accuracy: {valid_classification_accuracy:.3f}')
print(f'Validation Regression Accuracy: {valid_regression_accuracy:.3f}')



# Evaluate the model on the test set
test_classification_loss, test_regression_loss,test_classification_accuracy,test_regression_accuracy = evaluate_model(multi_task_model, test_loader,
                                                                 classification_criterion, regression_criterion)



print(f'Test Classification Loss: {test_classification_loss:.3f}, '
      f'Test Regression Loss: {test_regression_loss:.3f}')
print(f'Test Regression Accuracy: {test_regression_accuracy:.3f}')
print(f'Test Classification Accuracy: {test_classification_accuracy:.3f}')


#Save this Model
#torch.save(multi_task_model.state_dict(), 'my_custom_EmoRoBERTa_multi_model.pth')

# Use the trained model to predict custom input
tokenized_custom = tokenizer('What a wonderful and amazing day today', padding=True, truncation=True, return_tensors='pt')
custom_input_ids = tokenized_custom['input_ids']
custom_attention_mask = tokenized_custom['attention_mask']

tokenized_custom1 = tokenizer('I was shocked to hear the news about the war', padding=True, truncation=True, return_tensors='pt')
custom_input_ids1 = tokenized_custom1['input_ids']
custom_attention_mask1 = tokenized_custom1['attention_mask']


# classification_model.eval()

# outputs = multi_task_model(custom_input_ids, attention_mask=custom_attention_mask,output_hidden_states=True)

# pooled_output = outputs.hidden_states[-1][:, 0, :]


# # Pass custom hidden states through the regression head
# custom_predictions = MultiTaskEmotionModel(pooled_out)  # Use the CLS token representation

# # Flatten the predictions if needed
# custom_predictions = custom_predictions.view(-1)

# # Convert the tensor to a Python float
# custom_predicted_value = custom_predictions.item()

custom_input_ids = custom_input_ids.to(device)
custom_attention_mask = custom_attention_mask.to(device)

custom_input_ids1 = custom_input_ids1.to(device)
custom_attention_mask1 = custom_attention_mask1.to(device)

# Forward pass
with torch.no_grad():
    classification_logits, regression_output = multi_task_model(custom_input_ids, attention_mask=custom_attention_mask)
    classification_logits1, regression_output1 = multi_task_model(custom_input_ids1, attention_mask=custom_attention_mask1)



# Decode the classification output (if needed)
predicted_emotion_labels =  classification_logits.argmax(dim=1)


# Get the regression output
predicted_intensity_values = regression_output.cpu().numpy()


print(f"predicted_emotion_labels:{predicted_emotion_labels}")
print(f"predicted_intensity_values:{predicted_intensity_values}")

# Decode the classification output (if needed)
predicted_emotion_labels1 =  classification_logits1.argmax(dim=1)


# Get the regression output
predicted_intensity_values1 = regression_output1.cpu().numpy()


print(f"predicted_emotion_labels:{predicted_emotion_labels1}")
print(f"predicted_intensity_values:{predicted_intensity_values1}")

results_df['MultiModal'] = [total_classification_loss / len(train_loader),total_regression_loss / len(train_loader),classification_accuracy,valid_classification_accuracy,test_classification_accuracy,regression_accuracy,test_regression_accuracy,valid_regression_accuracy]
results_df


