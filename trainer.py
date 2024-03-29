import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from t_SNE import *
import os


def train_model(model, criterion, optimizer, dataloaders, metric_fc, is_softmax=False, num_epochs=10):
    """
    Trains a Deep Learning Model for Facial Recognition
    Args:
        model: Model object
            Deep Learning Model to be trained
        criterion: torch.nn.CrossEntropyLoss
        optimizer: torch.optim.Adam or torch.optim.SGD
        dataloaders: dictionary
            dictionary of dataloaders with keys "train" and "val"
        metric_fc: ArcFace object
            ArcFace layer implementation as class
        is_softmax: Boolean
            False means use ArcFace and True means use softmax only
        num_epochs: int
            No of epochs to train the model
    Returns:
        train_losses: list
            train loss at each epoch
        val_losses: list
            validation loss at each epoch
        save_dir: str
            directory where to store results
        dictionary of validation metrics i.e., accuracy and f1 score at each epoch


    """
    save_dir = "./Visualization/Cross_Entropy" if is_softmax else "./Visualization/ArcFace"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1scores = []
    for epoch in range(num_epochs):
        total_loss = []
        valid_loss = []
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for inputs, labels in dataloaders["train"]:
            model.train()
            # zero the parameter gradients
            optimizer.zero_grad()
            # get the class logits
            logits = model(inputs)
            # Pass through the ArcFace layer if needed
            if not is_softmax:
                logits = metric_fc(logits, labels)
            _, preds = torch.max(logits, dim=1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        avg_train_loss = sum(total_loss) / len(total_loss)
        train_losses.append(avg_train_loss)
        # Validation
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders["val"]):
                embedding = model(inputs)
                if not is_softmax:
                    # Compute cosine similarity for ArcFace
                    logits = metric_fc(embedding, labels)
                    weights = metric_fc.get_weights()
                    cos_sim = torch.tensor(cosine_similarity(embedding, weights))
                else:
                    logits = embedding
                loss = criterion(logits, labels)
                if i == 0:
                    # first batch
                    total_labels = labels
                    if not is_softmax:
                        _, total_pred = torch.max(cos_sim, dim=1)
                    else:
                        _, total_pred = torch.max(logits, dim=1)
                    total_embedding = embedding
                else:
                    # concatenate results to preceding batches
                    total_labels = torch.cat((total_labels, labels), dim=0)

                    if not is_softmax:
                        _, y_pred = torch.max(cos_sim, dim=1)
                    else:
                        _, y_pred = torch.max(logits, dim=1)
                    total_pred = torch.cat((total_pred, y_pred), dim=0)
                    total_embedding = torch.cat((total_embedding, embedding), dim=0)
                valid_loss.append(loss)
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        val_losses.append(avg_valid_loss)
        val_acc = accuracy_score(y_true=total_labels, y_pred=total_pred)
        valid_f1 = f1_score(y_true=total_labels, y_pred=total_pred, average="macro")
        val_accuracies.append(val_acc * 100)
        val_f1scores.append(valid_f1)
        computeTSNEProjectionOfLatentSpace(total_embedding, total_labels, path_of_visualizations=save_dir,
                                           epoch=epoch + 1)
        print("Epoch No: {}, Train Loss: {}, Valid Loss: {}".format(epoch + 1, avg_train_loss, avg_valid_loss))
        print("\t Valid Acc: {:.2f}%, Valid F1: {}".format(val_acc * 100.0, valid_f1))
    return train_losses, val_losses, save_dir, {"acc": val_accuracies, "f1": val_f1scores}
