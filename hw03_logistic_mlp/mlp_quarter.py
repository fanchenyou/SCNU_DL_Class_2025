import torch
from sklearn.model_selection import train_test_split
from get_quarter import train, get_quarter


class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(MLPModel, self).__init__()
        # TODO: implement your 2-layer MLP here
        self.mlp_1 = None
        self.mlp_2 = None

    def forward(self, x):
        # TODO: Implement forward function here
        outputs = None
        return outputs


if __name__ == '__main__':

    ###############################
    ####     DO NOT MODIFY    #####
    ###############################
    input_dim = 2  # Two inputs x1 and x2
    output_dim = 1  # The output is 0/1 binary classification prob.
    X, y = get_quarter()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    
    
    # TODO: you may modify hidden_size and learning_rate
    hidden_size = 2
    # TODO: implement the MLPModel
    model = MLPModel(input_dim, hidden_size, output_dim)
    # TODO: add a proper optimizer (SGD/Adam/Muon), tune the learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    
    train(X_train, X_test, y_train, y_test, optimizer=optimizer, model=model)

    for k, v in model.state_dict().items():
        print(k, v)
