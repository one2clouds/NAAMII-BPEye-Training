from syne_tune import StoppingCriterion, Tuner 
from syne_tune.config_space import loguniform, randint
from syne_tune.experiments import load_experiment 
from syne_tune.optimizer.baselines import RandomSearch 
from syne_tune.backend.python_backend.python_backend import PythonBackend
from syne_tune import Reporter
from scipy import stats
import torch.nn as nn 
import torch 
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
import numpy as np
 
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(): 
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
        
    def loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = torch.reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')    
    

class LeNet(Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes))
        

class Trainer():
    def __init__(self, train_data, test_data, model, max_epochs, batch_size=128, gradient_clip_val = 0):
        self.train_loader = DataLoader(train_data, batch_size, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size, shuffle=False) 
        self.model = model
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val

    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm          
    def fit(self,reporter):
        self.epoch = 0
        self.optim = self.model.configure_optimizer()
        self.train_batch_idx = 0
        self.test_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch(reporter)      
                
    def fit_epoch(self, reporter):
        self.model.train()
        train_loss = []
        train_metric = []
        for images, labels in self.train_loader:
            labels_pred = self.model(images)
            labels_pred_softmax = nn.Softmax(dim=1)(labels_pred)
            # print(y_pred)
            loss = nn.CrossEntropyLoss(reduction='mean')(labels_pred_softmax, labels.to(device)) #model.loss(labels_pred_softmax, labels)    
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                # print(torch.argmax(y_pred_softmax, dim=1))
                # print(y_pred_softmax.shape)
                # print(y.cpu())
                # print(torch.argmax(y_pred_softmax, dim=1).cpu())
                metric = accuracy_score(labels, torch.argmax(labels_pred_softmax, dim=1).cpu())
                train_loss.append(loss.item())
                train_metric.append(metric)
                # gradient clipping 
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        print(f'At epoch : {self.epoch}') 
        print(f'Train Loss:{np.array(train_loss).mean()}, Train Accuracy: {np.array(train_metric).mean()}')
            
        if self.test_loader is None:
            return 
        self.model.eval()
        test_loss = []
        test_metric = []
        
        for batch in self.test_loader:
            images, labels = batch[0], batch[1]
            labels_pred = self.model(images)
            labels_pred_softmax = nn.Softmax(dim=1)(labels_pred)
            with torch.no_grad():
                loss = nn.CrossEntropyLoss()(labels_pred_softmax, labels.to(device)) #model.loss(labels_pred_softmax, labels) #
                # print(torch.argmax(y_pred_softmax, dim=1))
                # print(y)
                metric = accuracy_score(labels, torch.argmax(labels_pred_softmax, dim=1).cpu())
                test_loss.append(loss.item())
                test_metric.append(metric)
            self.test_batch_idx += 1   

        reporter(epoch=self.epoch, mnist_val_acc=np.array(test_metric).mean())

        print(f'Test Loss:{np.array(test_loss).mean()}, Test Accuracy: {np.array(test_metric).mean()}')

    def validate_all(self, data):
        self.model.eval()
        x_overall, y_overall, y_pred_overall = [], [], []
        for single_example in test_data:
            x = single_example[0]
            y = single_example[1]
            y_pred = self.model(x)
            y_pred_softmax = nn.Softmax(dim=1)(y_pred)
    
            x_overall.append(x)
            y_overall.append(y)
            y_pred_overall.append(torch.argmax(y_pred_softmax, dim=1))

        return x_overall, y_overall, y_pred_overall
    
def hpo_lenet(learning_rate, batch_size, max_epochs):
    model = LeNet(lr=learning_rate,num_classes=10)
    
    transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
    train_data = FashionMNIST(root ='./data', train=True, transform=transform, download=True)
    test_data = FashionMNIST(root ='./data', train=False, transform=transform, download=True)

    train_subset_data = torch.utils.data.Subset(train_data, range(3))
    test_subset_data = torch.utils.data.Subset(test_data, range(3))
    
    trainer = Trainer(train_subset_data, test_subset_data, model, max_epochs=3)
    reporter = Reporter()
    
    trainer.fit(reporter)



if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
    train_data = FashionMNIST(root ='./data', train=True, transform=transform, download=True)
    test_data = FashionMNIST(root ='./data', train=False, transform=transform, download=True)   


    config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),}

    initial_config = {
        "learning_rate": 0.1,
        "batch_size": 128,
    }

    trial_backend = PythonBackend(
    tune_function=hpo_lenet,
    config_space=config_space,
)
    scheduler = RandomSearch(
    config_space, 
    metric="mnist_val_acc", 
    mode="max",
    points_to_evaluate=[initial_config],
)
    max_wallclock_time = 12 * 60 # 12 mins
    n_workers = 0

    stop_criterion = StoppingCriterion(max_wallclock_time = max_wallclock_time)

    tuner = Tuner(
        trial_backend = trial_backend,
        scheduler = scheduler,
        stop_criterion = stop_criterion, 
        n_workers = n_workers, 
        print_update_interval = int(max_wallclock_time * 0.6))
    
    tuner.run()
    
