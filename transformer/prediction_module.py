import numpy as np
from pytorch_lightning import LightningModule
import os 
from torch import nn
from torchmetrics import AUROC, F1Score
import torch
from sklearn.metrics import roc_auc_score

class DecompensationPrediction(LightningModule): #predicting dim-1 outputs

    def __init__(self, model, config=None):
        super().__init__()
        self.model = model
        self.gamma = config["gamma"]
        self.momentum = config["momentum"]
        self.minlr = config["minlr"]
        self.maxlr = config["maxlr"]
        self.step = config["step"]
        self.nesterov = config["nesterov"]
        self.wd = config["wd"]

    def training_step(self, batch, batch_idx):
        loss, out = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_loss": loss}
        self.log_dict(metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, out, F1, AUROC = self._shared_eval_step(batch, batch_idx, test=True)
        metrics = {"test_loss": loss, "F1":F1, "AUROC":AUROC}
        self.log_dict(metrics)
        if batch_idx == 0:
            print("Output Predictions for Test Set :", out)
        return {'out':out, 'label':batch['y'].view(-1, 1)}
    
    def test_epoch_end(self, outputs):
        # do something with the outputs of all test batches
        allpreds = []
        alllabels = []
        allvalues = []
        alltargets = [] 
        allout = []
        alllabel = []
        for x in outputs:
            num_batch = x['label'].shape[0]
            for i in range(num_batch):
                allout.append(x['out'][i].detach().cpu().numpy()[0])
                alllabel.append(x['label'][i].detach().cpu().numpy()[0])
        
        allout = np.array(allout)
        alllabel = np.array(alllabel)
        out_cpu = torch.tensor(allout, dtype=torch.float).view(-1, 1)
        label_cpu = torch.tensor(alllabel, dtype=torch.float).view(-1, 1)
        
        preds = torch.tensor(np.where(out_cpu >= 0.75, 1, 0), dtype=torch.int).view(-1, 1)
        targets = torch.tensor(np.where(label_cpu == 1, 1, 0), dtype=torch.int).view(-1, 1)
        loss = self.loss_fn(out_cpu,label_cpu)
        f1score = F1Score()
        AUROC_val = roc_auc_score(targets, out_cpu) 
        f1val = f1score(preds, targets)
        metrics = {"test_loss": loss, "F1":f1val, "AUROC":AUROC_val}
        self.log_dict(metrics)
        
    def _shared_eval_step(self, batch, batch_idx, test=False):
        # extract baseline patient features from batch
        x, x_wide, y = batch["x"], batch["x_wide"], batch["y"]
        
        vals = x.view(-1, x.shape[1], x.shape[-1])
        label = y.view(-1, 1) 
        wide_feat = x_wide.view(-1, np.shape(x_wide)[1])
        
        out = self.model(vals, wide_feat).view(-1, 1)
        loss = self.loss_fn(out, label)
        
        if test:
            out_cpu = out.detach().cpu()
            label_cpu = label.detach().cpu()
            auroc = AUROC(pos_label=1)
            f1score = F1Score()
            preds = torch.tensor(np.where(out_cpu >= 0.75, 1, 0), dtype=torch.int).view(-1, 1)
            targets = torch.tensor(np.where(label_cpu == 1, 1, 0), dtype=torch.int).view(-1, 1)
            AUROC_val = auroc(out_cpu, targets)
            f1val = f1score(preds, targets)
            return loss, out, f1val, AUROC_val
        else:
            return loss, out

    def loss_fn(self,out,target): #MAE
        return nn.BCELoss()(out.view(-1, 1), target.view(-1, 1))

    # With Learning Rate Decay / Reduce LR on Plateau
    def configure_optimizers(self):
        #after every 10 epochs, decay LR by gamma (0.5)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.maxlr, momentum=self.momentum, weight_decay = self.wd, nesterov=self.nesterov) # try SGD
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='exp_range', 
                                                         gamma = self.gamma, base_lr=self.minlr, max_lr=self.maxlr, 
                                                         step_size_up = self.step)
        return [optimizer], [lr_scheduler]
