#!coding:utf-8
import torch
from torch.nn import functional as F

from ramps import exp_rampup

class Trainer:

    def __init__(self, model, ema_model, optimizer, device):
        self.model      = model
        self.ema_model  = ema_model
        self.optimizer  = optimizer
        self.ce_loss    = torch.nn.BCEWithLogitsLoss()
        self.usp_weight = 30.0
        self.ema_decay  = 0.97
        self.rampup     = exp_rampup(30)
        self.device     = device
        self.global_step= 0

    def cons_loss(self, logit1, logit2):
        assert logit1.size() == logit2.size()
        return F.mse_loss(logit1, logit2)

    def train_iteration(self, data_loader_labeled, data_loader_unlabeled):

        # === training with label ===
        for x, y in data_loader_labeled:
            stduent_input = x.to(self.device)
            teacher_input = x.to(self.device)
            targets = y.to(self.device)
            self.global_step = self.global_step + 1

            # === forward ===
            outputs = self.model(stduent_input)
            loss = self.ce_loss(outputs, targets)
            print("labeled_loss: %0.3f" % loss.item())
            
            # === Semi-supervised Training ===
            self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)
            # consistency loss
            with torch.no_grad():
                ema_outputs = self.ema_model(teacher_input)
                ema_outputs = ema_outputs.detach()
            cons_loss  = self.cons_loss(outputs, ema_outputs)
            cons_loss *= self.rampup(self.epoch)*self.usp_weight
            loss += cons_loss
            print("consistent_loss: %0.3f" % cons_loss.item())

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # === training without label ===
        for x in data_loader_unlabeled:
            stduent_input = x.to(self.device)
            teacher_input = x.to(self.device)

            # === forward ===
            outputs = self.model(stduent_input)

            # === Semi-supervised Training ===
            self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)
            with torch.no_grad():
                ema_outputs = self.ema_model(teacher_input)
                ema_outputs = ema_outputs.detach()
            # === consistency loss ===
            cons_loss  = self.cons_loss(outputs, ema_outputs)
            cons_loss *= self.rampup(self.epoch)*self.usp_weight
            print("unlabeled_consistent_loss: %0.3f" % cons_loss.item())

            # backward
            self.optimizer.zero_grad()
            cons_loss.backward()
            self.optimizer.step()
        return self.model, self.ema_model

    def train(self, data_loader_labeled, data_loader_unlabeled):
        self.model.train()
        self.ema_model.train()
        with torch.enable_grad():
            return self.train_iteration(data_loader_labeled, data_loader_unlabeled)

    def test(self, model, ema_model, stu_ckpt, t_ckpt, test_data):
        step = 1
        model.load_state_dict(torch.load(stu_ckpt, map_location='cpu'))
        ema_model.load_state_dict(torch.load(t_ckpt, map_location='cpu'))
        for x, y in test_data:
            print("----- img %d -----" % step)
            stduent_input = x.to(self.device)
            teacher_input = x.to(self.device)
            targets = y.to(self.device)
            outputs = self.model(stduent_input)
            student_test_loss = self.ce_loss(outputs, targets)
            print("student_test_loss: %0.3f" % student_test_loss.item())
            outputs = self.ema_model(teacher_input)
            teacher_test_loss = self.ce_loss(outputs, targets)
            print("teacher_test_loss: %0.3f" % teacher_test_loss.item())
            step = step + 1

    def loop_train(self, epochs, train_data_labeled, train_data_unlabeled, scheduler=None):
        for ep in range(epochs):
            self.epoch = ep
            print("------ Training epochs: {} ------".format(ep))
            model, ema_model = self.train(train_data_labeled, train_data_unlabeled)
            if scheduler is not None:
                scheduler.step()
            torch.save(model.state_dict(), 'student_weights_%d.pth' % ep)
            torch.save(ema_model.state_dict(), 'teacher_weights_%d.pth' % ep)
            # save model
        print("Model is saved!")

    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
