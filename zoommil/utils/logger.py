import importlib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from sksurv.metrics import concordance_index_censored
import torch

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

class TBLogger(object):
    def __init__(self, log_dir=None):
        super(TBLogger, self).__init__()
        self.log_dir = log_dir
        tb_module = importlib.import_module("torch.utils.tensorboard")
        self.tb_logger = getattr(tb_module, "SummaryWriter")(log_dir=self.log_dir)
    
    def end(self):
        self.tb_logger.flush()
        self.tb_logger.close()
    
    def run(self, func_name, *args, mode="tb", **kwargs):
        if func_name == "log_scalars":
            return self.tb_log_scalars(*args, **kwargs)
        else:
            tb_log_func = getattr(self.tb_logger, func_name)
            return tb_log_func(*args, **kwargs)
        return None

    def tb_log_scalars(self, metric_dict, step):
        for k, v in metric_dict.items():
            self.tb_logger.add_scalar(k, v, step)

class MetricLoggerClassification(object):
    def __init__(self):
        super(MetricLoggerClassification, self).__init__()
        self.y_pred = []
        self.y_true = []

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.y_pred.append(Y_hat)
        self.y_true.append(Y)

    def get_summary(self):
        acc = accuracy_score(y_true=self.y_true, y_pred=self.y_pred) # accuracy
        f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average=None) # f1 score
        weighted_f1 = f1_score(y_true=self.y_true, y_pred=self.y_pred, average='weighted') # weighted f1 score
        kappa = cohen_kappa_score(y1=self.y_true, y2=self.y_pred, weights='quadratic') # cohen's kappa

        print('*** Metrics ***')
        print('* Accuracy: {}'.format(acc))
        for i in range(len(f1)):
            print('* Class {} f1-score: {}'.format(i, f1[i]))
        print('* Weighted f1-score: {}'.format(weighted_f1))
        print('* Kappa score: {}'.format(kappa))
        
        summary = {'accuracy': acc, 'weighted_f1': weighted_f1,'kappa': kappa}
        for i in range(len(f1)):
            summary[f'class_{i}_f1'] = f1[i]
        return summary

    def get_confusion_matrix(self):
        cf_matrix = confusion_matrix(np.array(self.y_true), np.array(self.y_pred)) # confusion matrix
        return cf_matrix


class MetricLoggerSurvival(object):
    """Tracks stats for c-index."""
    def __init__(self):
        super(MetricLoggerSurvival, self).__init__()
        self.all_censorships = []
        self.all_event_times = []
        self.all_risk_scores = []

    def log(self, survival, censors, hazards):
        # Track stats for c-index
        survival_pred = torch.cumprod(1 - hazards, dim=1)
        risk_pred = -torch.sum(survival_pred, dim=1)
        self.all_censorships.append(censors.detach().cpu().numpy())
        self.all_event_times.append(survival.detach().cpu().numpy())
        self.all_risk_scores.append(risk_pred.detach().cpu().numpy())

    def get_summary(self):
        all_censorships = (1 - np.concatenate(self.all_censorships)).astype(np.bool_)
        all_event_times = np.concatenate(self.all_event_times)
        all_risk_scores = np.concatenate(self.all_risk_scores)

        if np.sum(all_censorships).item() <= 1:
            print("Warning: all events censored")
            c_index = 0.5
        else:
            c_index = concordance_index_censored(all_censorships, all_event_times, all_risk_scores)[0]

        print('*** Metrics ***')
        print('* C-index: {}'.format(c_index))

        summary = {'c-index': c_index}
        return summary







