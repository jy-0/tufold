import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, List, Dict
from math import sqrt


def loop_train(model: nn.Module, data_loader: DataLoader, 
               loss_fn: Callable[..., torch.Tensor], optimizer: torch.optim.Optimizer,
               log_interval=20, use_new_approach=True) -> Dict[str, List]:
    total_steps = len(data_loader)
    model.train()
    device = next(model.parameters()).device
    avg_loss = 0.0

    # logging
    log_dict = {'step': [], 'current_loss': [], 'avg_loss': []}

    for step, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)

        # discard
        if str(device) == 'mps:0':
            # discard
            pred = torch.log_softmax(pred, dim=1)
            loss = -pred[[i for i in range(y.size(0)) for _ in range(y.size(1))], y.reshape(y.size(0) * y.size(1)), [j for _ in range(y.size(0)) for j in range(y.size(1))]]
            loss = loss.reshape(y.size(0), y.size(1))
        else:
            loss = loss_fn(pred, y)

        if use_new_approach:
            diag_elements = (y == torch.arange(y.size(1)).to(device))
            diag_num_per_batch = diag_elements.sum()
            not_diag_num_per_batch = y.size(0) * y.size(1) - diag_num_per_batch
            weight_for_diag_per_batch = not_diag_num_per_batch / diag_num_per_batch
            loss[diag_elements] *= weight_for_diag_per_batch

            loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        avg_loss += current_loss
        if step % log_interval == 0:
            print(f'Step {step + 1:>6d}/{total_steps:>6d}, current loss: {current_loss:>10.6f}, '
                  f'avg loss: {avg_loss / log_interval if step > 0 else avg_loss:>10.6f}')
            
            # logging
            log_dict['step'].append(step)
            log_dict['current_loss'].append(current_loss)
            log_dict['avg_loss'].append(avg_loss / log_interval if step > 0 else avg_loss)

            avg_loss = 0.0
    
    return log_dict


def loop_valid_test(model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    model.eval()
    sample_num = len(data_loader.dataset)
    p_sum, r_sum, f1_sum = 0.0, 0.0, 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            tp = torch.sum(pred * y, dim=(1, 2))
            fp = torch.sum(pred * (1 - y), dim=(1, 2))
            fn = torch.sum((1 - pred) * y, dim=(1, 2))
            tn = torch.sum((1 - pred) * (1 - y), dim=(1, 2))
            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            p_sum += torch.sum(p).item()
            r_sum += torch.sum(r).item()
            f1_sum += torch.sum(f1).item()
    print(f'P: {p_sum / sample_num:>10.6f}, R: {r_sum / sample_num:>10.6f}, F1: {f1_sum / sample_num:>10.6f}')

    # logging
    return {'P': p_sum / sample_num, 'R': r_sum / sample_num, 'F1': f1_sum / sample_num}


def loop_valid_test_list_results(model: nn.Module, data_loader: DataLoader):
    model.eval()
    sample_num = len(data_loader.dataset)
    p_sum, r_sum, f1_sum = [], [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            tp = torch.sum(pred * y, dim=(1, 2))
            fp = torch.sum(pred * (1 - y), dim=(1, 2))
            fn = torch.sum((1 - pred) * y, dim=(1, 2))
            tn = torch.sum((1 - pred) * (1 - y), dim=(1, 2))
            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            p_sum.extend(p.cpu().numpy())
            r_sum.extend(r.cpu().numpy())
            f1_sum.extend(f1.cpu().numpy())
    return p_sum, r_sum, f1_sum


def loop_valid_test_single(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    sample_num = x.size(0)
    p_sum, r_sum, f1_sum = 0.0, 0.0, 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        x, y = x.to(device), y.to(device)
        pred = model(x)

        tp = torch.sum(pred * y, dim=(1, 2))
        fp = torch.sum(pred * (1 - y), dim=(1, 2))
        fn = torch.sum((1 - pred) * y, dim=(1, 2))
        tn = torch.sum((1 - pred) * (1 - y), dim=(1, 2))
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        p_sum += torch.sum(p).item()
        r_sum += torch.sum(r).item()
        f1_sum += torch.sum(f1).item()
    print(f'P: {p_sum / sample_num:>10.6f}, R: {r_sum / sample_num:>10.6f}, F1: {f1_sum / sample_num:>10.6f}')

    # logging
    return {'P': p_sum / sample_num, 'R': r_sum / sample_num, 'F1': f1_sum / sample_num}
