# general
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
import tool

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class Coverage:
    def __init__(self, model, layer_size_dict, hyper=None, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.layer_size_dict = layer_size_dict
        self.init_variable(hyper, **kwargs)

    def init_variable(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def coverage(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building is not needed.')

    def assess(self, data_loader):
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.step(data)
            print(self.current)

    def step(self, data):
        cove_dict = self.calculate(data)
        gain = self.gain(cove_dict)
        if gain is not None:
            self.update(cove_dict, gain)

    def update(self, all_cove_dict, delta=None):
        self.coverage_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    def gain(self, cove_dict_new):
        new_rate = self.coverage(cove_dict_new)
        return new_rate - self.current


class NLC(Coverage):
    def init_variable(self, hyper=None):
        assert hyper is None, 'NLC has no hyper-parameter'
        self.estimator_dict = {}
        self.current = 1
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = tool.Estimator(feature_num=layer_size[0])

    def calculate(self, data):
        stat_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            info_dict = self.estimator_dict[layer_name].calculate(layer_output.to(self.device))
            stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])
        return stat_dict

    def update(self, stat_dict, gain=None):
        if gain is None:
            for i, layer_name in enumerate(stat_dict.keys()):
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current = self.coverage(self.estimator_dict)
        else:
            (delta, layer_to_update) = gain
            for layer_name in layer_to_update:
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current += delta

    def coverage(self, stat_dict):
        val = 0
        for i, layer_name in enumerate(stat_dict.keys()):
            # Ave = stat_dict[layer_name].Ave
            CoVariance = stat_dict[layer_name].CoVariance
            # Amount = stat_dict[layer_name].Amount
            val += self.norm(CoVariance)
        return val

    def gain(self, stat_new):
        total = 0
        layer_to_update = []
        for i, layer_name in enumerate(stat_new.keys()):
            (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
            value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance)
            if value > 0:
                layer_to_update.append(layer_name)
                total += value
        if total > 0:
            return (total, layer_to_update)
        else:
            return None

    def norm(self, vec, mode='L1', reduction='mean'):
        m = vec.size(0)
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = vec.abs().sum()
        elif mode == 'L2':
            total = vec.pow(2).sum().sqrt()
        if reduction == 'mean':
            return total / m
        elif reduction == 'sum':
            return total

    def save(self, path):
        print('Saving recorded NLC in %s...' % path)
        stat_dict = {}
        for layer_name in self.estimator_dict.keys():
            stat_dict[layer_name] = {
                'Ave': self.estimator_dict[layer_name].Ave,
                'CoVariance': self.estimator_dict[layer_name].CoVariance,
                'Amount': self.estimator_dict[layer_name].Amount
            }
        torch.save({'stat': stat_dict}, path)

    def load(self, path):
        print('Loading saved NLC from %s...' % path)
        ckpt = torch.load(path)
        stat_dict = ckpt['stat']
        for layer_name in stat_dict.keys():
            self.estimator_dict[layer_name].Ave = stat_dict[layer_name]['Ave']
            self.estimator_dict[layer_name].CoVariance = stat_dict[layer_name]['CoVariance']
            self.estimator_dict[layer_name].Amount = stat_dict[layer_name]['Amount']


class NC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.threshold = hyper
        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.coverage_dict[layer_name] = torch.zeros(layer_size[0]).type(torch.BoolTensor).to(self.device)
        self.current = 0

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            scaled_output = tool.scale(layer_output)
            mask_index = scaled_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return torch.true_divide(cove, total).item()

    def save(self, path):
        print('Saving recorded NC in %s...' % path)
        torch.save(self.coverage_dict, path)

    def load(self, path):
        print('Loading saved NC from %s...' % path)
        self.coverage_dict = torch.load(path)


class KMNC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.k = hyper
        self.name = 'KMNC'
        self.range_dict = {}
        coverage_multisec_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            coverage_multisec_dict[layer_name] = torch.zeros((num_neuron, self.k + 1)).type(torch.BoolTensor).to(
                self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000,
                                           torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'multisec': coverage_multisec_dict
        }
        self.current = 0

    def build(self, data_loader):
        print('Building range...')
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.set_range(data)

    def set_range(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            cur_max, _ = layer_output.max(0)
            cur_min, _ = layer_output.min(0)
            is_less = cur_min < self.range_dict[layer_name][0]
            is_greater = cur_max > self.range_dict[layer_name][1]
            self.range_dict[layer_name][0] = is_less * cur_min + ~is_less * self.range_dict[layer_name][0]
            self.range_dict[layer_name][1] = is_greater * cur_max + ~is_greater * self.range_dict[layer_name][1]

    def calculate(self, data):
        multisec_cove_dict = {}
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]
            num_neuron = layer_output.size(1)
            multisec_index = (u_bound > l_bound) & (layer_output >= l_bound) & (layer_output <= u_bound)
            multisec_covered = torch.zeros(num_neuron, self.k + 1).type(torch.BoolTensor).to(self.device)
            div_index = u_bound > l_bound
            div = (~div_index) * 1e-6 + div_index * (u_bound - l_bound)
            multisec_output = torch.ceil((layer_output - l_bound) / div * self.k).type(torch.LongTensor).to(
                self.device) * multisec_index
            # (1, k), index 0 indicates out-of-range output

            index = tuple([torch.LongTensor(list(range(num_neuron))), multisec_output])
            multisec_covered[index] = True
            multisec_cove_dict[layer_name] = multisec_covered | self.coverage_dict['multisec'][layer_name]

        return {
            'multisec': multisec_cove_dict
        }

    def coverage(self, cove_dict):
        multisec_cove_dict = cove_dict['multisec']
        (multisec_cove, multisec_total) = (0, 0)
        for layer_name in multisec_cove_dict.keys():
            multisec_covered = multisec_cove_dict[layer_name]
            num_neuron = multisec_covered.size(0)
            multisec_cove += torch.sum(multisec_covered[:, 1:])
            multisec_total += (num_neuron * self.k)
        multisec_rate = torch.true_divide(multisec_cove, multisec_total)
        return multisec_rate.item()

    def save(self, path):
        print('Saving recorded %s in \%s' % (self.name, path))
        state = {
            'range': self.range_dict,
            'coverage': self.coverage_dict
        }
        torch.save(state, path)

    def load(self, path):
        print('Loading saved %s from %s' % (self.name, path))
        state = torch.load(path)
        self.range_dict = state['range']
        self.coverage_dict = state['coverage']


class NBC(KMNC):
    def init_variable(self, hyper):
        assert hyper is None
        self.name = 'NBC'
        self.range_dict = {}
        coverage_lower_dict = {}
        coverage_upper_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            coverage_lower_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            coverage_upper_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
            self.range_dict[layer_name] = [torch.ones(num_neuron).to(self.device) * 10000,
                                           torch.ones(num_neuron).to(self.device) * -10000]
        self.coverage_dict = {
            'lower': coverage_lower_dict,
            'upper': coverage_upper_dict
        }
        self.current = 0

    def calculate(self, data):
        lower_cove_dict = {}
        upper_cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            [l_bound, u_bound] = self.range_dict[layer_name]

            lower_covered = (layer_output < l_bound).sum(0) > 0
            upper_covered = (layer_output > u_bound).sum(0) > 0

            lower_cove_dict[layer_name] = lower_covered | self.coverage_dict['lower'][layer_name]
            upper_cove_dict[layer_name] = upper_covered | self.coverage_dict['upper'][layer_name]

        return {
            'lower': lower_cove_dict,
            'upper': upper_cove_dict
        }

    def coverage(self, cove_dict):
        lower_cove_dict = cove_dict['lower']
        upper_cove_dict = cove_dict['upper']

        (lower_cove, lower_total) = (0, 0)
        (upper_cove, upper_total) = (0, 0)
        for layer_name in lower_cove_dict.keys():
            lower_covered = lower_cove_dict[layer_name]
            upper_covered = upper_cove_dict[layer_name]

            lower_cove += lower_covered.sum()
            upper_cove += upper_covered.sum()

            lower_total += len(lower_covered)
            upper_total += len(upper_covered)
        lower_rate = torch.true_divide(lower_cove, lower_total)
        upper_rate = torch.true_divide(upper_cove, upper_total)
        return (lower_rate + upper_rate).item() / 2


class TKNC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.k = hyper
        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
        self.current = 0

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            batch_size = layer_output.size(0)
            num_neuron = layer_output.size(1)
            # layer_output: (batch_size, num_neuron)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=False)
            # idx: (batch_size, k)
            covered = torch.zeros(layer_output.size()).to(self.device)
            index = tuple([torch.LongTensor(list(range(batch_size))), idx.transpose(0, 1)])
            covered[index] = 1
            is_covered = covered.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return torch.true_divide(cove, total).item()

    def save(self, path):
        print('Saving recorded TKNC in %s' % path)
        torch.save(self.coverage_dict, path)

    def load(self, path):
        print('Loading saved TKNC from %s' % path)
        self.coverage_dict = torch.load(path)


if __name__ == '__main__':
    pass

class NeuronCoverage:
    def __init__(self, model):
        self.model = model
        self.all_layer_name = self._get_all_layer_name()
        self.model_layer_dict = self._init_coverage_table()

    def _init_coverage_table(self):
        model_layer_dict = defaultdict(bool)
        for name, module in self.model.named_modules():
            if name in self.all_layer_name:
                if isinstance(module, torch.nn.Conv2d):
                    for index in range(module.out_channels):
                        model_layer_dict[(name, index)] = False
                if isinstance(module, torch.nn.Linear):
                    for index in range(module.out_features):
                        model_layer_dict[(name, index)] = False
        return model_layer_dict

    def _get_all_layer_name(self):
        all_layer_name = []
        # 研究一下卷积层的神经元覆盖率和全连接层有没有区别
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                all_layer_name.append(name)
            if isinstance(m, torch.nn.Linear):
                all_layer_name.append(name)
        return all_layer_name

    def get_neuron_to_cover(self):
        # 随机返回一个未覆盖的 layer_name 和 index,如果全覆盖则随机返回
        not_covered = [(layer_name, index) for (layer_name, index), v in self.model_layer_dict.items() if not v]
        if not_covered:
            layer_name, index = random.choice(not_covered)
        else:
            layer_name, index = random.choice(self.model_layer_dict.keys())
        return layer_name, index

    def update_coverage(self, data, threshold=0.75):
        layer_name = self.all_layer_name
        # intermediate_layer_outputs一个包含所有输出的list 每个元素的size为 (batch_size,channel,h,w)，batch_size = 1
        intermediate_layer_outputs = self._get_forward_value(data)
        for i, output in enumerate(intermediate_layer_outputs):
            scaled = self._scale(output[0])
            for num_neuron in range(scaled.shape[0]):
                if np.mean(scaled[num_neuron, ...]) > threshold \
                        and not self.model_layer_dict[(layer_name[i], num_neuron)]:
                    self.model_layer_dict[(layer_name[i], num_neuron)] = True

    def is_full_coverage(self):
        if False in self.model_layer_dict.values():
            return False
        else:
            return True

    def _scale(self, input, rmax=1, rmin=0):
        # input size (channel,h,w)-->(channel)
        input = input.cpu().detach().numpy()
        input_std = (input - np.min(input)) / (input.max() - input.min())
        input_scaled = input_std * (rmax - rmin) + rmin
        return input_scaled

    def neuron_coverage_rate(self):
        # 统计神经元覆盖率
        covered_neurons = len([v for v in self.model_layer_dict.values() if v])
        total_neurons = len(self.model_layer_dict)
        return covered_neurons / float(total_neurons)

    def _get_forward_value(self, data):
        all_out = []

        def forward_hook(module, input, output):
            all_out.append(output)

        for name, modules in self.model.named_modules():
            if name in self.all_layer_name:
                modules.register_forward_hook(forward_hook)
        output = self.model(data)
        return all_out

    # 目标函数1
    def compute_obj1(self, label, out):
        return sum(o for o in out) - out[label]

    # 目标函数2
    def compute_obj2(self, data):
        layer_name, index = self.get_neuron_to_cover()
        loss = get_output(self.model, data, layer_name, index).mean()
        return loss


def get_output(model, data, layer_name, index):
    out = []

    def forward_hook(module, input, output):
        out.append(output)

    for name, modules in model.named_modules():
        if name == layer_name:
            modules.register_forward_hook(forward_hook)
    forward_out = model(data)
    # out[0] : (batch_size,channel,h,w)
    index_out = out[0][0, index, ...]
    return index_out

# model = ResNet18()
# input_ = torch.randn(1, 3, 32, 32)
# aa = NeuronCoverage(model)
# print(aa.all_layer_name)
# print(aa.model_layer_dict)
# aa.update_coverage(data=input_)
# layer_name, index = aa.get_neuron_to_cover()
# print(layer_name, index)
# input_x = get_output(model, input_, layer_name, index)
# print(input_x)
