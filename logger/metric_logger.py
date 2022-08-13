from util.distributed import is_dist_avail_and_initialized
from util.console_color import Prints
import time
import datetime

import torch
import torch.distributed as dist

class MetricContainer:
    def __init__(self):
        self.container = []
        self.total = 0.0
        self.count = 0

    def append(self, value):
        self.container.append(value)
        self.count += 1
        self.total += value

    def _synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def avg(self):
        d = torch.tensor(self.container, dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        self._synchronize_between_processes()
        return self.total / self.count


class MetricLogger(object):
    def __init__(
        self,
        metrics: list | tuple,
        summary_printer,
        cur_epoch,
        num_epoch,
        num_iter,
        iter_printer=Prints.info,
        verbose=-1,
        delimiter='\t'
    ):
        self._meters = {metric: MetricContainer() for metric in metrics}
        self.metrics = metrics
        self.delimiter = delimiter
        self.summary_printer = summary_printer
        self.iter_printer = iter_printer
        self.num_epoch = num_epoch
        self.num_iter = num_iter
        self.verbose = verbose
        self.cur_epoch = cur_epoch

    def log(self, header, iter, metrics):
        msg = []
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            assert k in self._meters.keys()
            self._meters[k].append(v)
            msg.append(f'{k}={v:05.2f}')

        msg = self.delimiter.join(msg)
        msg = header + ': ' + msg
        if (iter == 0 or iter % self.verbose == 0 or \
            iter == self.num_iter - 1) and \
            self.verbose != -1:
            self.iter_printer(
                msg=msg,
                process=[{
                    'Name': 'It',
                    'Cur': iter + 1,
                    'Tot': self.num_iter
                }, {
                    'Name': 'Ep',
                    'Cur': self.cur_epoch + 1,
                    'Tot': self.num_epoch
                }]
            )

    def global_avg(self, header):
        self._synchronize_between_processes()
        msg = []
        for name, meter in self._meters.items():
            msg.append(f'{name}={meter.global_avg:05.2f}')
        msg = self.delimiter.join(msg)
        msg = header + ': ' + msg
        self.summary_printer(
            msg=msg,
            process=[{
                'Name': 'Ep',
                'Cur': self.cur_epoch + 1,
                'Tot': self.num_epoch
            }]
        )
    
    def _synchronize_between_processes(self):
        for meter in self._meters.values():
            meter._synchronize_between_processes()