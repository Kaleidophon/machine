from __future__ import print_function
from collections import defaultdict
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


class Log(object):
    """
    The Log can be used to store logs during training, write the to a file
    and read them again later.
    """

    def __init__(self, path=None):
        self.steps = []
        self.data = defaultdict(lambda: defaultdict(list))

        if path is not None:
            self.read_from_file(path)

    def write_to_log(self, dataname, losses, metrics, step):
        """
        Add new losses to Log object.
        """
        for metric in metrics:
            val = metric.get_val()
            self.data[dataname][metric.log_name].append(val)

        for loss in losses:
            val = loss.get_loss()
            self.data[dataname][loss.log_name].append(val)

    def update_step(self, step):
        self.steps.append(step)

    def write_to_file(self, path):
        """
        Write the contents of the log object to a file. Format:

        steps step1 step2 step3 ...
        name_of_dataset1
            metric_name val1 val2 val3 val4 ...
            loss_name val1 val2 val3 val4 ...
            ...
        name_of_dataset2
            ...
        """

        f = open(path, 'wb')

        # write steps
        steps = "steps %s\n" % ' '.join(['%i' % step for step in self.steps])
        f.write(steps.encode())

        # write logs
        for dataset in self.data.keys():
            f.write(dataset.encode() + b'\n')
            for metric in self.data[dataset]:
                data = "\t%s %s\n" % (metric, ' '.join(
                    [str(v) for v in self.data[dataset][metric]]))
                f.write(data.encode())

        f.close()

    def read_from_file(self, path):
        """
        Fill the contents of a log object reading information
        from a file that was also generated by a log object.
        The format of this file should be:

        steps step1 step2 step3 ...
        name_of_dataset1
            metric_name val1 val2 val3 val4 ...
            loss_name val1 val2 val3 val4 ...
            ...
        name_of_dataset2
            ...
        """
        f = open(path, 'rb')

        lines = f.readlines()
        self.steps = [int(i) for i in lines[0].split()[1:]]

        for line in lines[1:]:
            l_list = line.split()
            if len(l_list) == 1:
                cur_set = l_list[0].decode()
            else:
                data = [float(i) for i in l_list[1:]]
                self.data[cur_set][l_list[0].decode()] = data

    def get_logs(self):
        return self.data

    def get_steps(self):
        return self.steps


class LogCollection(object):

    def __init__(self):
        self.logs = []
        self.log_names = []

    def add_log_from_file(self, path):
        self.logs.append(Log(path))
        self.log_names.append(path)

    def add_log_from_folder(self, folder_path, ext='', name_parser=None):
        """
        Recursively loop through a folder and add all its
        the files with appropriate extension to self.logs.
        """

        for subdir, dirs, files in os.walk(folder_path):
            for fname in files:
                f = os.path.join(subdir, fname)

                if f.endswith(ext):
                    if name_parser:
                        log_name = name_parser(f, subdir)
                    else:
                        log_name = f

                    self.logs.append(Log(f))
                    self.log_names.append(log_name)

    def plot_metric(self, metric_name, restrict_model=lambda x: True, 
                    restrict_data=lambda x: True, data_name_parser=None,
                    color_group=False, title='', eor=-1,
                    show_figure=True, ylabel=None, **line_kwargs):
        """
        Plot all values for a specific metrics. A function restrict can be
        inputted to restrict the set of models being plotted. A function group
        can be used to group the results colour-wise.

        Args
            restrict (func):
            group (func):
        """

        # colormap = plt.get_cmap('plasma')(np.linspace(0,1, 25))
        fig, ax = plt.subplots(figsize=(13, 11))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.set_color_cycle(colormap)

        for i, name in enumerate(self.log_names):
            if restrict_model(name):
                label = name + ' '
                log = self.logs[i]
                for dataset in log.data.keys():
                    if restrict_data(dataset):
                        label_name = data_name_parser(
                            dataset, name) if data_name_parser else dataset
                        steps = [step / float(232) for step in log.steps[:eor]]
                        if color_group:
                            steps, data = self.prune_data(
                                steps, log.data[dataset][metric_name][:eor])
                            ax.plot(steps, data,
                                    color_group(name, dataset),
                                    label=label + label_name, linewidth=3.0, **line_kwargs)
                        else:
                            ax.plot(steps,
                                    log.data[dataset][metric_name][:eor],
                                    label=label + label_name, **line_kwargs)
                        ax.tick_params(
                            axis='both', which='major', labelsize=20)
                        plt.xlabel("Epochs", fontsize=24)
                        plt.ylabel(metric_name if ylabel is None else ylabel, fontsize=24)
                        plt.title(title)

        plt.legend()

        if show_figure:
            plt.show()

        return fig

    def find_highest_average(self, metric_name, find_basename,
                             restrict_model=lambda x: True,
                             restrict_data=lambda x: True,
                             find_data_name=lambda x: x):
        """
        Find the highest average over runs, things that have the same
        basename (as returned by 'find_basename') will be averaged.
        """

        data = dict()
        counts = dict()

        for i, name in enumerate(self.log_names):
            if restrict_model(name):
                log = self.logs[i]
                basename = find_basename(name)
                for dataset in log.data.keys():
                    dataname = find_data_name(dataset)
                    if restrict_data(dataset):
                        log_max = max(log.data[dataset][metric_name])
                        if basename in data:
                            if dataname in data[basename]:
                                data[basename][dataname] += log_max
                                counts[basename][dataname] += 1
                            else:
                                data[basename][dataname] = log_max
                                counts[basename][dataname] = 1
                        else:
                            data[basename] = dict()
                            data[basename][dataname] = log_max
                            counts[basename] = dict()
                            counts[basename][dataname] = 1

        # find max
        max_scores = {}
        for basename, datasets in data.items():
            max_scores[basename] = dict()
            for dataset in datasets:
                c = counts[basename][dataset]
                max_av = data[basename][dataset] / c
                max_scores[basename][dataset] = max_av

        return max_scores

    def group_data(self, metric_name, find_basename,
                   restrict_model=lambda x: True,
                   restrict_data=lambda x: True,
                   find_data_name=lambda x: x):

        group_data = defaultdict(lambda: defaultdict(list))

        for i, name in enumerate(self.log_names):
            if not restrict_model(name):
                continue

            log = self.logs[i]

            basename = find_basename(name)
            for dataset in log.data.keys():
                if not restrict_data(dataset):
                    continue

                dataname = find_data_name(dataset)
                group_data[basename][dataname].append(
                    log.data[dataset][metric_name])

        return group_data

    def plot_groups(self, metric_name, find_basename,
                    restrict_model=lambda x: True,
                    restrict_data=lambda x: True,
                    find_data_name=lambda x: x,
                    color_group=False, eor=-1):

        import numpy as np

        fig, ax = plt.subplots(figsize=(13, 11))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        group_data = self.group_data(metric_name=metric_name,
                                     restrict_model=restrict_model,
                                     find_basename=find_basename,
                                     find_data_name=find_data_name,
                                     restrict_data=restrict_data)

        steps = [step / float(232) for step in self.logs[0].steps[:eor]]
        for model, data in group_data.items():
            for dataset in data:
                av = np.mean(data[dataset], axis=0)[:eor]
                if color_group:
                    print(dataset, model, color_group(model, dataset))
                    ax.plot(steps, av, color_group(model, dataset),
                            label=model + dataset, linewidth=3.0)
                else:
                    ax.plot(steps, av, dataset, label=model + dataset)

        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.xlabel("Epochs", fontsize=24)
        plt.ylabel("Loss", fontsize=24)
        plt.legend()
        plt.show()

        return fig

    def prune_data(self, steps, data):
        return steps, data

        i = 1
        while i < len(data):
            d1, d2 = data[i - 1], data[i]
            if d1 - d2 > 1:
                del steps[i]
                del data[i]
                continue

            i += 1

        return steps, data