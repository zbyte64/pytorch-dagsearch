import torch


def inf_data(dataloader):
    i = None
    while True:
        if i is None:
            i = iter(dataloader)
        try:
            yield next(i)
        except StopIteration:
            i = None


def one_hot(num_classes):
    y = torch.eye(num_classes)
    def f(labels):
        if isinstance(labels, int):
            return y[labels]
        return y[labels.type(torch.int64)]
    return f