import numpy as np
from nn.loss_functions.loss import Loss


def hinge_loss(inpt, target):
    """Реализует функцию ошибки hinge loss

    ---------
    Параметры
    ---------
    inpt : Tensor
        Предсказание модели

    target
        Список реальных классов
        Одномерный массив

    ----------
    Возвращает
    ----------
    loss : Loss
        Ошибка
    """
    # Мы должны сконвертировать массив реальных меток
    # в двумерный массив размера (N, C),
    # где N -- число элементов
    # С -- число классов
    C = inpt.array.shape[-1]
    target = np.eye(C)[target]

    # hinge loss для бинарной классификации
    loss = np.mean(np.maximum(0, 1 - target * inpt.array))

    grad = -target * (target * inpt.array < 1)
    grad = grad / inpt.array.shape[-1]
    grad = grad / target.shape[0]

    return Loss(loss, grad, inpt.model)
