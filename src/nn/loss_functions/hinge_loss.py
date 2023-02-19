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
    correct_labels = (range(len(target)), target)
    correct_class_scores = inpt.array[correct_labels]  # Nx1

    loss_element = inpt.array - correct_class_scores[:, np.newaxis] + 1  # NxC
    correct_classifications = np.where(loss_element <= 0)

    loss_element[correct_classifications] = 0
    loss_element[correct_labels] = 0

    grad = np.ones(loss_element.shape, dtype=np.float16)
    grad[correct_classifications], grad[correct_labels] = 0, 0
    grad[correct_labels] = -1 * grad.sum(axis=-1)
    grad /= inpt.array.shape[0]

    loss = np.sum(loss_element) / inpt.array.shape[0]

    return Loss(loss, grad, inpt.model)

    # margins = np.maximum(0, scores - scores[np.arange(scores.shape[0]), y][:, np.newaxis] + margin)
    #
    # # set the margin of the correct class to zero
    # margins[np.arange(scores.shape[0]), y] = 0
    #
    # # compute the average hinge loss across all examples
    # loss = np.mean(margins)
    #
    # return loss
