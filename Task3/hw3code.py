import numpy as np
from collections import Counter


def compute_bias_variance(regressor, dependence_fun, x_generator=np.random.uniform, noise_generator=np.random.uniform,
                          sample_size=300, samples_num=300, objects_num=200, seed=1234):
    """
    После генерации всех необходимых объектов, должна вызываться функция compute_bias_variance_fixed_samples.

    Рекомендации:
    * Создайте вектор объектов для оценивания интеграла по $x$, затем вектор зашумленных правильных ответов.
      Оцените мат. ожидание шума с помощью генерации отдельной шумовой выборки длины objects_num.
    * Проверить правильность реализации можно на примерах, которые разбирались на семинаре и в домашней работе.

    :param regressor: объект sklearn-класса, реализующего регрессионный алгоритм (например, DecisionTreeRegressor,
     LinearRegression, Lasso, RandomForestRegressor ...)
    :param dependence_fun: функция, задающая истинную зависимость в данных. Принимает на вход вектор и возвращает вектор
     такой же длины. Примеры: np.sin, lambda x: x**2
    :param x_generator: функция, генерирующая одномерную выборку объектов и имеющая параметр size (число объектов в
     выборке). По умолчанию np.random.uniform
    :param noise_generator: функция, генерирующая одномерную выборку шумовых компонент (по одной на каждый объект) и
     имеющая параметр size (число объектов в выборке). По умолчанию np.random.uniform
    :param sample_size: число объектов в выборке
    :param samples_num: число выборок, которые нужно сгенерировать, чтобы оценить интеграл по X
    :param objects_num: число объектов, которые нужно сгенерировать, чтобы оценить интеграл по x
    :param seed: seed для функции np.random.seed

    :return bias: смещение алгоритма regressor (число)
    :return variance: разброс алгоритма regressor (число)
    """
    np.random.seed(seed)
    X_l = x_generator(size=(samples_num, sample_size))
#     y_l = dependence_fun(X_l) + noise_generator(size=(samples_num, sample_size))
    noise = noise_generator(size=(samples_num, sample_size))
    noise_X = noise_generator(size=objects_num)
    mean_noise = np.mean(noise)
    bias, variance = compute_bias_variance_fixed_samples(regressor, dependence_fun,
                                                         X_l, noise_X, noise, mean_noise)
    return bias, variance


def compute_bias_variance_fixed_samples(
        regressor, dependence_fun, samples, objects, noise, mean_noise):
    """
    В качестве допущения, будем оценивать $E_X\left[\mu(X)\right](x)$ как средний ответ на $x$ из samples_num
    алгоритмов, обученных на своих подвыборках $X$

    Рекомендации:
    * $\mathbb{E}[y|x]$ оценивается как сумма правильного ответа на объекте и мат. ожидания шума
      $\mathbb{E}_X [\mu(X)]$ оценивается как в предыдущей задаче: нужно обучить regressor на samples_num выборках длины
       sample_size и усреднить предсказания на сгенерированных ранее объектах.

    :param regressor: объект sklearn-класса, реализующего регрессионный алгоритм (например, DecisionTreeRegressor,
     LinearRegression, Lasso, RandomForestRegressor ...)
    :param dependence_fun: функция, задающая истинную зависимость в данных. Принимает на вход вектор и возвращает вектор
     такой же длины. Примеры: np.sin, lambda x: x**2
    :param samples: samples_num выборок длины sample_size для оценки интеграла по X
    :param objects: objects_num объектов для оценки интеграла по x
    :param noise: шумовая компонента размерности (samples_num, sample_size)
    :param mean_noise: среднее шумовой компоненты

    :return bias: смещение алгоритма regressor (число)
    :return variance: разброс алгоритма regressor (число)
    """
    mu = np.zeros((samples.shape[0], objects.shape[0]))
    y = dependence_fun(samples) + noise
    E_X_mu = np.zeros(objects.shape[0])
    for i in range(samples.shape[0]):
        x = samples[i]
        regressor.fit(x[:, np.newaxis], y[i])
        score = regressor.predict(objects[:, np.newaxis])
        mu[i] = score
        E_X_mu += score
    E_X_mu /= samples.shape[0]
    E_yORx = mean_noise + dependence_fun(objects)
    bias = ((E_X_mu - E_yORx)**2).mean()
    variance = ((mu - E_X_mu)**2).mean()
    return bias, variance


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный порог.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if np.all(feature_vector == feature_vector[0]):
        return -np.inf, -np.inf, -np.inf, -np.inf
    length = len(feature_vector)
    thresholds = np.array([])
    ginis = np.array([])
#     sorted_idxs = np.argsort(feature_vector)
#     sorted_vector = feature_vector[sorted_idxs]
#     sorted_target = target_vector[sorted_idxs]
    usorted_vector = np.unique(feature_vector)
    i = np.arange(1, len(usorted_vector))
    thresholds = 0.5 * (usorted_vector[i] + usorted_vector[i - 1])
    left = np.less(feature_vector, thresholds.reshape(len(thresholds), 1))
    right = np.greater(feature_vector, thresholds.reshape(len(thresholds), 1))
#     right = np.logical_not(left)
    length_left = np.sum(left, axis=1)
    length_right = np.sum(right, axis=1)
    p1 = np.sum(np.multiply(target_vector, left), axis=1) / length_left
    p0 = 1 - p1
    h_l = 1 - p0**2 - p1**2
    p1 = np.sum(np.multiply(target_vector, right), axis=1) / length_right
    p0 = 1 - p1
    h_r = 1 - p0**2 - p1**2
    ginis = np.array(- h_l * length_left / length -
                     h_r * length_right / length)
    index = np.argmax(ginis)
    gini_best = ginis[index]
    threshold_best = thresholds[index]
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None,
                 min_samples_split=None, min_samples_leaf=None):
        if np.any(
                list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        #         if len(sub_y) == 0:
        #             return
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if sub_X.shape[0] < 2:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(
                    zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(
                    list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        go_to_left = False
        if node["type"] == "terminal":
            return node["class"]
        feature_best = node["feature_split"]
        if self._feature_types[feature_best] == "real":
            thr = node["threshold"]
            go_to_left = x[feature_best] < thr
#             print(2)
        elif self._feature_types[feature_best] == "categorical":
            thr = node["categories_split"]
            go_to_left = x[feature_best] in thr
        else:
            raise ValueError
        if (go_to_left):
            #             print(1)
            return self._predict_node(x, node["left_child"])
        else:
            #             print(2)
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
