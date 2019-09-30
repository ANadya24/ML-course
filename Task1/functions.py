def prod_non_zero_diag(x):
    mult = 1
    for i in range(min(len(x), len(x[0]))):
        if (x[i][i] != 0):
            mult = mult * x[i][i]
    return mult


def are_multisets_equal(x, y):
    y = list(y)
    for i in x:
        if y.count(i) != 0:
            y.remove(i)
        else:
            return False
    return y == []


def max_after_zero(x):
    y = [x[i] for i in range(1, len(x)) if x[i-1] == 0]
    return max(y)


def convert_image(img, vec):
    vec = list(vec)
    img = list(img)
#     print (len(img), len(img[0]))
    im = []
    for i in range(len(img)):
        im.append([])
        for j in range(len(img[0])):
            im[i].append([0])
            a = 0
            for k in range(len(vec)):
                a = a + img[i][j][k] * vec[k]
            im[i][j] = a
    return im


def run_length_encoding(x):
    counts = []
    nums = [x[0]]
    k = 1
    for i in range(1, len(x)):
        if x[i] == x[i-1]:
            k = k + 1
        else:
            counts.append(k)
            k = 1
            nums.append(x[i])
    counts.append(k)
    return (nums, counts)


def pairwise_distance(x, y):
    a = []
    for i in range(len(x)):
        a.append([])
        for j in range(len(y)):
            a[i].append(([(x[i][k] - y[j][k])**2 for k in range(len(x[0]))]))
            a[i][j] = abs((sum(a[i][j]))**0.5)
    return a
