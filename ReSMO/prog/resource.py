import numpy as np
import math
from itertools import product
from collections import namedtuple
from pprint import pprint

# пространство состояний
StateSpace = namedtuple("StateSpace", ("n_m", "n_u"))
StateSpace.__str__ = lambda self: f"{self[0]},{self[1]}"
StateSpace.__hash__ = lambda self: hash((self.n_m, self.n_u))


class Resource:
    def __init__(self, N, b, lam_m, lam_u, mu_m, mu_u, N0, R, F):
        # исходные данные
        self.C = b*N  # количество ресурсных единиц в системе
        self.b = b  # количество ресурсных единиц в одном ресурсном блоке
        self.N = N  # количество ресурсных блоков
        self.lam_u = lam_u  # интенсивность поступления 1-заявок типа eMBB
        self.lam_m = lam_m  # интенсивность поступления 2-заявок типа URLLC
        self.mu_u = mu_u  # интенсивность обслуживания 1-заявок типа eMBB
        self.mu_m = mu_m  # интенсивность обслуживания 2-заявок типа URLLC
        # self.b = b # вектор скоростей обслуживания сессий eMBB
        # self.y = y # требование к ресурсу для передачи потокового трафика
        self.N0 = N0 # коэффициент мощности шума
        # self.PL = PL # коэффициент затухания сигнала
        self.R = R # радиус зоны обслуживания
        self.F = F # полоса пропускания частот
        self.A = 1 # константа
        self.B = 1 # константа

        # вспомогательные переменные
        self.rho_u = lam_u / mu_u  # интенсивность предложенной нагрузки 1-заявок типа eMBB
        self.rho_m = lam_m / mu_m  # интенсивность предложенной нагрузки 2-заявок типа URLLC

        # скорость передачи
        # self.b0 = y[0]*y[1]*math.log(1 + y[2]*PL/N0, 2)

        # количество активных сессий eMBB на скорости K
        self.m = [0] # for i in range (self.b)]

        # единичный вектор
        # self.e = lambda i: [1 if enum == i else 0 for enum, _ in enumerate(range(K))]

        # вероятность приема
        self.p = []

        # пространство состояний
        self.X = [
            StateSpace(n_m, n_u)
            # for m in product(range(self.b + 1), repeat=self.b)
            for n_m in range(self.N + 1)
            for n_u in range(self.C + 1)
            if self.b*n_m + n_u <= self.C
            # if sum(m) <= self.N
            # if sum([i * j for i, j in zip(m, self.b)]) + n <= self.C
            # if sum(m[1:]) <= n
        ]

        # распределение вероятностей
        self.P = self.pr_infinitesimal_generator()

    # расчет вероятности приема

    def pr_acceptance(self, n, t):
        if n == 0:
            return 1 - (self.N0 * (2**(t/self.F) - 1) * self.F)**(-2*self.B)/(self.A*self.R)**2
        else:
            # numerator = (1 - self.N0 * (2**(t*(n + 1)/self.F) - 1) * (self.F**(-1)) / (self.A * self.R))**(2 *(n + 1))
            # denominator = (1 - self.N0 * (2**(t*(n)/self.F) - 1) * (self.F**(-1)) / (self.A * self.R))**(2 *(n))
            result = ((1 - self.N0 * (2**(t*(n + 1)/self.F) - 1) * (self.F**(-1)) / (self.A * self.R))**(2 *(n + 1))/
                      #----------------------------------------------------------------------------------------------
                        (1 - self.N0 * (2**(t*(n)/self.F) - 1) * (self.F**(-1)) / (self.A * self.R))**(2 *(n)))
            return result

    # расчет матрицы интенсивностей переходов

    def infinitesimal_generator(self):

        # условия и интенсивности переходов

        condition_transform2intensity = {
            # 1
            lambda s, t: (
                    self.b*s.n_m + s.n_u + self.b <= self.C and
                    StateSpace(s.n_m + 1, s.n_u) == t
            ): lambda s: self.lam_m * self.pr_acceptance(s.n_m + s.n_u, self.b),

            # 2
            lambda s, t: (
                    self.b*s.n_m + s.n_u + 1 <= self.C and
                    StateSpace(s.n_m, s.n_u + 1) == t
            ): lambda s: self.lam_u * self.pr_acceptance(s.n_m + s.n_u, 1),

            # 3
            lambda s, t: (
                    self.b*s.n_m + s.n_u + 1 > self.C and
                    s.n_m >= 0 and
                    StateSpace(s.n_m - 1, s.n_u + 1) == t
            ): lambda s: self.lam_u * self.pr_acceptance(s.n_m + s.n_u, 1),

            # 4
            lambda s, t: (
                    s.n_m > 0 and
                    StateSpace(s.n_m - 1, s.n_u) == t
            ): lambda s: self.mu_m * s.n_m,

            # 5
            lambda s, t: (
                    s.n_u > 0 and
                    StateSpace(s.n_m, s.n_u - 1) == t
            )
            : lambda s: s.n_u*self.mu_u,
        }

        # матрица интенсивностей переходов

        matrix = [[0 for j in range(len(self.X))] for i in range(len(self.X))]

        for i, s_i in enumerate(self.X):
            for j, s_j in enumerate(self.X):
                for cond, intens in condition_transform2intensity.items():
                    if cond(s_i, s_j):
                        matrix[i][j] = intens(s_i)

        matrix = np.array(matrix)
        for i in range(len(self.X)):
            matrix[i, i] = -matrix[i, :].sum()

        return matrix

    # красивый вывод матрицы
    def pprint_matrix(self):
        matrix = self.infinitesimal_generator()
        X_to_print = list(map(str, self.X))
        X_to_print += [str(el) for lists in matrix for el in lists]
        # print(X_to_print)
        max_len = max(map(len, X_to_print))

        mm = matrix
        print("_" * (len(str(self.X[0]))), end="|")
        print("|".join(map(str, self.X)), end="|\n")
        for i, m_i in enumerate(mm):
            print(self.X[i], end="|")
            for j in m_i:
                print(f"{j:_<{max_len}}", end="|")
            print()


    def pr_infinitesimal_generator(self):
        a = self.infinitesimal_generator()
        b = np.zeros(len(self.X))
        a[:, 0] = np.ones(len(self.X))
        b[0] = 1.
        a = a.T
        p = np.linalg.solve(a, b)
        return p

    def blocking_pr_infinitesimal_generator(self):
        # словарь: состояние - вероятность
        d = {s:p for s,p in zip(self.X, self.P)}
        B = [0, 0]

        # Вероятность блокировки сессий eMBB
        # for i, s in enumerate(self.X):
        #     print(i, s.n_u, d[StateSpace((self.C - s.n_u)//self.b, s.n_u)])
        #     B[0] += d[StateSpace((self.C - s.n_u)//self.b, s.n_u)]
        B[0] = sum([d[s] for s in self.X if s.n_m == (self.C - s.n_u)//self.b])

        # вероятность блокировки сессий URLLC
        B[1] = d[StateSpace(0, self.C)]
        return B

    def average_number_of_sessions(self):
        # словарь: состояние - вероятность
        d = {s: p for s, p in zip(self.X, self.P)}
        Na = [0, 0]

        # среднее число сессий eMBB
        for i in range(1, self.N + 1):
            sum = 0
            for j in range(self.C - self.b * i + 1):
                sum += d[StateSpace(i, j)]
                # print (i, j, p[self.X.index([i, j])], sum)
            Na[0] += i*sum

        # среднее число сессий URLLC
        for i in range(1, self.C + 1):
            sum = 0
            for j in range((self.C - i)//self.b + 1):
                sum += d[StateSpace(j, i)]
                # print(j, i, p[self.X.index([j, i])], sum)
            Na[1] += i*sum

            # print(Na[1], i, sum, i*sum)
        return Na


    # Вероятность прерывания eMBB
    def service_interruption_probability(self):
        # словарь: состояние - вероятность
        d = {s: p for s, p in zip(self.X, self.P)}
        I = 0

        for i in range(1, self.N):
            j = self.C - self.b*i
            # print(i, j)
            tmp = sum([d[s] for s in self.X if s.n_m == i and s.n_u == j])
            tmp *= (self.pr_acceptance(i + j, 1)*self.lam_u /
                    (self.pr_acceptance(i + j, 1)*self.lam_u + i*self.mu_u + j*self.mu_u) *
                    1/i)
            I += tmp


        i = self.N
        j = self.C - self.b*i
        I += sum([d[s] for s in self.X if s.n_m == i and s.n_u == i])*(self.pr_acceptance(i + j, 1)*self.lam_u /
                    (self.pr_acceptance(i + j, 1)*self.lam_u + i*self.mu_u) *
                    1/i)

        return I
    #
    # def util(self):
    #     p = self.pr_infinitesimal_generator()
    #     U = 0
    #     for i in range(1, self.M + 1):
    #         for j in range(self.C - self.b*i + self.d*self.I(i)):
    #             U += (self.b*i + j)*p[self.X.index([i, j])]
    #     U +=1
    #     U = U/self.C
    #     return U