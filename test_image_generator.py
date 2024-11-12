from abc import ABCMeta

import numpy as np
from PIL import Image


grassland = Image.open("assets/pexels-stephanthem-753869.jpg")
flower1 = Image.open("assets/Honeycrisp-Apple.png").resize((100, 100))
flower2 = Image.open("assets/Bananas.svg.png").resize((100, 100))


class Normal(metaclass=ABCMeta):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.variance = sigma ** 2
        self.coefficient = 1 / np.sqrt(2 * np.pi * self.variance)

    def f(self, x):
        return self.coefficient * np.exp(-0.5 * (x - self.mu) ** 2 / self.variance)


class Logistic(metaclass=ABCMeta):
    def __init__(self, l, k, x0, offset):
        self.l = l
        self.k = k
        self.x0 = x0
        self.offset = offset

    def f(self, x):
        return self.offset + self.l / (1 + np.exp(-1 * self.k * (x - self.x0)))


def generate_2d_random_points(x_min, x_max, y_min, y_max, num_points):
    x_coords = np.random.uniform(x_min, x_max, num_points)
    y_coords = np.random.uniform(y_min, y_max, num_points)

    points = np.column_stack((x_coords, y_coords))
    return points

def generate_single_species():
    normal = Normal(15, 6)

    max_flower = 25
    max_value = normal.f(15)
    coefficient = int(max_flower / max_value)

    for i in range(30):
        flower_number = int(coefficient * normal.f(i))
        result = grassland.copy()
        positions = generate_2d_random_points(0, grassland.width, 0, grassland.height, flower_number)
        for p in positions:
            # x and y value of box must be int,
            # the first `flower` is the image to be pasted, and the second one is a mask to keep its transparency.
            result.paste(flower1, (int(p[0]), int(p[1])), flower1)

        result.save(f"test_images/2024-06-{str(i + 1).zfill(2)}.JPG")  # fill 0, keep its length 2 chars

def generate_invasive():
    pass