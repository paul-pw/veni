import torch
from torch import sin, cos, atan2, acos, pi

def rot_x(alpha, degrees=False):
    if degrees:
        alpha = alpha * pi / 180
    return torch.tensor([
        [1, 0, 0],
        [0, cos(alpha), -sin(alpha)],
        [0, sin(alpha), cos(alpha)]
    ], dtype=alpha.dtype)

def rot_y(beta, degrees=False):
    if degrees:
        beta = beta * pi / 180
    return torch.tensor([
        [cos(beta), 0, sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, cos(beta)]
    ], dtype=beta.dtype)

def rot_z(gamma, degrees=False):
    if degrees:
        gamma = gamma * pi / 180
    return torch.tensor([
        [cos(gamma), -sin(gamma), 0],
        [sin(gamma), cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)

def rot(alpha, beta, gamma, degrees=False):
    if degrees:
        alpha = alpha * pi / 180
        beta = beta * pi / 180
        gamma = gamma * pi / 180
    return rot_x(alpha) @ rot_y(beta) @ rot_z(gamma)

def rot_2d(theta, degrees=False):
    if degrees:
        theta = theta * pi / 180
    return torch.tensor([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ], dtype=theta.dtype)
