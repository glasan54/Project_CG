#pakapol Sanarge 5810405223
import sys
import numpy as np
import math as m
import mpmath as mp


def normalize(v):
    len_v = np.linalg.norm(v)
    if len_v != 0:
        v = v*(1/len_v)
    return v


def Identity():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)

def Translate(a, b, c):
    return np.array([[1, 0, 0, a],
                     [0, 1, 0, b],
                     [0, 0, 1, c],
                     [0, 0, 0, 1]], dtype=np.float32)

def Rotate(angle, x, y, z):
    c = m.cos(angle*m.pi/180)
    s = m.sin(angle*m.pi/180)

    return np.array([[m.pow(x,2)*(1-c)+c, (x*y)*(1-c)-(z*s), (x*z)*(1-c)+(y*s), 0],
                     [(x*y)*(1-c)+(z*s), m.pow(y,2)*(1-c)+c, (y*z)*(1-c)-(x*s), 0],
                     [(x*z)*(1-c)-(y*s), (y*z)*(1-c)+(x*s), m.pow(z,2)*(1-c)+c, 0],
                     [0, 0, 0, 1]], dtype=np.float32)

def Scale(x, y, z):
    return np.array([[x, 0, 0, 0],
                     [0, y, 0, 0],
                     [0, 0, z, 0],
                     [0, 0, 0, 1]], dtype=np.float32)

def LookAt(eyex, eyey, eyez, atx, aty, atz, upx, upy, upz):
    
    eye = np.array((eyex, eyey, eyez), dtype=np.float32)
    at = np.array((atx, aty, atz), dtype=np.float32)
    up = np.array((upx,upy,upz), dtype=np.float32)

    z = normalize(eye - at)
    y_up = normalize(up)
    x = normalize(np.cross(y_up,z))
    y = normalize(np.cross(z,x))
    
    return np.array([[x[0], x[1], x[2], -np.dot(x,eye)],
                     [y[0], y[1], y[2], -np.dot(y,eye)],
                     [z[0], z[1], z[2], -np.dot(z,eye)],
                     [0, 0, 0, 1]], dtype=np.float32)

def Ortho(left, right, bottom, top, near, far):
    x = -(right + left)/(right - left)
    y = -(top + bottom)/(top - bottom)
    z =  -(far + near)/(far - near)

    return np.array([[2/(right-left), 0, 0, x],
                     [0, 2/(top-bottom), 0, y],
                     [0, 0, -2/(far-near), z],
                     [0, 0, 0, 1]], dtype=np.float32)


def Frustum(left, right, bottom, top, near, far):
    x = (right + left)/(right - left)
    y = (top + bottom)/(top - bottom)
    z = -(far + near)/(far - near)
    w = -(2*near*far)/(far-near)

    return np.array([[(2*near)/(right - left), 0, x, 0],
                     [0, (2*near)/(top - bottom), y, 0],
                     [0, 0, z, w],
                     [0, 0, -1, 0]], dtype=np.float32)


def Perspective(fovy, aspect, near, far):
    f = 1/m.tan((fovy/2)* m.pi/180)
    x = f/aspect
    y = f
    z = -(far+near)/(far-near)
    w = -(2*near*far)/(far-near)

    return np.array([[x, 0, 0, 0],
                     [0, y, 0, 0],
                     [0, 0, z, w],
                     [0, 0, -1, 0]], dtype=np.float32)


