from time import time
import cupy as cp
from scipy.ndimage import zoom


import pygame
import sys
from pygame.locals import *
pygame.init()

RESOLUTION = 100
SAMPLES = 70
FOV = 1
DEPTH = 15
UPSAMPLE = 4
disp = pygame.display.set_mode((RESOLUTION * UPSAMPLE, RESOLUTION * UPSAMPLE))
font = pygame.font.Font(pygame.font.get_default_font(), 12)

decay = cp.linspace(.5, 1, SAMPLES)
sample_steps = decay.cumsum()
sample_steps *= DEPTH / sample_steps[-1]
sample_steps = sample_steps[cp.newaxis, cp.newaxis, ..., cp.newaxis]


def unitize(v):
    v = v / (v * v).sum()**.5
    return v


def view_dirs(norm: cp.ndarray) -> cp.ndarray:
    dir_0 = cp.asarray([-float(norm[1]), float(norm[0]), 0.])
    dir_1 = cp.cross(norm, dir_0)
    dir_0 = unitize(dir_0) * FOV
    dir_1 = unitize(dir_1) * FOV
    dirs = cp.tile(unitize(norm) + dir_0 * cp.linspace(-1, 1, RESOLUTION)[..., cp.newaxis], (RESOLUTION, 1)).reshape(RESOLUTION, RESOLUTION, 3)
    dirs += (dir_1 * cp.linspace(-1, 1, RESOLUTION)[..., cp.newaxis])[:, cp.newaxis]
    return dirs


def get_paths(point, dirs) -> cp.ndarray:
    #        (1 x 1 x 1 x 3) + (RES x RES x 3) * (1 x 1 x SAMPLES x 1) = RES x RES x SAMPLES x 3
    return (point[cp.newaxis, cp.newaxis, cp.newaxis]
            + dirs[..., cp.newaxis, :] * sample_steps
            + cp.random.rand(RESOLUTION, RESOLUTION, SAMPLES, 3) * DEPTH / SAMPLES / 3) * decay[cp.newaxis, cp.newaxis, ..., cp.newaxis]  # dither


def render(swatch, points, frame):
    r, g, b = swatch(points[..., 0], points[..., 1], points[..., 2], frame)
    r = r.mean(axis=-1)
    g = g.mean(axis=-1)
    b = b.mean(axis=-1)

    return cp.clip(cp.stack((r,g,b), axis=-1), 0, 1)


def add_swatches(a, b):
    def added(x, y, z, t):
        r0,g0,b0 = a(x,y,z,t)
        r1,g1,b1 = b(x,y,z,t)
        return r0+r1, g0+g1, b0+b1
    return added


def color_axes(x, y, z, _):
    r = (cp.abs(y) < .5) & (cp.abs(z) < .5) & (x > -.5)
    g = (cp.abs(x) < .5) & (cp.abs(z) < .5) & (y > -.5)
    b = (cp.abs(x) < .5) & (cp.abs(y) < .5) & (z > -.5)
    return 5 * r, 5 * g, 5 * b


def color_axes2(x, y, z):
    fx = (1 + cp.sin(3 * x)) / 2
    r = ((cp.abs(y) < .5 *fx) & (cp.abs(z) < .5*fx) & (x > -.5))
    g = ((cp.abs(x) < .5) & (cp.abs(z) < .5) & (y > -.5))
    b = ((cp.abs(x) < .5) & (cp.abs(y) < .5) & (z > -.5))
    return 5 * r, 5 * g, 5 * b


def sphere_swatch(x, y, z, _):
    X = x % .5
    Y = y % .5
    Z = z % .5
    r = (.25 - (X - .25) ** 2 + (Y - .25) ** 2 + (Z - .25) ** 2)**2 * (x % 3 < .5)
    g = (.25 - (X - .25) ** 2 + (Y - .25) ** 2 + (Z - .25) ** 2)**2 * (y % 3 < .5)
    b = (.25 - (X - .25) ** 2 + (Y - .25) ** 2 + (Z - .25) ** 2)**2 * (z % 3 < .5)
    return r * 40, g * 40, 40 * b


def meteors(x, y, z, t):
    water = (z < 0) / (z**2 + 1) / 10
    fade = (z > 0) + water * 2
    z = cp.abs(z)
    X = x % 3
    Y = (y + .75 * t) % 3
    Z = (z + (0.75) * t + 1274321 * cp.sin((cp.floor(456764 + x/3) * 1029037) * cp.floor(64767 + (y + .75 * t)/3))) % 13
    Y2, Z2 = Y-1.25, Z-1.25
    dist = (X - 1.5)**2 + (1.5 - Y)**2 + (11.5 - Z)**2
    dist2 = (X - 1.5)**2 + (1.5 - Y2)**2 + (11.5 - Z2)**2
    r = (dist < 1) * dist + (dist2 < .8) * dist2
    g = (dist < .5) + (dist2 < .3) * dist2
    b = ((dist < .25) + (dist2 < .2) * dist2) * fade + water
    return 30*r * fade, 10*g * fade, 20*b


def spherez(x,y,z,t):
    X = x % 2
    Y = y % 2
    Z = z % 2
    dist = (X - 1)**2 + (Y - 1)**2 + (Z - 1)**2
    r = (dist < 0.5) & (x % 6 < 2)
    g = (dist < 0.5) & (y % 6 < 2)
    b = (dist < 0.5) & (z % 6 < 2)
    return r * 5,g * 5,b * 5


def shell(x, y, z):
    dist = (x**2 + y**2 + z**2)**.5
    r = (dist > 1) & (dist < 1.5)
    g = (dist > .5) & (dist < 1)
    b =  (dist < .5)
    return r * 3, g * 3, b * 5


def shell_overlap(x, y, z, t):
    t /= 10
    dist = (x**2 + y**2 + z**2)**.5
    r = (dist > 1) & (dist < 1.5)
    dist = ((x - cp.sin(t))**2 + y**2 + z**2)**.5
    g = (dist > .5) & (dist < 1)
    dist = (x**2 + y**2 + (z + cp.cos(t) * .5)**2)**.5
    b = dist < .5
    return r * 3, g * 3, b * 5


def ripple(x,y,z, t):
    f = cp.sin(x + t / 5) + cp.sin(y + t / 5)
    r = cp.abs(z - f) < 0.2
    f = cp.cos(x + t / 5) + cp.cos(y + t / 5)
    g = cp.abs(z - f) < 0.2
    b = g * 0
    return r * 10, g * 10, b


def helix(x,y,z, t):
    R = 3
    z2 = z + cp.sin(-2 * z) + (t + cp.sin(t / 10)) / 5
    x2 = R * cp.cos(z2)
    y2 = R * cp.sin(z2)
    dist = (x - x2)**2 + (y - y2)**2
    r = (dist < 1)
    g = r * 0
    b = g
    return r * 10, g, b


def helix2(x,y,z, t):
    R = 2
    t *= -1
    z2 = z + cp.sin(3 * z) + (t + cp.sin(t / 10)) / 5
    x2 = R * cp.sin(z2)
    y2 = R * cp.cos(z2)
    dist = (x - x2)**2 + (y - y2)**2
    g = dist < 1
    return g * 2, g * 10, g * 5


def bandz(x,y,z,t):
    f = cp.sin(x) + cp.cos(y) + cp.sin(z**2)
    r = ((t / 50) % 1 < f % 1) & ((t / 50 + 0.3 ) % 1> f % 1)
    g = ((t / 50 + 0.6) % 1 < f % 1) & ((t / 50 + 0.9) % 1  > f % 1)
    b = ((t / 50 + 0.3 )% 1 < f % 1) & ((t / 50 + 0.6 ) % 1> f % 1)
    return r,g,b


double_helix = add_swatches(helix, helix2)


def display(a):
    surf = pygame.surfarray.make_surface(zoom(
        cp.asnumpy(cp.flip(cp.flip(cp.swapaxes(a, 0, 1), 1), 0)) * 255, (float(UPSAMPLE), float(UPSAMPLE), 1), order=1))
    disp.blit(surf, (0, 0))


phi = cp.pi / 2
theta = 0
pos = cp.zeros(3)
dir = cp.array([cp.sin(phi) * cp.cos(theta), cp.sin(phi) * cp.sin(theta), cp.cos(phi)])
frame = 0
while True:
    now = time()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        dir2 = cp.array([-float(dir[1]), float(dir[0]), 0])
        dir2 = unitize(dir2)
        pos += dir2 * 0.1
    if keys[pygame.K_RIGHT]:
        dir2 = cp.array([-float(dir[1]), float(dir[0]), 0])
        dir2 = unitize(dir2)
        pos -= dir2 * 0.1
    if keys[pygame.K_LSHIFT]:
        pos -= cp.array([0, 0, 1]) * 0.1
    if keys[pygame.K_SPACE]:
        pos += cp.array([0, 0, 1]) * 0.1
    if keys[pygame.K_DOWN]:
        pos -= dir * 0.1
    if keys[pygame.K_UP]:
        pos += dir * 0.1
    if keys[pygame.K_w]:
        phi += 0.1
    if keys[pygame.K_a]:
        theta += 0.1
    if keys[pygame.K_s]:
        phi -= 0.1
    if keys[pygame.K_d]:
        theta -= 0.1

    frame += 1
    dir = cp.array([cp.sin(phi) * cp.cos(theta), cp.sin(phi) * cp.sin(theta), cp.cos(phi)])
    v = view_dirs(dir)
    p = get_paths(pos, v)
    r = render(double_helix, p, frame)
    display(r)
    numpy_pos = cp.asnumpy(pos)
    text_surface = font.render(f'{1/(time() - now):.0f} FPS',
                               True, (255, 255, 255))
    disp.blit(text_surface, dest=(0, 20))
    text_surface = font.render(f'{numpy_pos[0]:.1f}, {numpy_pos[1]:.1f}, {numpy_pos[2]:.1f}',
                               True, (255, 255, 255))
    disp.blit(text_surface, dest=(0, 0))
    pygame.display.update()
