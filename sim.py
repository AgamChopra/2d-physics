import pygame
from numpy import concatenate, nan_to_num, where, sum, asarray
from numpy.random import randint
from scipy.spatial.distance import cdist
from numba import jit

SIG = 2.6E-3
EPS = 3E-5
MASS = 4.65E-27
FPS = 144
SPF = 1/FPS
RADIUS = 3

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 1600, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('sim')

EPSILON = 1E-20


@jit(nopython=False)
def Lennard_Jones_dynamics(ringo, color, RADIUS=1E-6, MASS=1E-6, SPF=1/144, WIDTH=600, HEIGHT=600, SIG=SIG, EPS=EPS):
    #Coordinate vectors
    x = ringo[:, :2]
    
    #Distance matrix R(N,N) and direction matrin R_(N,N,2)
    R = cdist(x, x, metric='euclidean')
    R_ = asarray([(x - x[i]) / (R[i].reshape(R.shape[0], 1) + EPSILON) for i in range(x.shape[0])])

    #Calculating aceleration for timestep and updating velocity and position(Eucledean)
    a = sum(nan_to_num((48/MASS) * (EPS/SIG) * ((SIG/(R + EPSILON)) ** 13 -
            (((SIG/(R + EPSILON)) ** 7) * 0.5))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    v = ringo[:, 2:] + (a * SPF)
    x = x + (v * SPF)
    
    #Applying boundry condition
    v = v * concatenate((where(x[:, 0] > WIDTH - 1, -1, 1).reshape(x.shape[0], 1), where(
        x[:, 1] > HEIGHT - 1, -1, 1).reshape(x.shape[0], 1)), 1) * where(x < 1, -1, 1)
    
    #Outputs ringo-physics, arty-rendering
    ringo = concatenate((x, v), axis=1)
    arty = concatenate((x.astype('int32'), color), axis=1)
    return arty, ringo


def draw_window(tensor, lighting):
    DISH.fill(lighting)
    for cell in tensor:
        pygame.draw.circle(
            DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS)
    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    run = True
    N = 300
    width = randint(0, WIDTH, (N, 1)).astype('float64')
    height = randint(0, HEIGHT, (N, 1)).astype('float64')
    vx = randint(-50, 50, (N, 1)).astype('float64')*0.
    vy = randint(-50, 50, (N, 1)).astype('float64')*0.
    ringo = concatenate((width, height, vx, vy), axis=1)
    color = randint(150, 255, (N, 3))
    time_step = 0
    lighting = BLACK

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        arty, ringo = Lennard_Jones_dynamics(
            ringo, color, RADIUS=RADIUS*1E-6, MASS=MASS, WIDTH=WIDTH, HEIGHT=HEIGHT, SPF=SPF, EPS=EPS)
        draw_window(arty, lighting)
        time_step += 1
        # print(ringo[0])
        # print(arty[0])
    pygame.quit()


if __name__ == '__main__':
    main()
