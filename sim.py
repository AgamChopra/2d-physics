import pygame
from numpy import concatenate, nan_to_num, where, sum, asarray, sin, ones, nan
from numpy.random import randint
from scipy.spatial.distance import cdist
from numba import jit


SIG = 3.3E-12
EPS = 9.98E-11
MASS = 4.65E-26
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
def Lennard_Jones_dynamics(ringo, color, counter, RADIUS=1E-6, MASS=1E-6, SPF=1/144, WIDTH=600, HEIGHT=600, SIG=SIG, EPS=EPS):
    #Coordinate vectors
    x = ringo[:, :2]
    
    #Distance matrix R(N,N) and direction matrin R_(N,N,2)
    R = cdist(x, x, metric='euclidean')
    R_ = asarray([(x - x[i]) / (R[i].reshape(R.shape[0], 1) + EPSILON) for i in counter])

    #Calculating aceleration for timestep and updating velocity and position(Eucledean)
    a = sum(nan_to_num((48/MASS) * (EPS/SIG) * ((SIG/(R + EPSILON)) ** 13 -
            (((SIG/(R + EPSILON)) ** 7) * 0.5))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    v = ringo[:, 2:] + (a * SPF)
    x = x + (v * SPF)
    
    #Applying boundry condition
    v = v * concatenate((where(((x[:,0] > WIDTH) * v[:,0].astype('int32')) > 0, -1, 1).reshape(x.shape[0], 1),
                        where(((x[:,1] > HEIGHT) * v[:,1].astype('int32')) > 0, -1, 1).reshape(x.shape[0], 1)),1) *\
            concatenate((where(((x[:,0] < 0) * v[:,0].astype('int32')) < 0, -1, 1).reshape(x.shape[0], 1),
                        where(((x[:,1] < 0) * v[:,1].astype('int32')) < 0, -1, 1).reshape(x.shape[0], 1)),1)
    x = nan_to_num(x * concatenate((where(x[:,0] > WIDTH + 10, nan, 1).reshape(x.shape[0], 1), ones((x.shape[0],1))),1), nan=WIDTH)
    x = nan_to_num(x * concatenate((ones((x.shape[0],1)),where(x[:,1] > HEIGHT + 10, nan, 1).reshape(x.shape[0], 1)),1), nan=HEIGHT)
    
    #Outputs ringo-physics, arty-rendering
    ringo = concatenate((x, v), axis=1)
    arty = concatenate((x.astype('int32'), color), axis=1)
    return arty, ringo


def draw_window(arty, lighting):
    DISH.fill(lighting)
    [pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS) for cell in arty]
    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    run = True
    N = 500
    W_ = 250
    width = randint(0, WIDTH-W_, (N, 1)).astype('float64')
    height = randint(0, HEIGHT, (N, 1)).astype('float64')
    vx = randint(-1900, 1900, (N, 1)).astype('float64')*0.3
    vy = randint(-1900, 1900, (N, 1)).astype('float64')*0.3
    ringo = concatenate((width, height, vx, vy), axis=1)
    color = randint(150, 255, (N, 3))
    time_step = 0
    lighting = BLACK
    counter = range(N)

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        arty, ringo = Lennard_Jones_dynamics(
            ringo, color, counter, RADIUS=RADIUS*1E-6, MASS=MASS, WIDTH=WIDTH + 2 * W_ * (sin(sin(time_step*0.01)*0.01 * time_step)-1), HEIGHT=HEIGHT+ W_ * (sin(sin(time_step*0.001)*0.01 * time_step)-1), SPF=SPF, EPS=EPS)
        draw_window(arty, lighting)
        time_step += 1
        # print(ringo[0])
        # print(arty[0])
    pygame.quit()


if __name__ == '__main__':
    main()
