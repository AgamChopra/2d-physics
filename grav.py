import pygame
from numpy import concatenate, nan_to_num, where, zeros, sum, dot, ones
from numpy.random import randint
from scipy.spatial.distance import cdist
from numba import jit

G = 6.67430E-11
FPS = 144
SPF = 1/FPS
RADIUS = 3

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 900, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('sim')

EPSILON = 1E-20


@jit(nopython=False)
def newtonian_gravitational_dynamics(ringo, color, M, SPF=1/144, WIDTH=600, HEIGHT=600):
    x = ringo[:, :2]
    R = cdist(x, x, metric='euclidean')
    R_ = zeros((x.shape[0], x.shape[0], x.shape[1]))
    M_ = dot(M,M.T)

    for i in range(x.shape[0]):
        R_[i] = (x - x[i]) / (R[i].reshape(R.shape[0], 1) + EPSILON)
    
    a = sum(nan_to_num((1/(M * ones((R.shape[0], R.shape[0])))) * G * (M_/((R ** 2) + EPSILON))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    v = ringo[:, 2:] + (a * SPF)
    x = x + (v * SPF)
    
    # Boundry Condition
    v = v * concatenate((where(x[:, 0] > WIDTH - 1, -1, 1).reshape(x.shape[0], 1), where(
        x[:, 1] > HEIGHT - 1, -1, 1).reshape(x.shape[0], 1)), 1) * where(x < 1, -1, 1)
    
    # Outputs
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
    N = 9
    M = ones((N, 1))*1E9
    M[0] = M[0]*1E6
    width = randint(0, WIDTH, (N, 1)).astype('float64')
    width[0] = WIDTH/2
    height = randint(0, HEIGHT, (N, 1)).astype('float64')
    height[0] = HEIGHT/2
    vx = randint(-50, 50, (N, 1)).astype('float64')*0.4
    vy = randint(-50, 50, (N, 1)).astype('float64')*0.4
    vx[0] = 0.
    vy[0] = 0.
    ringo = concatenate((width, height, vx, vy), axis=1)
    color = randint(150, 255, (N, 3))
    time_step = 0
    lighting = BLACK

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        arty, ringo = newtonian_gravitational_dynamics(ringo, color, M=M, WIDTH=WIDTH, HEIGHT=HEIGHT, SPF=SPF)
        draw_window(arty, lighting)
        time_step += 1
        # print(ringo[0])
        # print(arty[0])
    pygame.quit()


if __name__ == '__main__':
    main()
