import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random as rd

def carve_maze(grid: np.ndarray, size: int) -> np.ndarray:
    output_grid = np.empty([size*3, size*3], dtype=str)
    output_grid[:] = '#'

    i = 0
    j = 0
    while i < size:
        w = i*3 + 1
        while j < size:
            k = j*3 + 1
            toss = grid[i, j]
            output_grid[w, k] = ' '
            if toss == 0 and k+2 < size*3:
                output_grid[w, k+1] = ' '
                output_grid[w, k+2] = ' '
            if toss == 1 and w-2 >= 0:
                output_grid[w-1, k] = ' '
                output_grid[w-2, k] = ' '

            j = j + 1

        i = i + 1
        j = 0

    return output_grid


def preprocess_grid(grid: np.ndarray, size: int) -> np.ndarray:
    # fix first row and last column to avoid digging outside the maze external borders
    first_row = grid[0]
    first_row[first_row == 1] = 0
    grid[0] = first_row
    for i in range(1, size):
        grid[i, size-1] = 1
    return grid


def ald(grid:np.ndarray,size:int) -> np.ndarray:
    output_grid = np.empty([size*3, size*3],dtype=str)
    output_grid[:] = '#'
    c = size*size # number of cells to be visited
    i = rd.randrange(size)
    j = rd.randrange(size)
    while np.count_nonzero(grid) < c:
  
        # visit this cell
        grid[i,j] = 1

        w = i*3 + 1
        k = j*3 + 1
        output_grid[w,k] = ' '

        can_go = [1,1,1,1]

        if i == 0:
            can_go[0] = 0
        if i == size-1:
            can_go[2] = 0
        if j == 0:
            can_go[3] = 0
        if j == size-1:
            can_go[1] = 0
        
        # it makes sense to choose neighbour among available directions
        neighbour_idx = np.random.choice(np.nonzero(can_go)[0]) # n,e,s,w

        if neighbour_idx == 0:
            # has been visited?
            if grid[i-1,j] == 0:
                # goto n
                output_grid[w-1,k] = ' '
                output_grid[w-2,k] = ' '
            i -= 1
                    
        
        if neighbour_idx == 1:
            if grid[i,j+1] == 0:
                # goto e
                output_grid[w,k+1] = ' '
                output_grid[w,k+2] = ' '
            j += 1
          
        if neighbour_idx == 2:
            if grid[i+1,j] == 0:
                # goto s
                output_grid[w+1,k] = ' '
                output_grid[w+2,k] = ' '  
            i += 1
        

        if neighbour_idx == 3:
            # goto w
            if grid[i,j-1] == 0:
                output_grid[w,k-1] = ' '
                output_grid[w,k-2] = ' '
            j -= 1
            
    return output_grid