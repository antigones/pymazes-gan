import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import maze_utils

import numpy as np
import random as rd

def main():
  
    n_samples = 5000
    n=1
    
    size=12
    img_mode = 'RGB'

    for i in range((n_samples//3)*2):
        p = rd.uniform(0.3,0.8)
        grid = np.random.binomial(n,p, size=(size,size))
        grid = maze_utils.preprocess_grid(grid, size)
        output = maze_utils.carve_maze(grid, size)

        s = ""
        img_out = []
        for elm in output:
            s = "".join(elm)
            img_out.append(list(map(int,s.replace("#","1").replace(" ","0"))))
        arr = np.asarray(img_out)
        arr[arr==0] = 0
        arr[arr==1] = 255
        im = Image.fromarray(arr)
        if im.mode != img_mode:
            im = im.convert(img_mode)
        imageio.imsave("imgs/train/"+str(i)+".gif", im)

        
    for i in range(n_samples//3):
        p= rd.uniform(0.3,0.8)
        grid = np.random.binomial(n,p, size=(size,size))
        grid = maze_utils.preprocess_grid(grid, size)
        output = maze_utils.carve_maze(grid, size)

        s = ""
        img_out = []
        for elm in output:
            s = "".join(elm)
            img_out.append(list(map(int,s.replace("#","1").replace(" ","0"))))
        arr = np.asarray(img_out)
        arr[arr==0] = 0
        arr[arr==1] = 255
        im = Image.fromarray(arr)
        if im.mode != img_mode:
            im = im.convert(img_mode)
        imageio.imsave("imgs/test/"+str(i)+".gif", im)

if __name__ == '__main__':
    main()

