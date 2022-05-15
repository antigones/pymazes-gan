# pymazes-gan
Maze generation with GAN

**How to run the script**

1) Create a virtual environment (Optional)
2) Install the requirements:
`pip install -r requirements.txt`

3) Generate input images to train the GAN on:
`python augment_maze_img.py`

4) Train the GAN:
`python tf_gan_maze.py`

5) By default, the script prints one random sample to generated.png and save an animation to display training process to dcgan.gif

**Loading weights without retraining**

Comment the line:

`train(train_dataset, EPOCHS)`
`make_animation()`

**Notes**

Maze size should match variational GAN layers architecture.
In augment_maze_img.py:
`size=12`
In tf_gan_maze.py:
`size = 12 * 3`
