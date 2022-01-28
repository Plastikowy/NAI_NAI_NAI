We have used Neural Style Transfer for teaching our "AI" how to paint an image that keeps content of base image, but uses style of style image

1. You can find our code in main.py file.
2. Packages required to run program u can find at the beginning of the file main.py or bellow: <br/>
Prerequisites: <br/>
Before you run program, you need to install: Numpy, matplotlib, Pillow, IPython and TensorFlow  packages. <br/>
You can use for example use PIP package manager do to that: <br/>
   - pip install numpy <br/>
   - pip install Pillow <br/>
   - pip install ipython(we recommend v7.31.1) <br/>
   - pip install opencv-python <br/>
   - pip install tensorflow <br/>

3. Before running program set variables such as:
   - style_path (pick from variables starting as "style_*")
   - smallSizeEnabled (flag if we want output image by smallest size set by us, otherwise output image has size of smaller one content/style image)
   - smallSize (lowest possible size of output image, larger = longer learning process)
   - number_of_iterations (larger number = longer learning process)
   - every_which_iteration_save_to_file 

You can add your styles if you want to, but it is required to create new variable or change existing one

4. Image created by program is saved with name template "generated_{iteration}_{style_path_without_extension}.png"

Examples:

![Alt text](./generated_0_The_Great_Wave_off_Kanagawa.png?raw=true "Iteration 0") 
![Alt text](./generated_150_The_Great_Wave_off_Kanagawa.png?raw=true "Iteration 150") 
![Alt text](./generated_350_The_Great_Wave_off_Kanagawa.png?raw=true "Iteration 350") <img src="https://raw.githubusercontent.com/Plastikowy/NAI_NAI_NAI/lab7/lab7/The_Great_Wave_off_Kanagawa.jpg" width="350" height="231" />

![Alt text](./generated_0_comic_style.png?raw=true "Iteration 0") 
![Alt text](./generated_150_comic_style.png?raw=true "Iteration 150") 
![Alt text](./generated_350_comic_style.png?raw=true "Iteration 350") <img src="https://raw.githubusercontent.com/Plastikowy/NAI_NAI_NAI/lab7/lab7/comic_style.jpg" width="350" height="231" />

![Alt text](./generated_0_Gouache-style.png?raw=true "Iteration 0") 
![Alt text](./generated_150_Gouache-style.png?raw=true "Iteration 150") 
![Alt text](./generated_350_Gouache-style.png?raw=true "Iteration 350") <img src="https://raw.githubusercontent.com/Plastikowy/NAI_NAI_NAI/lab7/lab7/Gouache-style.jpg" width="350" height="231" />
