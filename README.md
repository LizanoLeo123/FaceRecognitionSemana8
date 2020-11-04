# Face Recognition model using PCA in Python

## Execution

You can open the project in a python IDE or run from command prompt, “python load_image_data.py -d ./Dataset”, this will  generate the data.csv file required by the other file.
Then run “python face_recog_pca.py”

## Components of this project

### Dataset

The dataset consist of a folder called Dataset, which contains 10 folders with 10 images each, wich are photos from the course classmates. The images are 512x512 pixels and there is a full dataset on Google Drive [here](https://drive.google.com/drive/folders/1IhufSGU5Llbp_3pO8lw2A4g3cz12IlRB)

### Load images from dataset

There are two files that makes the first part of the job of this project.
The file called "load_image_data.py" is the one that navigates trough the local dataset and converts each image into grayscale, and after that it turn them into frameworks so it could be readen from a .csv file.
The csv file contains the grayscale value of each image and in the end of each row, it has a "target" column, which is the name of the person in that image. So it can be readen by the "face_recog_pca.py" file.

### PCA model and visualization

Let's begin talking about PCA, also known as Principal Component Analysis, in the most perfect case, PCA can be understood in 2D data analysis, by grabbing values of (x,y) in a plane.

Face recognition can be aplied the same way, but of course, faces have more than x and y factors that can be analyzed to know who is which person, eyes shape, eyebrows shape, nose, mouth, bear, etc.

PCA in face recognition is based on two main things, calculate the gray scale matrix of an image, and then calculate the eigen faces, also known as principal components of a face.

#### GrayScale matrix

The load image file makes this labor, it grabs each image of the dataset, it loads the image in grayscale and converts every each row of pixels into numeric values of a dataframe, and then saves it in csv

This is the way to reduce some dimensions of the images, now we can actually plot these new pixel value matrixes using matplotlib and that's how we can actually see the faces.

#### Eigen Faces

Now we create our PCA model, which is an extension from the sklearn package.
PCA in image recognition with our image matrix parameters, makes something called eigenfaces, which are a bunch of random images mixed and modified with some filters sklearn defines.
This spectrums can be analyzed as the PRINCIPAL COMPONENTS of our images, and that's how this model trains itself and use this eigenfaces to make predictions.

This project also makes a plot, showing how the amount of principal components(Eigen Faces) selected to make the model (we worked with 30) contains the most of the principal components and shapes of the images.
This is how PCA is applied in face recognition, the PCA gives us a reference of which of this eigen faces we should us. And by using 30 and not 82(the limit registered by the program), we have a lot of dimension reduction.

The first 30 eigen faces contain more than 90% of the principal components of these images.

Then we show the eigen faces and after that we display a summary about the precision of the model. It lacks precision, probably because the dataset comes from different sources and the only common pattern is the size of the images, also the eigen faces algorithm is very fragile to lights.

# The Perceptron

The Perceptron infographic is in the Perceptron PDF.

## Authors

### Danilo Chaves and Leonardo Lizano
