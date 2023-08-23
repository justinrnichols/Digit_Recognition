# Machine Learning Digit Recognition Using SVD Machine Learning Algorithm

## Team Members

Webster Gordon - wgordo1@lsu.edu
Mollee Swift - mswift5@lsu.edu
Justin Nichols - jnich56@lsu.edu


## Dataset

We use a Handwritten digits USPS dataset as sample data.
The dataset can be found at https://www.kaggle.com/bistaumanga/usps-dataset
The dataset has 7291 train and 2007 test images in h5 format.
The images are 16x16 grayscale pixels matrices.


## Format of Data

The dataset includes training and testing images. 
1. The X_train and X_test columns contain the digits as vectors that are represented 
as flattened arrays with a size of 256 = 16x16 grayscale pixels.
2. The y_train and y_test contain the actual class of each digit (i.e., values 0-9.)


## Technologies and Packages/Modules Used

1. Python using the Anaconda distribution
2. Scipy
3. Skikit-Learn
4. Numpy
5. h5py
6. matplotlib
7. Microsoft PowerPoint


## Project Description

Using the USPS handwritten digits dataset, we used SVD to train our model 
to learn the numbers 0-9. The goal is to then give our model a new data sample
containing an unseen written single digit and test if the model can accurately
predict the value of the digit. We decided to use SVD over PCA because SVD is
more efficient than PCA when working with dense data.


## Digit Classification Steps 

1. Split X_train and store in a dictionary, where the key is the y_train 
value (digit) and the value is an array of all the images corresponding to
that digit. For example, the key 0 will hold all the images that are apart
of the digit class 0.
2. Perform SVD on each digit class array. For example, SVD will collectively
be performed on all the images that are associated with digit 0.
3. Find the N rank 1 matrix for each digit.
4. Use the Least Square equation to find the accuracy for N singular values.


## Data Analysis

- Print train and test shapes
- Show the first 10 images in the dataset
- Graph each digit sample distribution
- Output all split image data
- Output all image data for a specific digit
- Output SVD matrix data
- Output the N rank1 approximation images
- Output 10 clearest images for a specific digit
- Graph singular values for each number
- Output the accuracy percentage for N singular values
- Output the individual digit accuracy
- Print the classification report
- Output the misclassified images


## How to Run Instructions

1. The easiest way to run the program is to put the project in visual studio code
and run it using the anaconda distribution as it comes with the necessary packages.
(You can download anaconda here: https://www.anaconda.com/)
2. Otherwise, install any missing packages and run it using Python 3.
