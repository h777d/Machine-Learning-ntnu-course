# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: TTT4185
#     language: python
#     name: ttt4185
# ---

# # TTT4185 Machine learning for Speech technology
# ## Voluntary computer assignment 0: Python introduction

# ## Installation

# ### Install Python 3 through Miniconda
#
# Different projects may use different versions of Python and its packages. As a best practice, each project should have its own virtual environment. To manage this, we recommend you to install Miniconda, the minimal installer for conda, an open-source package and environment manager for many programming languages, including Python.
#
# You can find the suitable Miniconda installer (remember to select Python 3) for you here:
# https://docs.conda.io/en/latest/miniconda.html
#
# PS: This installation tutorial assumes that you have Miniconda installed, but if you do not want to use prompt commands, then we recommend you to install Anaconda instead. Anaconda includes everything that Miniconda does, plus many packages (a lot more than needed for this course) and the Anaconda Navigator, allowing you to manage environments and packages without using the prompt.
# You can download Anaconda here: https://www.anaconda.com/products/individual

# ### Create an environment for this course
#
# Use the terminal (or the Anaconda prompt) to run the following commands.
#
# To create a new environment:
#
# - conda create --name ttt4185
#
# To verify if it was correctly installed, check if the new environment appears here:
#
# - conda env list
#
# To learn more about environment management, go here:
# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

# ### Install packages
#
# There are three main channels through where we can install packages in a conda environment: pip, conda defaults and conda-forge.
#
# To install the primary packages for this course, we use conda defaults. Remember to always activate your environment first.
# - conda activate ttt4185
# - conda install PACKAGE
#
# *Task: Activate the environment you've created for the course and install numpy, pandas, matplotlib and scikit-learn using conda defaults*
#
# You might need to install extra packages throughout the course. If you can not find the desired package on conda defaults, try using conda-forge or pip.
#
# To install packages from conda-forge:
# - conda install -c conda-forge PACKAGE
#
# To install packages through the pip command, you should first install it:
# - conda install pip
# - pip install PACKAGE

# ### Install Jupyter notebook
#
# We use Jupyter notebooks to combine code, outputs and text in the same file. This gives us a lot more interaction with the data and an easier time explaining what has been done and documenting the code.
#
# To install Jupyter on the new environment:
# - conda activate ttt4185
# - pip install jupyter
#
# To be able to use the created environment in Jupyter, we need to create a kernel for it:
# - conda install ipykernel --name ttt4185
# - python -m ipykernel install --name TTT4185 --user
#
# Now you are able to launch notebook through the following command:
# - jupyter notebook
#
# From now on, you can open the .ipynb files and select kernel to the created environment "TTT4185".
#
# A Jupyter notebook is composed of cells (either text or code based) and you can run cells by pressing the "Run" button or shift+return. If you are unfamiliar with Jupyter, go to Help>User Interface Tour for a quick presentation.

# ### Import packages and modules
#
# Python is based on packages and modules that must be loaded to become available. So to use a function from a package/module on Python, you need to import it first. Example:

import math
math.sqrt(4)

# If you're using a package many times, it's recommended to rename it using "as". Please follow the common existent abbreviations such as:

import numpy as np
import pandas as pd

# If you're using specific modules of a package, you can import only these modules by using "from". Examples:

# +
# instead of this:
import numpy as np
np.sqrt(4), np.cos(np.pi)

# you can use this:
from numpy import sqrt, sin, cos, pi
sqrt(4), cos(pi)
# -

# *Try: Import and use the functions "exp" and "log" from NumPy.*



# ## Python data types

# ### Integer and float
#
# Consider the following variables:

a = 2
b = 2.
c = 2.6
a,type(a),b,type(b),c,type(c)

# *Try: Use the comparison operator "==" to verify that "a" and "b" have the same value but not the same type.*



# You can transform a float to an integer using the functions "int", "round" and also "ceil" and "floor" from NumPy.

int(b), int(c), round(c), np.floor(c), np.ceil(c)

# *Try: Divide 9 and 10 by 3 using the different operators "/" and "//". What are the differences between the results?*



# ### Set, list and range
#
# Sets and lists are collections of objects with two main differences. While sets are unordered and their elements are unique, lists are ordered and can contain repeated elements.

aset = {0, 5, 2, 0, 2, 5} # the repeated elements are ignored
alist = [0, 5, 2, 0, 2, 5]
aset, type(aset), alist, type(alist)

# To get their lenghts, use function "len":

len(aset), len(alist)

# To add new elements:

aset.add(3) # the set is reordered
alist.append(3) # new element inserted in the end of the list
aset, alist

# Sets and lists are iterables, so you can use them in loops with "for".

for element in aset:
# for element in alist:
    print(element)

# It's very common to use the function "enumerate" to iterate over a set or list, so you keep the track of the element index.

# for index, element in enumerate(aset):
for index, element in enumerate(alist):
    print(index, element)

# You may also use the function "range" which returns a sequence of numbers, but it is not simply a list of items. Both have the same effects used in a for loop, but they are different kind of objects.
#
# *Try: Print range(10) and list(range(10)) and check their types to see the difference.*



# ### Dictionary
#
# As lists and sets, dictionaries are also collections of objects. The main difference is that dictionary's elements are stored and accessed through keys, instead of indexing.
#
# Take a look at a dictionary example:

mydict = {
    'key':'value',
    'code':'ttt4185',
    'name':'machine learning',
    'n_students':60,
    'computer_assignments':['python','speech','bayes','dnns']
}
mydict

# To access an item's value, use its key:

mydict['key']

# You can also get all keys or values separately:

mydict.keys(),mydict.values()

# or as key-value pairs:

mydict.items()

# *Try: Create two lists with the same lenght (one for keys and one for values) and use them to create a dictionary. To add a new item to a dictionary, you can use "mydict[newkey]=newvalue".*



# ## Functions
#
# Functions are used to avoid repetition of code, making it reusable and consequentely more organized.
#
# Let's define a simple function as an example of the syntax. The arguments "exponent" and "base" are examples of named and default arguments, respectively. While named arguments are necessary to run a fuction, if a default argument is not given, the function uses its default value.

def mypowerfunction(exponent, base=10):
    result = base**exponent
    return result


# Using default value:

mypowerfunction(2)

# Using a different base:

mypowerfunction(2,base=2)

# *Try: Create a function that greets someone using their name as argument with default value "you". Run it with and without a name to check if it's working as expected.*



# ### Global versus local variables
#
# What differs a variable to be global or local is if the variable is defined outside or inside a function, respectively. See the following example:

# +
x = 'global'

def afunction():
    x = 'local'
    return x

print(x)
print(afunction())
# -

# To avoid confusion, use different names for global and local variables.

# ### How to use your own .py file
# If you want to use packages and functions from your own .py files on Jupyter notebook, you just need to import your code. If there is any package or function being imported in this file, it will also be available in the notebook.
#
# *Try: Create a file called "mycode.py" containing a simple function called "myfunction", use the command "import mycode" and run the function here by using "mycode.myfunction(arguments).*



# If there is an error in your .py file, you can use the module "reload" from "importlib" to debug it in here.
#
# *Try: Import importlib, add an error on you file on purpose and use "importlib.reload(mycode)" to get the error message.*



# ## Plotting with Matplotlib

# Import matplotlib and use inline to show the graphs as notebook outputs instead of opening separate windows.

import matplotlib.pyplot as plt
# %matplotlib inline

# For a simple single axis plot:

# +
x = np.linspace(0,4*pi,100)

fig, ax = plt.subplots()
ax.plot(x,sin(x))
# -

# For multiple plots, you should add legend and you can also specify colors and line patterns.

fig, ax = plt.subplots()
ax.plot(x,sin(x),'r',linestyle=':',label='sin')
ax.plot(x,cos(x),'b',linestyle='--',label='cos')
ax.legend()

# For multiple axes:

# +
fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)

axs[0].plot(x,sin(x))
axs[0].set(ylabel='sin(x)')

axs[1].plot(x+pi,sin(x+pi))
axs[1].set(ylabel='sin(x+pi)')
# -

# *Try: Plot $sin(x)$ and $2sin(x)$ side by side (instead of one on top of the other, like the previous example).*



# To save a figure that you've created:

fig.savefig('myawesomeplot.pdf')

# ## NumPy
#
# NumPy is a Python package to work with multidimensional vectors. The use of matrices are deprecated, so we use a 2-dimensional array instead.
#
# Example: you can write the matrix $A = \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix}$ as:

A = np.array([[1, 2],
              [3, 4]])

# Since numpy arrays are row-based, to access a specific row, you just need to use the index:

print('First row:', A[0]) # or A[0,:]
print('Second row:', A[1]) # or A[1,:]

# To select a specific column:

print('First column:', A[:,0])
print('Second column:', A[:,1])

# When applying functions to rows or columns, you should specify which axis you want to use. Rows are axis 1 and columns axis 0.

print("Mean of the rows' elements:", A.mean(axis=1))
print("Mean of the columns' elements:", A.mean(axis=0))

# To multiply two matrices, you should use the "dot" operator which calculates the dot product between two multidimensional arrays.

A.dot(A) # or np.dot(A,A)

# *Try: Multiply A by A using the operator asterisk "$*$" and guess what calculations are performed.*



# *Try: Consider the system $Ax = b$. Find the least-squares solution $\hat{x} = (A^TA)^{-1}A^Tb$ using $A$ and $b$ given in the next code cell.*
#
# *To calculate the transpose and inverse of a matrix, use matrix.T and np.linalg.inv(matrix).*

A = np.array([[0,1],[1,1],[2,1]])
b = np.array([[6],[0],[0]])



# For those who are used to Matlab, we recommend checking this link for differences between Matlab and NumPy:
# http://mathesaurus.sourceforge.net/matlab-numpy.html

# ## Pandas
#
# Pandas is a Python library commonly used in data science. Its main object is called a dataframe which is a 2d-structure, easily manipulated for data exploration.

# here we use a dataset imported from sklearn
from sklearn import datasets
data = datasets.load_wine()
df = pd.DataFrame(data.data, columns = data.feature_names)

# To view the first or last records of the data:

df.head()
# df.tail()

# To get some information and statistics of the data:

df.info()

df.describe()

# ### Pandas indexing
# There are many different ways of indexing on Pandas: ".", "[]", ".loc[]", ".iloc[]", ".ix[]", what can sometimes cause confusion.
#
# Use "." or "[]" to select a column:

df.alcohol.head()

df['alcohol'].head()

# You can also use "[]" to select multiple columns:

df[['alcohol','magnesium']].head()

# Use ".loc[]" to select columns and elements by their labels:

df.loc[0]

df.loc[5:10, ['alcohol','magnesium']]

# Use ".iloc[]" to select elements by their positions instead of labels:

# to show the difference, we will use only a part of the dataframe
partofdf = df[['alcohol','magnesium']].tail(10)
partofdf

partofdf.iloc[3:5]

# The operator ".ix[]" works primarily as ".loc[]", i.e., label based. However, if the indexes are integers, ".ix[]" considers the position.
#
# Since it can be a bit confusing, we recommend you to avoid the use of ".ix[]" to instead use ".loc[]" and ".iloc[]", explicitly showing if you are filtering by label or position, respectively.

# ### Filtering records by column condition
#
# Here is an example on how to filter the dataframe's records by a column condition. We are selecting all the records that has color intensity greater than 5.

df[df['color_intensity']>5]

# *Try: Filter the dataframe by two conditions instead of only one. Use each condition into parentheses and the operator "&" (and) or "|" (or) to connect them.*



# ### Your time to explore
#
# *Try: Load one of the sklearn datasets (https://scikit-learn.org/stable/datasets/) and explore it by for example plotting histograms (plt.hist(x) or df.hist()) and scatter plots (plt.scatter(x1,x2)) of the variables. Can you see any correlation between them?*







# ## Last tips: Solving mistakes on code
#
# The best way to not get an error is, of course, to avoid mistakes. Get used to check if each new line you write looks correct before jumping to the next one. But even doing it, we are still going to bump into mistakes.
#
# If your code is raising an error message, you can go directly to the core problem. In case you don't understand what the error means, search it on Stack overflow for example https://stackoverflow.com/. Python has a huge community and most errors you encounter have already been discussed there.
#
# If there are no error messages, but you are not getting the expected result, there are also a few tips to debug your code and find mistakes:
# - print variables and look for inconsistences
# - check if variables' types are correct
# - check arrays' shapes
# - plot intermediary results
# - break complex expressions into simpler ones
