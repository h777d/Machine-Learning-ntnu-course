# Machine Learning Course
Computer tasks for Machine Learning course for Signal Processing at NTNU ELSYS department

**Important**: we do not use `.ipynb` files (jupyter notebooks) in this repository because they contain the results of execution as well as code, and make version control impossible. Instead, we converted the `.ipynb` files into `.py` files using `jupytext`.

These `.py` files can be used directly in Jupyter, but will not save intermediate results, making it possible to track changes in the code.

## To modify the tasks:
* open the `.py` file in jupyter and make your modifications
* save the `.py` file and add/commit your changes to the repository
* **DO NOT** add the `.ipynb` file to the git repository (the .gitignore file takes care of that)

## To add new assignment files:
* install jupytext through `pip install jupytext` or `conda install -c conda-forge jupytext`
* create a new jupyter notebook. The file will be saved as `.ipynb` file
* use jupytext to convert the `.ipynb` file into `.py` with `jupytext --to py --output myfile.py myfile.ipynb`
* add and commit **only** the `.py` file.

## to link the `.ipynb` to the `.py` file
(this is only necessary if you want to save temporary results in the jupyter notebook)
* open the `.ipynb` and click on Edit/Edit Notebook Metadata
* add the line `"jupytext": {"formats": "ipynb,py"},` after the first `{` as in
```
{
  "jupytext": {"formats": "ipynb,py"},
  "kernelspec": {
    (...)
  },
  "language_info": {
    (...)
  }
}
```

## To release the tasks (to be distributed to the students)
We could distribute the `.py` files directly. However, to avoid confusion among the students, it is better to convert them back to `.ipynb` files before distribution. It is a good idea to attach the current date to the filename, in case we need to update the files to fix things later. For example:
```
jupytext --to notebook --output release/task_0_2022-08-30.ipynb task0_Python_introduction/task_0.py 
```
Because ipynb files are ignored by git, you will need to add the releases to the repository with, for example
```
git add -f release/task_0_2022-08-30.ipynb
```
