# Rational Expectations model
This project shows how to use Chris Sim's gensys algorithm to solve a linear rational expectations model. 

## What files do we have

- The file $\textit{REmodelToolkit.py}$ includes the Python version of Chris Sims gensys algorithm.
- The file $\textit{NKmodelSimulations.ipynb}$ sets up a three equation New Keynesian model in the form that is consistent with the gensys alogorithm. It then uses the algorithm to solve the model.

## Usage
You can either run the project on your machine using softwares like VS Studio, Anaconda etc. or on Google Colab.

### On your machine
Have both the files in the same folder before running the file $\textit{NKmodelSimulations.ipynb}$. Make sure that the toolkit file has exactly the same name i.e. $\textit{REmodelToolkit.py}$.

### Using Google Colab
To run the $\textit{NKmodelSimulations.ipynb}$ file on Google Colab, follow these steps: 

1. Go to your google drive (online).
2. Under My Drive, there will be a folder called Colab Notebooks. If not, create one.
3. Upload both the files to this folder. Make sure that you upload the toolkit file with exactly the same name i.e. $\textit{REmodelToolkit.py}$.
4. Open the $\textit{NKmodelSimulations.ipynb}$ file.
5. Connect and run the cells under SET PATH FOR TOOLKIT IN GOOGLE DRIVE.
   If you get an error, it could be because: i) you did not give all the permissions when you were prompted; ii) you uploaded the toolkit file with a name than $\textit{REmodelToolkit.py}$; iii) you did not upload the toolkit file in the Colab Notebooks folder in My Drive. 
7. Import the libraries.

You are now good to go!

