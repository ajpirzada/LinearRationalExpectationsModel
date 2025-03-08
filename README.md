# Linear Rational Expectations model
This project shows how to use Chris Sim's gensys algorithm to solve a linear rational expectations model. 

## Files

- The file **REmodelToolkit.py** includes the Python version of Chris Sims gensys algorithm.
- The file **NKmodelSimulations.ipynb** sets up a three equation New Keynesian model in the form that is consistent with the gensys alogorithm. It then uses the algorithm to solve the model.

## Usage
You can either run the project on your machine using softwares like VS Studio, Anaconda etc. or on Google Colab.

### 1. On your Machine
Have both the files in the same folder before running the file **NKmodelSimulations.ipynb**. Make sure that the toolkit file has exactly the same name i.e. **REmodelToolkit.py**.

### 2. Using Google Colab
To run the **NKmodelSimulations.ipynb** file on Google Colab, follow these steps: 

1. Go to your *google drive* (online).
2. Under *My Drive*, there will be a folder called *Colab Notebooks*. If not, create one.
3. Upload both the files to this folder. Make sure that you upload the toolkit file with exactly the same name i.e. **REmodelToolkit.py**.
4. Open the **NKmodelSimulations.ipynb** file.
5. Connect and run the cells under SET PATH FOR TOOLKIT IN GOOGLE DRIVE.

```python
# This line imports the necessary functionality to interact with your Google Drive from within the Colab notebook.
from google.colab import drive

# This line makes sure that your Google Drive files are accessible within the notebook
drive.mount('/content/drive')

# This command lists the files and directories inside the "Colab Notebooks" folder located in your Google Drive. You should see REmodelToolkit.py here.
!ls "/content/drive/My Drive/Colab Notebooks"

# These two lines ensure Python can find and import custom modules (e.g REmodelToolkit.py) located within your 'Colab Notebooks' folder in Google Drive, making them available to use in the notebook.
import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks')
```

  If you get an error, it could be because: 
  - You did not give all the permissions when you were prompted to do so.
  - You uploaded the toolkit file with a different name than **REmodelToolkit.py**.
  - You did not upload the toolkit file in the Colab Notebooks folder in My Drive.

Fix it. You are now good to go!

## Chris Sims (2001) Gensys algorithm

The function `gensys` solves a **linear rational expectations model** of the form:

```latex
G_{0} * y_{t} = G_{1} * y_{t-1} + C + \Psi * \epsilon_{t} + \Pi * \eta_{t}
```

Much of the hardwork goes into writing your linear rational expectations model in this form. Section 1 in **NKmodelSimulations.ipynb** shows how to do it for a three equation New Keynesian model. The trick is to define auxiliary variables for each of the forward looking variable in the model i.e. for expected inflation and expected output gap. Once you have you matrices (G0, G1, C, Psi, PI), you can use the gensys function from REmodelToolkit.py to solve your RE model. The solution to your model takes the following form:

y_t = G1_sol * y_{t-1} + C_vec + Impact * epsilon_t

where:
- `G1_sol` is the transition matrix mapping past states to current states.
- `C_vec` is the vector of constants.
- `Impact` determines how shocks influence the system.
- `gev` contains the **generalised eigenvalues**, which help assess stability.
- `eu` (existence and uniqueness flags) indicate whether a unique solution exists.
