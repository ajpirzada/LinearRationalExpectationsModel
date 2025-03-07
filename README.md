# Rational Expectations model
This project shows how to use Chris Sim's gensys algorithm to solve a linear rational expectations model. 

## What files do we have

- The file **REmodelToolkit.py** includes the Python version of Chris Sims gensys algorithm.
- The file **NKmodelSimulations.ipynb** sets up a three equation New Keynesian model in the form that is consistent with the gensys alogorithm. It then uses the algorithm to solve the model.

## Usage
You can either run the project on your machine using softwares like VS Studio, Anaconda etc. or on Google Colab.

### 1. On your Machine
Have both the files in the same folder before running the file **NKmodelSimulations.ipynb**. Make sure that the toolkit file has exactly the same name i.e. **REmodelToolkit.py**.

### 2. Using Google Colab
To run the **NKmodelSimulations.ipynb** file on Google Colab, follow these steps: 

1. Go to your google drive (online).
2. Under My Drive, there will be a folder called Colab Notebooks. If not, create one.
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

```markdown
If you get an error, it could be because: i) you did not give all the permissions when you were prompted; ii) you uploaded the toolkit file with a different name than $\textit{REmodelToolkit.py}$; iii) you did not upload the toolkit file in the Colab Notebooks folder in My Drive. 
7. Import the libraries.

You are now good to go!

