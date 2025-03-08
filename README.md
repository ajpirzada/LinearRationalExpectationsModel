# Linear Rational Expectations models
This project makes available the Python version of Chris Sims (2001) gensys algorithm to broader community. I also show how the algorithm can be used to solve a New Keynesian model and further analyse the economy.
## Files

- The file **REmodelToolkit.py** includes the Python version of Chris Sims gensys algorithm.
- The file **NKmodelSimulations.ipynb** sets up a three equation New Keynesian model in the form that is consistent with the gensys alogorithm. It then uses the algorithm to solve the model.

## Usage
You can either run the project on your machine using softwares like *VS Studio*, *Anaconda* etc. or on *Google Colab*.

### 1. On your Machine
Have both files in the same folder before running the script **NKmodelSimulations.ipynb**. Make sure that the toolkit file has exactly the same name i.e. **REmodelToolkit.py**. You do not need to run the cells under SET PATH FOR TOOLKIT IN GOOGLE DRIVE. Instead, start from IMPORT PYTHON LIBRARIES.

### 2. Using Google Colab
To run the **NKmodelSimulations.ipynb** file on *Google Colab*, follow these steps: 

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

```math
G0 y_{t} = C + G1 y_{t-1} + \Psi \epsilon_{t} + \Pi \eta_{t}
```

where $y$ is a vector of endogenous variables and $\epsilon$ is a vector of exogenous shocks. $\eta$ is a vector of expectation errors. Much of the hardwork goes into writing your linear rational expectations model in this form. Section 1 in **NKmodelSimulations.ipynb** shows how to do it for a three equation New Keynesian model. The trick is to define auxiliary variables for each of the forward looking variable in the model i.e. for expected inflation and expected output gap. Once you have your matrices (G0, G1, C, $\Psi$, $\Pi$), you can use the gensys function from **REmodelToolkit.py** to solve your RE model. The solution to your model takes the following form:

```math
y_{t} = C_{vec} + G1_{sol} y_{t-1} + Impact \epsilon_{t}
```

where:
- `G1_sol` is the transition matrix mapping past states to current states.
- `C_vec` is the vector of constants.
- `Impact` determines how shocks influence the system.
- `gev` contains the **generalised eigenvalues**, which help assess stability.
- `eu` (existence and uniqueness flags) indicate whether a unique solution exists.

## EXAMPLE: New Keynesian model

The **NKmodelSimulations.ipynb** file uses the gensys algorithm to solve a standard three equation New Keynesian model of the form:

1. **IS curve**: $x_{t} = \beta E_{t}x_{t+1} - \varphi(i_{t} - E_{t}\pi_{t+1} - \bar{r}) - a_{t}$
2. **NKPC curve**: $\pi_{t} - \bar{\pi} = \beta E_{t}(\pi_{t+1} - \bar{\pi}) + \lambda x_{t} + u_{t}$
3. **Taylor rule**: $i_{t} = \bar{r} + \bar{\pi} + \chi_{\pi}(\pi_{t} - \bar{\pi}) + \chi_{x}x_{t} + m_{t}$

where $x_{t}$ is output gap, $\pi_{t}$ is inflation, and $i_{t}$ is the nominal interest rate. The model economy is also hit by three shocks. These are productivity shocks ($a_{t}$), cost shocks ($u_{t}$), and monetary policy shocks ($m_{t}$). Each shock follows an AR(1) process of the form:


4. **cost shock**: $u_{t} = \rho_{u}u_{t-1} + \epsilon_{t}^{u}$
5. **monetary shock**: $m_{t} = \rho_{m}m_{t-1} + \epsilon_{t}^{m}$
6. **productivity shock**: $a_{t} = \rho_{a}a_{t-1} + \epsilon_{t}^{a}$

where $\epsilon_{t}^{j}$ are i.i.d. shocks ($j \in \{ u, \pi, i \}$) with mean zero and standard deviation, $\sigma_{\hat{j}}$.

The **NKmodelSimulations.ipynb** file is divided into several sections. Arguably, the most critical is section 1. Recall, gensys requires the model to be in the following form:

```math
G0 y_{t} = C + G1 y_{t-1} + \Psi * \epsilon_{t} + \Pi \eta_{t}
```

### Section 1: Setup the matrices defining your model

Section 1 is where you setup the matrices that define your model i.e. G0, G1, $\Psi$, and $\Pi$. Note, however, the NK model defined above has two forward looking variables, $E_{t}x_{t+1}$ and $E_{t}\pi_{t+1}$. In contrast, the gensys algorithm only takes as input matrices associated with current and previous period variables, $y_{t}$ and $y_{t-1}$, and those associated with $\epsilon_{t}$ and $\eta_{t}$. 

To transform the model, you should define an auxiliary variable for each of the forward looking variables:

```math
z_{t}^{x} = E_{t}x_{t+1}
```
```math
z_{t}^{\pi} = E_{t}\pi_{t+1}
```

The transformed version of the model has 8 equations: 3 for $x$, $\pi$, and $i$; 3 for the shocks; and 2 for the auxiliary variables. Let's write the transformed model:

1. **Auxiliary variable 1**: $x_{t} = z_{t-1}^{x} + \eta_{t}^{x}$
2. **Auxiliary variable 2**: $\pi_{t} = z_{t-1}^{\pi} + \eta_{t}^{\pi}$
3. **IS curve**: $x_{t} + \varphi i_{t} + a_{t} - \beta z_{t}^{x} - \varphi z_{t}^{\pi} = \varphi\bar{r}$
4. **NKPC curve**: $-\lambda x_{t} + \pi_{t} - \beta z_{t}^{\pi} - u_{t} = (1-\beta)\bar{\pi}$
5. **Taylor rule**: $-\chi_{x}x_{t} - \chi_{\pi}\pi_{t} + i_{t} - m_{t} = \bar{r} + (1 - \chi_{pi})*\bar{\pi}$
6. **Cost shock**: $u_{t} = \rho_{u}u_{t-1} + \epsilon_{t}^{u}$
7. **Prod shock**: $a_{t} = \rho_{a}a_{t-1} + \epsilon_{t}^{a}$
8. **Monetary shock**: $m_{t} = \rho_{m}m_{t-1} + \epsilon_{t}^{m}$

Note, the two equations for auxiliary variables also include expectation errors, $\eta_{t}^{x}$ and $\eta_{t}^{\pi}$. These are critical to solving a rational expectations model. Gensys solves the linear rational expectations model such that the expectations error equal zero on average. 

We can now write this system of equations in matrix form as is required by the gensys algorithm. The vector $y$ includes ${[x, \pi, i, u, a, m, \eta^{x}, \eta^{\pi}]}^{T}$.

### Section 2-7: Solving and analysing the model economy

**Section 2** uses the gensys algorithm to solve the model. **Section 3** explicitly reports the solution for each endogenous variable in the model. **Section 4** reports the variance-covariance matrix. **Section 5** plots impulse responses to each of the shock considered in the model. **Section 6** simulates the model economy by generating a series of i.i.d. shocks for all three shocks considered here. The i.i.d. shocks are drawn from a normal distribution with mean zero and standard deviation, $\sigma_{\hat{j}}$. **Section 7** reports variance decomposition over different forecast horizon. Finally, **section 8** reports historical decomposition showing the contribution of each of the shock in driving fluctuations in endogenous variables over the simulation period.
