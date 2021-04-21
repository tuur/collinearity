### Contact 
###### Date: 14-04-2021
###### Email: aleeuw15@umcutrecht.nl


#### Preprint
This repository corresponds to the experiments and results reported in the preprint article at https://arxiv.org/abs/2101.01603

#### This directory contains:
- The code used to conduct the experiments to compare different prediction modeling methods in settings with different levels of collinearity among predictors.

#### For patient privacy reasons, it does NOT contain:
- The data used to setup the simulations.

#### List of subdirectories:
- _scripts_: This directory contains the bash scripts that were used to call the python code for the experiments reported in the article.
- _yamls_: This directory contains config files (yaml format) that specify the characteristics of each of the simulated settings.
- _code_: This directory contains (1) the python code used to conduct the simulations: develop and evaluate the modelling methods, and (2) an ipython notebook that was used to merge the output from the experiments into the tables and figures used in the article (in either pdf or LaTex format).


#### Note: 
As the data is not shareable, it is not possible to fully replicate the exact experiments from the article with the same data. The aim of releasing these materials is to allow code inspection and make setting up similar experiments with other data more accessible. Using (elements of) the code for your own data does require some familiarity with Python (v. 3). The list of python packages required to run the code can be found in the requirements.txt file.

#### Questions? Feel free to email!
