# multicamera_keypoints
A pipeline to go from 6-cam videos to usable keypoints.

## Install
* `git clone https://github.com/dattalab-6-cam/multicamera_keypoints.git`
* `cd multicamera_keypoints`
* [do optional ssh fixes, see below]
* `conda env create --file ./environment.yml --name multicamera_keypoints`
* `pip install -e .`

NB: you may have to fight with the github lines over https/ssh. If using ssh, then make sure you've configurd your github ssh to work from O2, and add the lines
```
Host github.com
   StrictHostKeyChecking no
```
to your ~/.ssh/config file.
Then run `chmod 600 ~/.ssh/config`.


## Overview

Each step requires a config that has, at minimum, these components:

* `slurm_params`: describes the SLURM sbatch job that will be used to run the step.
    * mem (str): amount of memory, as a string ("4GB")
    * gpu (bool): whether to use a GPU, as a boolean (True or False)
    * sec_per_frame (float): how long each frame should take, in seconds (often fractional, eg 0.021)
    * ncpus (int): number of CPUs to use (e.g. 2)
    * jobs_in_progress (dict): a dictionary that will be used to keep track of the jobs that are currently running.

* `wrap_params`: describes the function that will be run.
    * func_path (str): the (absolute) path to the function that will be run.
    * conda_env (str): the conda environment that will be activated in the job.

* `func_args`: the arguments that will be passed to the function.
    [For example, for the `segmentation` step, the arguments are `video_path` and `weights_path`.]
    [These arguments **must** be in the right order here.]

* `func_kwargs`: the keyword arguments that will be passed to the function.
    [For example, for the `segmentation` step, the keyword arguments are `output_name`.]
    [These arguments can be in any order.]

* `output_info`: describes the output of the function that is run.
    * output_name (str): the name of the output file that will be created.

* `step_dependencies`: a list of the steps that must be run before this step can be run. Should be empty if there are no dependencies.

* `pipeline_info`: metadata that describes this step in the pipeline.
    * processing_level (str): whether this step operates at the video level ("video"),
      the session level ("session"), or is a calibration step ("calibration").
