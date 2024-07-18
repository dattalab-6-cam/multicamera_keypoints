# multicamera_keypoints
A pipeline to go from 6-cam videos to usable keypoints.

## Install
* `git clone https://github.com/dattalab-6-cam/multicamera_keypoints.git`
* `cd multicamera_keypoints`
* [do optional ssh fixes, see below]
* `conda env create --file ./environment.yml --name multicamera_keypoints`
* `pip install -e .`

NB: you may have to fight with the github lines of the `environment.yml` file over https/ssh. If using ssh, then make sure you've [configured your github ssh](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to work from O2, and add the lines
```
Host github.com
   StrictHostKeyChecking no
```
to your ~/.ssh/config file.
Then run `chmod 600 ~/.ssh/config`.

> [!IMPORTANT]
> On O2, there are now lots of new L40S gpu's that don't work with cuda 11.7, which is the version we're currently using (the second-latest version on O2). We will try to upgrade to CUDA 12 soon, but this will work for now. As such, you will need to exclude any L40S from the GPUs that you use to run these jobs.
> The following nodes are currently incompatible for this reason, or because they have too little vram: `compute-g-16-[175-177,194,197,254,255],compute-g-17-[166-171]`.
> So you can use the following argument to sbatch or srun on O2 to exclude them: `sbatch --qos=gpuquad_qos -p gpu_quad,gpu -x "compute-g-16-[175-177,194,197,254,255],compute-g-17-[166-171]"`
> You can pipe that into the batch scripts with the following lines in the notebooks:
> ```python
> sbatch_alias = 'sbatch --qos=gpuquad_qos -p gpu_quad,gpu -x "compute-g-16-[175-177,194,197,254,255],compute-g-17-[166-171]"'
> mck.batch.prepare_batch(project_path, processing_step="GIMBAL", sbatch_alias=sbatch_alias)
> ```

> ### Check that the installation will work with your GPUs
> You can check that the installation will work with your GPUs by running the following commands:
* Grab a GPU job: `srun --qos=gpuquad_qos -p gpu_quad,gpu -x "compute-g-16-[175-177,194,197,254,255],compute-g-17-[166-171] -c 1 --mem=4G -t 10:00 --gres=gpu:1 --pty bash`"
* Activate the conda environment: `conda activate multicamera_keypoints`
* Run the following command: `python -c "import torch; print(torch.cuda.is_available())"`
    * Output should be `True`
* Run the following command: `python -c "import jax; print(jax.default_backend()); print(len(jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])))"`
    * Output should be `gpu` and `1`

> [!NOTE]
> You can check JAX compatibility with various CUDA versions here: https://jax.readthedocs.io/en/latest/changelog.html. See the environment file here for details, but you probably want to be specifying specific versions, as the latest version doesn't always work with what we have on O2.



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
