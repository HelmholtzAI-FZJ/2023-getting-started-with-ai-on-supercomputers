---
author: Alexandre Strube // Sabrina Benassou
title: Getting Started with AI on Supercomputers 
# subtitle: A primer in supercomputers`
date: February 29, 2023
---
## Communication:

- [Zoom](https://fz-juelich-de.zoom.us/j/98120874933?pwd=NXJJNXo1Nkx4OGNVNEhkWXBNTWlZUT09)
- [Slack](https://introscfeb2023.slack.com)

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg)


---

## Goals for this course:

- Make sure you know how to access and use our machines
- Distribute your ML workload.

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg)

---

## Team:

::: {.container}
:::: {.col}
![Alexandre Strube](pics/alex.jpg)
::::
:::: {.col}
![Sabrina Benassou](pics/sabrina.jpg)
::::
:::

![](images/Logo_FZ_Juelich_rgb_Schutzzone_transparent.svg)

---

### Schedule for day 1

| Time          | Title        |
| ------------- | -----------  |
| 09:00 - 09:15 | Welcome      |
| 09:15 - 10:00 | Introduction |
| 10:00 - 11:00 | Judoor, Keys |
| 11:00 - 12:00 | SSH, Jupyter, VS Code |
| 12:00 - 13:00 | Sync (everyone should be at the same point) |

---

### Jülich Supercomputers

![JSC Supercomputer Stragegy](images/machines.png)

---

### What is a supercomputer?

- Compute cluster: Many computers bound together locally
- Supercomputer: A damn lot of computers bound together locally

---

### Anatomy of a supercomputer

-  Login Nodes: Normal machines, for compilation, data transfer,  scripting, etc. No GPUs.
- Compute Nodes: Guess what :-)
- Network file system
- Scratch file system accessible from compute nodes
- Key stats:
    - Number of Nodes
    - CPUs, Number of Cores, Single-core Performance
    - RAM
    - Network: Bandwidth, Latency
    - Accelerators (e.g. GPUs)

---

### JUWELS Booster

- 936 Nodes
- AMD EPYC Rome 7402 CPU 2.7 GHz (2 × 24 cores x 2 SMT threads = 96 virtual cores/node)
- 512 GiB memory
- Network Mellanox HDR infiniband (FAST💨 and EXPENSIVE💸)
- 4x NVIDIA A100 with 40gb.

TL;DR: 89856 cores, 3744 GPUs, 468 TB RAM 💪

Way deeper technical info at [Juwels Booster Overview](https://apps.fz-juelich.de/jsc/hps/juwels/booster-overview.html)

---

### JUSUF

- 205 Nodes
- AMD EPYC Rome 7742 CPU 2.25 GHz (2 × 64 cores / node)
- 256 GiB memory
- 61 nodes with 4x NVIDIA V100 with 16gb.

TL;DR: Smaller than JUWELS Booster, but still packs a punch 🤜

Way deeper technical info at [JUSUF Overview](https://apps.fz-juelich.de/jsc/hps/jusuf/cluster/configuration.html)

---

## How do I use a Supercomputer?

- Batch: For heavy compute, ML training
- Interactively: Jupyter

---

### You don't use the whole supercomputer

#### You submit jobs to a queue asking for resources

![](images/supercomputer-queue.svg)

---

### You don't use the whole supercomputer

#### And get results back

![](images/supercomputer-queue-2.svg)

---

### You don't use the whole supercomputer

#### You are just submitting jobs via the login node

![](images/supercomputer-queue-3.svg)

---

### You don't use the whole supercomputer

#### You are just submitting jobs via the login node

![](images/supercomputer-queue-4.svg)

---

### You don't use the whole supercomputer

#### You are just submitting jobs via the login node

![](images/supercomputer-queue-5.svg)

---

### You don't use the whole supercomputer



::: {.container}
:::: {.col}
- Your job(s) enter the queue, and wait for its turn
- When there are enough resources for that job, it runs
::::
:::: {.col}
![](images/midjourney-queue.png)
::::
:::

![]()

---

### You don't use the whole supercomputer

#### And get results back

![](images/queue-finished.svg)

---

### Supercomputer Usage Model
- Using the the supercomputer means submitting a job to a batch system.
- No node-sharing. The smallest allocation for jobs is one compute node (4 GPUs).
- Maximum runtime of a job: 24h.

---

### Recap:

- Login nodes are for submitting jobs, move files, compile, etc
- NOT FOR TRAINING NEURAL NETS

---

### Recap:

- User submit jobs
- Job enters the queue
- When it can, it runs
- Sends results back to user

---

## VSCode

- [Download VScode: code.visualstudio.com](https://code.visualstudio.com/download)
- Install it
- Install [Remote Development Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
- On the local terminal, type `code`

---

## VSCode

### Now with the remote explorer tab
![](images/vscode-welcome.png)

---
### Connecting to JUWELS BOOSTER

#### Getting compute time
- Go to [https://judoor.fz-juelich.de/projects/join/training2303](https://judoor.fz-juelich.de/projects/join/training2303)
- Join the course project `training2303`
- Compute time allocation is based on compute projects. For every compute job, a compute project pays.

---

### Connecting to JUWELS BOOSTER and JUSUF

#### SSH
- SSH is a secure shell (terminal) connection to another computer
- You connect from your computer to the LOGIN NODE

---

### SSH

- Security is given by public/private keys
- You connect from your computer to the LOGIN NODE

--- 

### SSH

#### Create key

```bash
ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519-JSC
```

```bash
$ ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519-JSC
Generating public/private ed25519 key pair.
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /Users/strube1/.ssh/id_ed25519-JSC
Your public key has been saved in /Users/strube1/.ssh/id_ed25519-JSC.pub
The key fingerprint is:
SHA256:EGNNC1NTaN8fHwpfuZRPa50qXHmGcQjxp0JuU0ZA86U strube1@Strube-16
The keys randomart image is:
+--[ED25519 256]--+
|      *++oo=o. . |
|     . =+o .= o  |
|      .... o.E..o|
|       .  +.+o+B.|
|        S  =o.o+B|
|          . o*.B+|
|          . . =  |
|           o .   |
|            .    |
+----[SHA256]-----+
```


---

### SSH

#### Configure SSH session

```bash
code $HOME/.ssh/config
```

---

### SSH

#### Configure SSH session

```bash
Host jusuf
        HostName jusuf.fz-juelich.de
        User [MY_USERNAME]
        IdentityFile ~/.ssh/id_ed25519-JSC

Host booster
        HostName juwels-booster.fz-juelich.de
        User [MY_USERNAME]
        IdentityFile ~/.ssh/id_ed25519-JSC


```
---

### SSH

####  Jülich Supercomputing Centre restricts where you can login from
#### So we need to:
1. Find our ip range
2. Add the range and key to [Judoor](https://judoor.fz-juelich.de)

---

### SSH

#### Find your ip/name range

- Terminal: `curl ifconfig.me`

```bash
$ curl ifconfig.me 
93.199.55.160%
```

(Ignore the `%` sign)

- Let's keep this inside vscode: `code key.txt`

---

### SSH

#### Find your ip/name range

- Browser: [https://www.whatismyip.com](https://www.whatismyip.com)

---

### SSH

#### Find your ip/name range

![](images/whatismyip.png)

---

### SSH - Example: `93.199.55.160`

- Let's make it simpler: `93.199.0.0/16`

(because the last numbers change)

---

### SSH - Example: `93.199.0.0/16`

#### Copy your ssh key
- Terminal: 
```bash
$ cat ~/.ssh/id_ed25519-JSC.pub
ssh-ed25519 AAAAC3NzaC1lZDE1NTA4AAAAIHaoOJF3gqXd7CV6wncoob0DL2OJNfvjgnHLKEniHV6F strube@demonstration.fz-juelich.de
```
- Copy this line to the clipboard (or to an empty file on VS Code: `code key.txt`)

---

### SSH

#### Example: `93.199.0.0/16`

- Put them together and copy again:

```bash
from="93.199.0.0/16" ssh-ed25519 AAAAC3NzaC1lZDE1NTA4AAAAIHaoOJF3gqXd7CV6wncoob0DL2OJNfvjgnHLKEniHV6F strube@demonstration.fz-juelich.de
```

---

### SSH

![](images/manage-ssh-keys.png)

Do it for JUWELS, JUSUF and JUDAC with the same key

---

### SSH

#### Add new key to [Judoor](https://judoor.fz-juelich.de)

![](images/manage-ssh-keys-from-and-key.png)

This might take some minutes

---

### SSH: Exercise

That's it! Give it a try (and answer yes)

```bash
$ ssh jusuf
The authenticity of host 'jusuf.fz-juelich.de (134.94.0.185)' cannot be established.
ED25519 key fingerprint is SHA256:ASeu9MJbkFx3kL1FWrysz6+paaznGenChgEkUW8nRQU.
This key is not known by any other names
Are you sure you want to continue connecting (yes/no/[fingerprint])? Yes
**************************************************************************
*                            Welcome to JUSUF                            *
**************************************************************************
...
...
strube1@jusuf ~ $ 
```

---

### SSH: Exercise 
#### Make sure you are connected to the supercomputer

```bash
# Create a shortcut for the project on the home folder
ln -s $PROJECT_training2303 ./course

# Enter course folder and create a folder for myself
cd course
mkdir $USER
cd $USER

# Where am I?
pwd
```

---

## Working with the supercomputer's software

We have literally thousands of software packages, hand-compiled for the specifics of the supercomputer.

---

## Software

#### Tool for finding software: `module spider`

```bash
strube1$ module spider PyTorch
------------------------------------------------------------------------------------
  PyTorch:
------------------------------------------------------------------------------------
    Description:
      Tensors and Dynamic neural networks in Python with strong GPU acceleration. 
      PyTorch is a deep learning framework that puts Python first.

     Versions:
        PyTorch/1.7.0-Python-3.8.5
        PyTorch/1.8.1-Python-3.8.5
        PyTorch/1.11-CUDA-11.5
        PyTorch/1.12.0-CUDA-11.7
     Other possible modules matches:
        PyTorch-Geometric  PyTorch-Lightning
...
```

---

## Module hierarchy

- Compiler
- MPI
- Module

Eg: `module load GCC OpenMPI PyTorch`

---

## What do we have?

`module avail` (Inside hierarchy)

#### What do I need to load such software?

`module spider Software/version`

---

## Example: PyTorch

Search for the software itself - it will suggest a version

![](images/module-spider-1.png)

---

## Example: PyTorch

Search with the version - it will suggest the hierarchy

![](images/module-spider-2.png)

---

## Example: PyTorch

```bash
$ python
-bash: python: command not found
```

Oh noes! 🙈

Let's bring Python together with PyTorch!

---

## Example: PyTorch

Copy and paste these lines
```bash
# This command fails, as we have no proper python
python 
# So, we load the correct modules...
module load Stages/2023
module load GCC OpenMPI Python PyTorch
# And we run a small test: import pytorch and ask its version
python -c "import torch ; print(torch.__version__)" 
```

Should look like this:
```bash
$ python
-bash: python: command not found
$ module load Stages/2023
$ module load GCC OpenMPI Python PyTorch
$ python -c "import torch ; print(torch.__version__)" 
1.12.0
```
---

## Python Modules

#### Some of the python softwares are part of Python itself, or of other softwares. Use `module key`

```bash
module key toml
The following modules match your search criteria: "toml"
------------------------------------------------------------------------------------

  Jupyter: Jupyter/2020.2.5-Python-3.8.5, Jupyter/2021.3.1-Python-3.8.5, Jupyter/2021.3.2-Python-3.8.5, Jupyter/2022.3.3, Jupyter/2022.3.4
    Project Jupyter exists to develop open-source software, open-standards, and services for interactive computing across dozens of programming languages.
    

  PyQuil: PyQuil/3.0.1
    PyQuil is a library for generating and executing Quil programs on the Rigetti Forest platform.

  Python: Python/3.8.5, Python/3.9.6, Python/3.10.4
    Python is a programming language that lets you work more quickly and integrate your systems more effectively.

------------------------------------------------------------------------------------
```
---

## VSCode
#### Editing files on the supercomputers

![](images/vscode-remotes.png)

---

## VSCode

![](images/vscode-jusuf.png)

---


## VSCode

- From the ssh connection, navigate to your "course" folder and to the name you created earlier.
- This is out working directory. We do everything here.

---

Now we can edit some code and submit it to the queue....
---

But we need to learn about the queue! 🤯
---

## SLURM

---

## Jupyter

[jupyter-jsc.fz-juelich.de](https://jupyter-jsc.fz-juelich.de)

- [TODO] Explain partitions, training, reservation

---

## Kernel and modules:

[Venv/Kernel template](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template)

```bash
git clone https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template.git
cd sc_venv_template
```

---

## Backup slides

---

``` {.java .number-lines}
// Some js

    a = 1;
    b = 2;
    let c = x => 1 + 2 + x;
    c(3);

```

```python
n = 0
while n < 10:
  if n % 2 == 0:
    print(f"{n} is even")
  else:
    print(f"{n} is odd")
  n += 1
```

---