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

## How do I use a Supercomputer?

- Interactively: E.g. Jupyter
- Batch: For heavy compute, ML training

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
- Solution for long-running tasks: Job arrays.

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

- Jülich Supercomputing Centre restricts where you can login from
- This is done at the "Manage SSH keys" on Judoor

![](images/manage-ssh-keys.png)

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