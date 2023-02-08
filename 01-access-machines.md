---
author: Alexandre Strube // Sabrina Benassou
title: How do I get access to the machines? 
subtitle: A primer in supercomputers`
date: February 29, 2023
---
## Communication:

- Zoom: [TODO]
- Slack: [TODO]

---

## Goals for this course:

- Make sure you know how to access and use our machines
- Distribute your ML workload.

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

---

### Schedule for day 1

| Time          | Title        |
| ------------- | -----------  |
| 09:00         | Introduction |
| [TODO]        | [TODO]       |
| 12:00 - 13:00 | [TODO]       |

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

### Supercomputer Usage Model
- Using the the supercomputer means submitting a job to a batch system.
- No node-sharing. The smallest allocation for jobs is one compute node (4 GPUs).
- Maximum runtime of a job: 24h. Please implement checkpointing; or make your code faster 😉.
- Solution for long-running tasks: Job arrays.

---

### Connecting to JUWELS BOOSTER

#### Getting compute time
- Go to [http://judoor.fz-juelich.de](http://judoor.fz-juelich.de)
- Join the course project
- Compute time allocation is based on compute projects. For every compute job, a compute project pays.

---

### Connecting to JUWELS BOOSTER

#### SSH
- [TODO]

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