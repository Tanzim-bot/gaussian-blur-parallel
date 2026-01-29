# Gaussian Blur – Parallel vs Sequential Image Processing

This project implements a **Gaussian blur** image filter using both:

- A **sequential** (single-core) approach
- A **parallel** version using Python's `multiprocessing` module

The goal is to compare performance, measure speedup, and analyse how the application scales with additional CPU processes.

This project is part of the **CS3S666 – Parallel and Concurrent Programming** module.

---

# 📁 Project Structure

Project folder should look like this:

```
gaussian_blur_parallel/
│
├── input/
│   └── input_large.jpg            # Your test image
│
├── output/
│   ├── images/                    # Output blurred images
│   ├── results/                   # results.csv from performance tests
│   └── plots/                     # Graphs generated for the report
│
├── blur_sequential.py             # Sequential Gaussian blur
├── blur_parallel.py               # Parallel Gaussian blur
├── performance_test.py            # Benchmarks sequential vs parallel
├── plot_results.py                # Generates graphs for the report
├── utils.py                       # Gaussian kernel generation
│
├── README.md                      # This file
└── venv/                          # Virtual environment (optional)
```

All output files are neatly organised into subfolders inside `output/`.

---

### 📥 Cloning the Repository

To download this project to your own machine, run the following command in your terminal:

git clone https://github.com/Tanzim-bot/gaussian-blur-parallel.git

Then navigate into the project folder:

cd gaussian-blur-parallel

# 🧪 Virtual Environment Setup (Recommended)

### 1. Create a virtual environment

**Windows**
```
python -m venv venv
```

**Mac/Linux**
```
python3 -m venv venv
```

### 2. Activate the environment

**Windows (PowerShell)**
```
venv\Scripts\activate
```

**Mac/Linux**
```
source venv/bin/activate
```

### 3. Install dependencies

```
pip install numpy pillow matplotlib pandas
```

---

# ▶️ How to Run the Application

## **1. Run the sequential blur**

This applies a Gaussian blur using **one process**:

```
python blur_sequential.py
```

Output:

- Blurred image saved to:
  ```
  output/images/output_sequential.jpg
  ```
- Execution time printed in terminal

---

## **2. Run the parallel blur**

Uses multiple CPU cores with `multiprocessing`:

```
python blur_parallel.py
```

Output:

- Blurred image saved to:
  ```
  output/images/output_parallel_<num_cores>procs.jpg
  ```
- Parallel execution time printed in terminal

---

## **3. Run performance tests**

This script automatically:

- Resizes your input image to sizes: **512, 1024, 2048**
- Runs sequential and parallel blurs
- Tests multiple process counts: **1, 2, 4, 8**
- Calculates speedup and efficiency
- Saves results to CSV

Run:

```
python performance_test.py
```

Output file:

```
output/results/results.csv
```

---

## **4. Generate graphs for your report**

This script takes `results.csv` and produces:

- Execution time vs processes
- Speedup vs processes
- Efficiency vs processes

Run:

```
python plot_results.py
```

Graphs saved to:

```
output/plots/
    execution_time_512.png
    speedup_512.png
    efficiency_512.png
    (and so on for 1024 and 2048)
```

Insert these graphs into your report for the **Performance Analysis** section.

---

# 🧠 How the Algorithm Works

### **Sequential Version**
- Converts image to grayscale  
- Pads edges for correct convolution  
- Applies Gaussian kernel using nested loops  
- Slow, but accurate baseline to compare against

### **Parallel Version**
- Image is split into horizontal chunks  
- Each chunk processed in a separate process  
- Uses `multiprocessing.Pool.map()`  
- Results combined using `numpy.vstack()`  

This demonstrates how CPU-bound tasks benefit from multiprocessing.

---

# 📘 Key Files Explained

### `utils.py`
Generates the Gaussian kernel used by both sequential & parallel implementations.

### `blur_sequential.py`
Straightforward nested-loop convolution.

### `blur_parallel.py`
Parallel version:
- Uses CPU cores
- Divides image rows evenly between workers
- Each worker applies Gaussian blur to its chunk
- All chunks stitched together into final image

### `performance_test.py`
Runs benchmarks:
- Tests 3 image sizes
- Tests multiple process counts
- Computes speedup & efficiency
- Outputs CSV for graph generation

### `plot_results.py`
Creates all graphs needed for your coursework report.

---

# 📝 Notes for the Coursework Report

The project naturally supports:

- Performance graphs (execution time, speedup, efficiency)
- Discussion of parallel overhead
- Amdahl’s Law and scalability
- Real-world justification for parallel image processing
- Reflection on development challenges (e.g., multiprocessing on Windows, chunking strategy)

---

# ✔ Requirements

- Python 3.8+
- numpy  
- pillow  
- pandas  
- matplotlib  

These are all widely available and cross-platform.

---


