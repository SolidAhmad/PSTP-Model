This repository contains the implementation of the Power System Transition Planning (PSTP) optimization model.  
The model is written in Python using Pyomo and solved with the Gurobi optimizer.

---

## 1. Prerequisites
- **Gurobi Optimizer**  
  Download from: https://www.gurobi.com/downloads/  
### Required Python Packages
- **Pyomo**
- **gurobipy** 
- **pickle** 
- **re** 
### Common Libraries
- **numpy**
- **pandas**
- **math**
---

## 2. Running the Optimization

To execute the PSTP optimization, simply run:

```bash
python PSTP.py
````

Running this script will:

* Build the complete multi-stage PSTP model
* Call the Gurobi solver using settings specified inside `PSTP.py`
* Generate and display gurobi results and save the model outputs

All optimization results, including the full set of decision variables, are stored in the **`Results/`** folder as CSV files for easy inspection and analysis.

---

## 3. Modifying Gurobi Solver Settings

Solver settings can be changed directly inside the `PSTP.py` script.

Look for a code block similar to:

```python
solver = SolverFactory("gurobi")
solver.options["Threads"] = ...
solver.options["MIPGap"] = ...
solver.options["TimeLimit"] = ...
```

---

## 4. Inspecting or Exporting the Full Model

Pyomo provides several ways to inspect the full model structure.

### Print the full model to the terminal

Add the following line after the model is built:

```python
model.pprint()
```

### Save the full model to a text file

```python
model.pprint(filename="full_model.txt")
```

### Export the model to LP or NL formats

These formats are particularly useful for debugging, viewing, or sending to another solver:

```python
model.write("model.lp", io_options={"symbolic_solver_labels": True})
```

You may also export in nonlinear form:

```python
model.write("model.nl")
```

### Export the model data to JSON or YAML

These formats allow external inspection or re-use:

```python
model.write("model.json")
model.write("model.yaml")
```



