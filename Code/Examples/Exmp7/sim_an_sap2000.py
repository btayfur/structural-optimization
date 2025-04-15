import os
import sys
import random
import math
import numpy as np
import comtypes.client
import win32com.client
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define W-sections for A36 steel
W_SECTIONS = [
    "W12X14", "W12X16", "W12X19", "W12X22", "W12X26", "W12X30", "W12X35", "W12X40", 
    "W12X45", "W12X50", "W12X53", "W12X58", "W12X65", "W12X72", "W12X79", "W12X87", 
    "W12X96", "W12X106", "W12X120", "W12X136", "W12X152", "W12X170", "W12X190", "W12X210", 
    "W12X230", "W12X252", "W12X279", "W12X305", "W12X336", "W14X22", "W14X26", "W14X30", 
    "W14X34", "W14X38", "W14X43", "W14X48", "W14X53", "W14X61", "W14X68", "W14X74", "W14X82", 
    "W14X90", "W14X99", "W14X109", "W14X120", "W14X132", "W14X145", "W14X159", "W14X176", 
    "W14X193", "W14X211", "W14X233", "W14X257", "W14X283", "W14X311", "W14X342", "W14X370", 
    "W14X398", "W14X426", "W14X455", "W14X500", "W14X550", "W14X605", "W14X665", "W14X730", 
    "W16X26", "W16X31", "W16X36", "W16X40", "W16X45", "W16X50", "W16X57", "W16X67", "W16X77", 
    "W16X89", "W16X100", "W18X35", "W18X40", "W18X46", "W18X50", "W18X55", "W18X60", "W18X65",
    "W18X71", "W18X76", "W18X86", "W18X97", "W18X106", "W18X119", "W18X130", "W18X143", "W18X158",
    "W18X175", "W21X44", "W21X48", "W21X50", "W21X55", "W21X57", "W21X62", "W21X68", "W21X73", 
    "W21X83", "W21X93", "W21X101", "W21X111", "W21X122", "W21X132", "W21X147", "W21X166", 
    "W21X182", "W21X201", "W24X55", "W24X62", "W24X68", "W24X76", "W24X84", "W24X94", 
    "W24X103", "W24X104", "W24X117", "W24X131", "W24X146", "W24X162", "W24X176", "W24X192",
    "W24X207", "W24X229", "W24X250", "W24X279", "W24X306", "W24X335", "W24X370", "W27X84",
    "W27X94", "W27X102", "W27X114", "W27X129", "W27X146", "W27X161", "W27X178", "W27X194",
    "W27X217", "W27X235", "W27X258", "W27X281", "W27X307", "W27X336", "W27X368", "W27X539", 
    "W30X90", "W30X99", "W30X108", "W30X116", "W30X124", "W30X132", "W30X148", "W30X173", 
    "W30X191", "W30X211", "W30X235", "W30X261", "W30X292", "W30X326", "W30X357", "W30X391",
    "W33X118", "W33X130", "W33X141", "W33X152", "W33X169", "W33X201", "W33X221", 
    "W33X241", "W33X263", "W33X291", "W33X318", "W33X354", "W33X387"]

# SAP2000 model parameters
MODEL_FILE = "model_v24_2.sdb"  # Path to your SAP2000 model file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Simulated Annealing parameters
INITIAL_TEMPERATURE = 100
COOLING_RATE = 0.98
MIN_TEMPERATURE = 0.1
MAX_ITERATIONS = 80
NUM_ELEMENTS = 8  # Number of frame elements to optimize

def start_sap():
    try:
        # Try to get an existing instance of SAP2000
        try:
            sap_object = win32com.client.GetActiveObject("CSI.SAP2000.API.SapObject")
            print("Connected to existing SAP2000 instance")
        except:
            # If no existing instance, create a new one
            print("Creating new SAP2000 instance")
            sap_object = win32com.client.Dispatch("CSI.SAP2000.API.SapObject")
        
        # Start SAP2000
        ret = sap_object.ApplicationStart()
        if ret != 0:
            raise Exception("Failed to start SAP2000")
        
        # Get SAP2000 model
        sap_model = sap_object.SapModel
        
        # Initialize model
        ret = sap_model.InitializeNewModel()
        if ret != 0:
            raise Exception("Failed to initialize new model")
        
        return sap_object, sap_model
        
    except Exception as e:
        print(f"Error starting SAP2000: {str(e)}")
        raise

def analyze_model(sap_model, solution):
    # Open the existing model file
    print(f"Opening model file: {MODEL_FILE}")
    
    # Get the full path to the model file
    
    model_path = os.path.join(current_directory, MODEL_FILE)
    
    # Open the model
    ret = sap_model.File.OpenFile(model_path)
    if ret != 0:
        raise Exception(f"Failed to open model file: {MODEL_FILE}")
    
    # Save a temporary copy
    newpath = os.path.join(current_directory, "model_temp3.sdb")
    ret = sap_model.File.Save(newpath)
    
    print("Model file opened and saved successfully")

    # Add material 
    #sap_model.PropMaterial.AddMaterial("A36", 1, "United States", "ASTM A36", "Grade 36")

    # Import all W-sections from the database
    #for section in W_SECTIONS:
    #    sap_model.PropFrame.ImportProp(section, "A36", "Sections8.pro", section)
    
    # Apply sections to all frame elements
    for i in range(len(solution)):
        ret = sap_model.FrameObj.SetSection(f"G{i+1}", W_SECTIONS[solution[i]], ItemType=1)
    
    # Run analysis
    ret = sap_model.Analyze.RunAnalysis()

    # Check Design Criteria
    ret = sap_model.DesignSteel.StartDesign()
    
    # Call VerifyPassed with the correct parameters
    ret, NonFeasible, Failed, NotChecked, FailedFrames = sap_model.DesignSteel.VerifyPassed()
    
    joints = ["1", "10", "19", "28", "37"]

    ret = sap_model.Results.Setup.SetCaseSelectedForOutput("weight")

    # Initialize variables to store the output parameters
    NumberResults = 0
    Obj = []
    Elm = []
    LoadCase = []
    StepType = []
    StepNum = []
    F1 = []
    F2 = []
    F3 = []
    M1 = []
    M2 = []
    M3 = []

    weight = 0

    for i in range(len(joints)):
        result = sap_model.Results.JointReact(joints[i], 1, NumberResults, Obj, Elm, LoadCase, 
                                      StepType, StepNum, F1, F2, F3, M1, M2, M3)
        weight += result[9][0]

    print(f"Solution: {[W_SECTIONS[i] for i in solution]}")
    print(f"Total weight: {weight}, Feasible: {NonFeasible == 0}")

    return weight, NonFeasible

def generate_random_solution():
    # Generate a random initial solution 
    return [41 for _ in range(NUM_ELEMENTS)] # to save time started with an conservative solution

def generate_neighbor(solution):
    # Create a neighbor by changing one or two section randomly
    neighbor = solution.copy()
    
    # Number of elements to change (1 or 2)
    num_changes = random.randint(1, 2)
    
    for _ in range(num_changes):
        # Select random element to change
        idx = random.randint(0, NUM_ELEMENTS - 1)
        
        # Select a new section different from the current one
        current_section = neighbor[idx]
        new_section = current_section
        while new_section == current_section:
            new_section = random.randint(0, len(W_SECTIONS) - 1)
        
        neighbor[idx] = new_section
    
    return neighbor

def acceptance_probability(old_cost, new_cost, temperature):
    # If new solution is better, accept it
    if new_cost < old_cost:
        return 1.0
    
    # If new solution is worse, calculate acceptance probability
    # We use a huge penalty for non-feasible solutions by multiplying by 1000
    return math.exp((old_cost - new_cost) / temperature)

def simulated_annealing(sap_model):
    # Start with a random solution
    current_solution = generate_random_solution()
    
    # Evaluate the initial solution
    current_weight, current_non_feasible = analyze_model(sap_model, current_solution)
    
    # Apply a penalty for non-feasible solutions
    current_cost = current_weight
    if current_non_feasible > 0:
        current_cost *= 1000  # Large penalty for non-feasible solutions
    
    best_solution = current_solution.copy()
    best_cost = current_cost
    best_weight = current_weight
    best_feasible = (current_non_feasible == 0)
    best_non_feasible = current_non_feasible
    
    # Initialize temperature
    temperature = INITIAL_TEMPERATURE
    
    # Keep track of iterations and improvements
    iteration = 0
    no_improvement_count = 0
    
    # Track all solutions for reporting
    solutions_history = [(best_solution.copy(), best_weight, best_feasible)]
    
    # For visualization
    weights_history = [current_weight]
    iterations_history = [0]
    feasible_history = [best_feasible]
    
    # Set up the plot
    #plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(iterations_history, weights_history, 'b-', label='Weight')
    feasible_points = ax.scatter([], [], c='g', marker='o', label='Feasible')
    infeasible_points = ax.scatter([], [], c='r', marker='x', label='Infeasible')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Weight')
    ax.set_title('Simulated Annealing Optimization Progress')
    ax.legend()
    ax.grid(True)
    
    print("\nStarting Simulated Annealing optimization...")
    print(f"Initial solution: {[W_SECTIONS[i] for i in current_solution]}")
    print(f"Initial weight: {current_weight}, Feasible: {current_non_feasible == 0}")
    
    # Main simulated annealing loop
    while temperature > MIN_TEMPERATURE and iteration < MAX_ITERATIONS:
        iteration += 1
        
        # Generate a neighboring solution
        neighbor_solution = generate_neighbor(current_solution)
        
        # Evaluate the neighboring solution
        neighbor_weight, neighbor_non_feasible = analyze_model(sap_model, neighbor_solution)
        
        # Apply a penalty for non-feasible solutions
        neighbor_cost = neighbor_weight
        if neighbor_non_feasible > 0:
            neighbor_cost *= 1000  # Large penalty for non-feasible solutions
        
        # Decide whether to accept the new solution
        if random.random() < acceptance_probability(current_cost, neighbor_cost, temperature):
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            current_weight = neighbor_weight
            current_non_feasible = neighbor_non_feasible
            
            print(f"\nIteration {iteration}: Accepted new solution")
            print(f"Temperature: {temperature:.2f}")
            print(f"Weight: {current_weight}, Feasible: {current_non_feasible == 0}")
            
            # Check if this is the best solution so far
            if (current_cost < best_cost) or (current_cost == best_cost and current_non_feasible < best_non_feasible):
                best_solution = current_solution.copy()
                best_cost = current_cost
                best_weight = current_weight
                best_feasible = (current_non_feasible == 0)
                best_non_feasible = current_non_feasible
                no_improvement_count = 0
                
                print(f"New best solution found! Weight: {best_weight}, Feasible: {best_feasible}")
                solutions_history.append((best_solution.copy(), best_weight, best_feasible))
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
            print(f"\nIteration {iteration}: Rejected new solution")
            print(f"Temperature: {temperature:.2f}")
        
        # Update visualization
        weights_history.append(current_weight)
        iterations_history.append(iteration)
        feasible_history.append(current_non_feasible == 0)
        
        # Update the plot
        line.set_data(iterations_history, weights_history)
        
        # Update scatter points for feasible/infeasible solutions
        feasible_indices = [i for i, f in enumerate(feasible_history) if f]
        infeasible_indices = [i for i, f in enumerate(feasible_history) if not f]
        
        feasible_points.set_offsets(np.column_stack(([iterations_history[i] for i in feasible_indices],
                                                   [weights_history[i] for i in feasible_indices])))
        infeasible_points.set_offsets(np.column_stack(([iterations_history[i] for i in infeasible_indices],
                                                     [weights_history[i] for i in infeasible_indices])))
        
        # Adjust the plot limits
        ax.relim()
        ax.autoscale_view()
        
        # Draw the plot
        plt.savefig('optimization_progress.png')
        plt.pause(0.02)  # Small pause to allow the plot to update
        
        # Cool down the temperature
        temperature *= COOLING_RATE
        
        # If no improvement for a while, restart from the best solution
        if no_improvement_count >= 10:
            print("\nRestarting from best solution due to no improvements")
            current_solution = best_solution.copy()
            current_cost = best_cost
            current_weight = best_weight
            current_non_feasible = best_non_feasible
            no_improvement_count = 0
    
    # Return the best solution found
    print("\nSimulated Annealing completed!")
    print(f"Best solution: {[W_SECTIONS[i] for i in best_solution]}")
    print(f"Best weight: {best_weight}, Feasible: {best_feasible}")
    
    # Print the solution history
    print("\nSolution History:")
    for i, (sol, weight, feasible) in enumerate(solutions_history):
        print(f"{i+1}. Weight: {weight}, Feasible: {feasible}")
        print(f"   Sections: {[W_SECTIONS[i] for i in sol]}")
    
    # Save the final plot
    plt.savefig('optimization_progress.png')
    plt.close()
    
    return best_solution, best_weight, best_feasible

def main():
    try:
        # Start SAP2000 and open model
        sap_object, sap_model = start_sap()
        
        # Run the simulated annealing optimization
        best_solution, best_weight, best_feasible = simulated_annealing(sap_model)
        
        # Apply the best solution to the model and save it
        if best_feasible:
            print("\nApplying the best solution to the model...")
            analyze_model(sap_model, best_solution)
            
            # Save the optimized model
            optimized_model_path = os.path.join(current_directory, "model_optimized.sdb")
            ret = sap_model.File.Save(optimized_model_path)
            print(f"Optimized model saved to: {optimized_model_path}")
        else:
            print("\nWarning: Best solution is not feasible. Model not saved.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close SAP2000
        if 'sap_object' in locals():
            try:
                print("\nClosing SAP2000...")
                sap_object.ApplicationExit(False)
                print("SAP2000 closed successfully")
            except:
                print("Error closing SAP2000")

if __name__ == "__main__":
    main() 