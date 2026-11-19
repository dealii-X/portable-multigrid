import csv
import re

def parse_solver_output(input_file, n_cores):
    # Regex to find n_dofs and the data rows
    dof_pattern = re.compile(r"^0:\s+n_dofs\s+=\s+(\d+)")
    data_pattern = re.compile(r"^0:\s+([\w_]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+%?)\s+(\d+)")
    
    all_data = []
    solve_data = []
    matvec_data = []
    
    current_dof = None
    seen_dofs_this_cycle = set() # To handle the repeated n_dofs lines in your log

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 1. Extract n_dofs
            dof_match = dof_pattern.match(line)
            if dof_match:
                current_dof = dof_match.group(1)
                continue
            
            # 2. Extract Table Data
            data_match = data_pattern.match(line)
            if data_match and current_dof:
                name = data_match.group(1)
                # row structure: dofs, name, min, max, mean, std_dev, pct, samples
                row = [current_dof] + list(data_match.groups())
                
                # Add to Full Dataset
                all_data.append(row)
                
                # Subset 1: solve0, solve1, solve2 (Note: your log showed solve2, I included solve2/3)
                if name in ['solve0', 'solve1', 'solve2', 'solve3']:
                    solve_data.append(row)
                
                # Subset 2: matvec_double
                if name == 'matvec_double':
                    matvec_data.append(row)

    # Helper function to save files
    headers = ["n_dofs", "name", "min", "max", "mean", "std_dev", "mean_solver_pct", "samples"]
    
    def save_csv(filename, data):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

    # Generate the 3 files
    save_csv(f'{n_cores}-core-full_statistics.csv', all_data)
    save_csv(f'{n_cores}-core-solves_only.csv', solve_data)
    save_csv(f'{n_cores}-core-matvec_only.csv', matvec_data)

    print(f"Extraction complete!")
    print(f"- '{n_cores}-core-full_statistics.csv' ({len(all_data)} rows)")
    print(f"- '{n_cores}-core-solves_only.csv' ({len(solve_data)} rows)")
    print(f"- '{n_cores}-core-matvec_only.csv' ({len(matvec_data)} rows)")

# Run the script
parse_solver_output('output-cores-1 copy.out', 1)