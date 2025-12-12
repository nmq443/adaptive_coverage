import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# REMINDER: Set these paths before running!
# <<< 1. SET YOUR ROOT DIRECTORY HERE
ROOT_DIR = "results"
# <<< 2. SET YOUR TARGET SAVE DIRECTORY HERE
TARGET_SAVE_DIR = "C:/Users/ADMIN/Desktop/Log_Lambda2_Plots"
FILENAME_TO_FIND = "ld2s_data.npy"
# ---------------------


def process_and_plot_file(file_path, save_directory):
    """
    Loads data, applies the cleaning step, plots it, and saves the figure.
    The filename is constructed from the parent directories.
    """
    try:
        lambda2_values = np.load(file_path)
        timesteps = np.arange(len(lambda2_values))

        # --- FILENAME CONSTRUCTION LOGIC ---
        # The file structure is generally:
        # .../METHOD/ENV/AGENTS/RUN_ID/ld2s.npy

        # 1. Get the directory path (e.g., .../RUN_ID)
        run_dir_path = os.path.dirname(file_path)
        # 2. Split the path into parts
        path_parts = run_dir_path.split(os.sep)

        # Assuming the ld2s.npy is 4 levels deep from the main identifier (hexagon/voronoi)
        # Example: G:\...results\voronoi\env1\20_agents\run0

        # Extract the relevant parts based on position relative to 'ld2s.npy'
        # RUN_ID is the parent (run0)
        RUN_ID = path_parts[-1]
        # AGENTS is the grandparent (20_agents)
        AGENTS = path_parts[-2]
        # ENV is the great-grandparent (env1)
        ENV = path_parts[-3]
        # METHOD is the great-great-grandparent (voronoi or original/pso)
        # This part requires a small adjustment due to the 'original/pso' structure

        # If the structure is .../voronoi/env1/..., METHOD is 'voronoi'
        # If the structure is .../hexagon/env1/20_agents/original/..., METHOD is 'original'
        if path_parts[-4] in ['original', 'pso']:
            # For 'hexagon' tree
            # e.g., hexagon_original
            METHOD = f"{path_parts[-5]}_{path_parts[-4]}"
        else:
            # For 'voronoi' tree (where method is the top level)
            METHOD = path_parts[-4]  # e.g., voronoi

        # Create a descriptive output filename
        output_filename = f"{METHOD}_{ENV}_{AGENTS}_{RUN_ID}_ld2s_log.png"
        save_path = os.path.join(save_directory, output_filename)

        # --- CLEAN STEP: replace values < 1e-6 (copied from your code) ---
        threshold = 1e-6
        values = lambda2_values.copy()
        mask_bad = values < threshold

        if np.any(mask_bad):
            good_idx = np.where(values >= threshold)[0]
            if len(good_idx) > 0:
                nearest_good_idx = good_idx[np.abs(np.subtract.outer(
                    np.arange(len(values)), good_idx)).argmin(axis=1)]
                values[mask_bad] = values[nearest_good_idx[mask_bad]]
            # If all values are below threshold, no correction is possible, data will remain flat.
        # -----------------------------------------------------------------

        # --- PLOTTING ---
        plt.figure(figsize=(12, 7))
        plt.plot(timesteps, values,
                 marker='.',
                 linestyle='-',
                 markersize=0.5,
                 color='cornflowerblue',
                 label='Lambda2 Value')

        plt.ylim(0, 5)  # Set reasonable limits for the log plot

        # plot_title = f'Lambda2 Decay: {output_filename.replace("_ld2s_log.png", "")}'

        plt.xlabel('Timestep')
        plt.ylabel('Lambda2')
        # plt.title(plot_title)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Save and close the figure
        plt.savefig(save_path)
        plt.close()

        print(f"Successfully processed and saved: {output_filename}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    """
    Main function to traverse directories and call the plotting function.
    """
    if not os.path.exists(TARGET_SAVE_DIR):
        os.makedirs(TARGET_SAVE_DIR)
        print(f"Created target directory: {TARGET_SAVE_DIR}")

    file_count = 0
    # os.walk traverses from ROOT_DIR down
    for root, dirs, files in os.walk(ROOT_DIR):
        if FILENAME_TO_FIND in files:
            file_path = os.path.join(root, FILENAME_TO_FIND)
            process_and_plot_file(file_path, TARGET_SAVE_DIR)
            file_count += 1

    print("-" * 30)
    print(f"Directory traversal finished. Total files processed: {file_count}")


if __name__ == "__main__":
    # Ensure you set the ROOT_DIR and TARGET_SAVE_DIR variables before running!
    # Example usage:
    # ROOT_DIR = r"G:\Projects\adaptive_coverage\results"
    # TARGET_SAVE_DIR = r"C:\Users\ADMIN\Desktop\Log_Lambda2_Plots"
    main()
