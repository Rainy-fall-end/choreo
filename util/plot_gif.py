from plotting import animate_stick,load_data
ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = load_data(pattern="/root/autodl-tmp/TSAT-main/Data/choreo/data/mariel_betternot_and_retrograde.npy")
def plot_gif(save_path,ori_data,ghost_data = None):
    animation = animate_stick(ori_data,
                            ghost = ghost_data,
                            figsize=(10,8), 
                            cmap='inferno', 
                            cloud=False,
                            dot_size=10,
                            speed=1
                            )
    animation.save(save_path, writer='pillow')