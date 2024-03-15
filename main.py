import glob
import os
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from os.path import basename

# define collumn names
col_cart_a = "Cart A"
col_cart_b = "Cart B"
col_pos = "Position (m)"
col_vel = "Velocity (m/s)"
col_acc = "Acceleration (m/s²)"


def create_folder_structure():
    os.makedirs("plots/Elastic", exist_ok=True)
    os.makedirs("plots/Explosive", exist_ok=True)
    os.makedirs("plots/Inelastic", exist_ok=True)

    os.makedirs("results/Elastic", exist_ok=True)
    os.makedirs("results/Explosive", exist_ok=True)
    os.makedirs("results/Inelastic", exist_ok=True)
    os.makedirs("results/tex/Elastic", exist_ok=True)
    os.makedirs("results/tex/Explosive", exist_ok=True)
    os.makedirs("results/tex/Inelastic", exist_ok=True)


def find_acceleration_peaks(data):
    # Find peaks in the smoothed acceleration
    min_threshold = 0.4
    max_threshold = 10
    l_peaks, _ = find_peaks(data[f'smoothed {col_acc}'],
                            height=(min_threshold, max_threshold))  # Adjust threshold as needed
    return l_peaks


def find_acceleration_spikes(peaks, data):
    # Extend the spike on both sides of each peak
    spike_start = []
    spike_end = []

    for peak in peaks:
        start = peak
        end = peak

        # Extend to the left until acceleration starts increasing
        while (start > 0 and data[f'smoothed {col_acc}'].iloc[start] >
               data[f'smoothed {col_acc}'].iloc[start - 1]):
            start -= 1

        # Extend to the right until acceleration starts increasing
        while (end < len(data) - 1 and data[f'smoothed {col_acc}'].iloc[end] >
               data[f'smoothed {col_acc}'].iloc[end + 1]):
            end += 1

        spike_start.append(start)
        spike_end.append(end)

    spike_start = [data.index[i] for i in spike_start]
    spike_end = [data.index[i] for i in spike_end]
    # print(spike_start, spike_end)
    spikes = [i for i in zip(spike_start, spike_end)]
    # print(spikes)
    return spikes


def load_weights():
    return pd.read_csv("data/weight_config.csv").set_index("weight_config")


def load_uncertainties():
    return pd.read_csv("data/uncertainties.csv").set_index("quantity")


def load_files(experiment, selected_weight_config):
    data_path = f"data/{experiment}"

    csv_files = glob.glob(data_path + f"/*{selected_weight_config}*.csv")

    df_list = {basename(file): pd.read_csv(file).set_index("Time (s)") for file in csv_files}
    return df_list


def make_plot():
    fig, ax = plt.subplots()
    return fig, ax


def plot_data(data, fig, ax, label):
    data.plot(y=col_vel, ax=ax, label=label)
    # data.plot(y=col_acc, ax=ax)


def filter_acceleration(data):
    data[f'abs {col_acc}'] = np.abs(data[col_acc])
    # Smooth the acceleration using Savitzky-Golay filter
    window_size = 10  # Adjust the window size based on your data
    data[f'smoothed {col_acc}'] = savgol_filter(data[f'abs {col_acc}'], window_size, 3)


def find_regions(data):
    filter_acceleration(data)
    peaks = find_acceleration_peaks(data)
    print(peaks)
    spikes = find_acceleration_spikes(peaks, data)
    print(spikes)

    spike_regions = []
    for i in range(len(spikes) - 1):
        spike_regions.append(spikes[i])
        spike_regions.append((spikes[i][1], spikes[i + 1][0]))
    spike_regions.append(spikes[-1])

    start_region = (data.index[0], spikes[0][0])
    end_region = (spikes[-1][1], data.index[-1])
    regions = [start_region] + spike_regions + [end_region]

    return regions


def plot_region(data, region, fig, ax, label='Cart'):
    plot_data(data[region[0]:region[1]], fig, ax, label)


def find_intersecting_region(regions_a, center):
    for region in regions_a:
        if region[0] <= center <= region[1]:
            return region


def do_analysis(carts, uncertainties):
    speeds = {}
    regions_cart_a = find_regions(carts[col_cart_a])
    regions_cart_b = find_regions(carts[col_cart_b])

    center_b = ((regions_cart_b[1][1] - regions_cart_b[1][0]) / 2) + regions_cart_b[1][0]
    intersecting_region = find_intersecting_region(regions_cart_a, center_b)

    # print(regions_cart_a)
    # print(regions_cart_b)

    fig, ax = make_plot()
    # plot_data(carts[col_cart_b][0:4.5], fig, ax)

    initial_speed_a = carts[col_cart_a][col_vel][regions_cart_b[1][0]]
    initial_speed_b = carts[col_cart_b][col_vel][regions_cart_b[1][0]]
    final_speed_a = carts[col_cart_a][col_vel][regions_cart_b[1][1]]
    final_speed_b = carts[col_cart_b][col_vel][regions_cart_b[1][1]]

    speeds['initial_a'] = initial_speed_a
    speeds['initial_b'] = initial_speed_b
    speeds['final_a'] = final_speed_a
    speeds['final_b'] = final_speed_b

    speeds['initial_a_err'] = calculate_speed_error(carts[col_cart_a][col_pos][regions_cart_b[1][0]],
                                                    regions_cart_b[1][0], uncertainties)
    speeds['initial_b_err'] = calculate_speed_error(carts[col_cart_b][col_pos][regions_cart_b[1][0]],
                                                    regions_cart_b[1][0], uncertainties)

    speeds['final_a_err'] = calculate_speed_error(carts[col_cart_a][col_pos][regions_cart_b[1][1]],
                                                  regions_cart_b[1][1], uncertainties)
    speeds['final_b_err'] = calculate_speed_error(carts[col_cart_b][col_pos][regions_cart_b[1][1]],
                                                  regions_cart_b[1][1], uncertainties)
    # plot_data(carts[col_cart_a], fig, ax, "dfq")

    plot_region(carts[col_cart_a], intersecting_region, fig, ax, label='Cart A')
    plot_region(carts[col_cart_b], intersecting_region, fig, ax, label='Cart B')
    ax.grid(True, which='both')
    ax.set_ylabel('Velocity (m/s)')
    # ax.legend().remove()
    return speeds


def build_dataframe(data):
    carts = {}

    vehicle_a = pd.DataFrame()
    vehicle_a.index = data.index
    vehicle_a[col_pos] = data['Position 1 (m)']
    vehicle_a[col_vel] = data['Velocity 1 (m/s)']
    vehicle_a[col_acc] = data['Acceleration 1 (m/s²)']

    vehicle_b = pd.DataFrame()
    vehicle_b[col_pos] = -1 * data['Position 2 (m)']
    vehicle_b[col_vel] = -1 * data['Velocity 2 (m/s)']
    vehicle_b[col_acc] = -1 * data['Acceleration 2 (m/s²)']

    carts[col_cart_a] = vehicle_a
    carts[col_cart_b] = vehicle_b

    return carts


def add_velocities_to_table(speeds, weight_config, vel_table):
    new_row = [weight_config['cartA'], speeds['initial_a'], speeds['initial_a_err'],
               speeds['final_a'], speeds['final_a_err'],
               weight_config['cartB'], speeds['initial_b'], speeds['final_a_err'],
               speeds['final_b'], speeds['final_b_err']]
    vel_table.loc[len(vel_table)] = pd.Series(new_row, index=vel_table.columns)


def calculate_speed_error(position, time, uncertainties):
    a = (1/time) * (uncertainties.loc['position'].value * position)
    b = position/(time**2) * uncertainties.loc['time'].value
    return np.sqrt(a ** 2 + b ** 2)


def calculate_kinetic_energy_error(mass, velocity, velocity_acc, uncertainties):
    a = (velocity**2 / 2) * uncertainties.loc['mass'].value
    b = mass * velocity_acc
    return np.sqrt(a ** 2 + b ** 2)


def calculate_momentum_error(mass, velocity, velocity_acc, uncertainties):
    a = velocity * uncertainties.loc['mass'].value
    b = mass * velocity_acc
    return np.sqrt(a ** 2 + b ** 2)


def calculate_momentum(velocity, mass):
    return velocity * mass


def calculate_kinetic_energy(mass, velocity):
    return (mass / 2) * (velocity ** 2)


def add_momenta_to_table(speeds, run, weight_config, momenta_table, uncertainties):
    weight = weight_config[1]
    weight_cart_a = weight['cartA']
    weight_cart_b = weight['cartB']

    initial_momentum_cart_a = calculate_momentum(weight_cart_a, speeds['initial_a'])
    initial_momentum_error_cart_a = calculate_momentum_error(weight_cart_a, speeds['initial_a'],
                                                             speeds['initial_a_err'], uncertainties)
    initial_momentum_cart_b = calculate_momentum(weight_cart_b, speeds['initial_b'])
    initial_momentum_error_cart_b = calculate_momentum_error(weight_cart_b, speeds['initial_b'],
                                                             speeds['initial_b_err'], uncertainties)

    final_momentum_cart_a = calculate_momentum(weight_cart_a, speeds['final_a'])
    final_momentum_error_cart_a = calculate_momentum_error(weight_cart_a, speeds['final_a'],
                                                           speeds['final_a_err'], uncertainties)
    final_momentum_cart_b = calculate_momentum(weight_cart_b, speeds['final_b'])
    final_momentum_error_cart_b = calculate_momentum_error(weight_cart_b, speeds['final_b'],
                                                           speeds['final_b_err'], uncertainties)

    total_momentum_before = initial_momentum_cart_a + initial_momentum_cart_b
    total_momentum_error_before = initial_momentum_error_cart_a + initial_momentum_error_cart_b
    total_momentum_after = final_momentum_cart_a + final_momentum_cart_b
    total_momentum_error_after = final_momentum_error_cart_a + final_momentum_error_cart_b

    initial_kinetic_energy_cart_a = calculate_kinetic_energy(weight_cart_a, speeds['initial_a'])
    initial_kinetic_energy_error_cart_a = (
        calculate_kinetic_energy_error(weight_cart_a, speeds['initial_a'], speeds['initial_a_err'], uncertainties))
    initial_kinetic_energy_cart_b = calculate_kinetic_energy(weight_cart_b, speeds['initial_b'])
    initial_kinetic_energy_error_cart_b = (
        calculate_kinetic_energy_error(weight_cart_b, speeds['initial_b'], speeds['initial_b_err'], uncertainties))

    final_kinetic_energy_cart_a = calculate_kinetic_energy(weight_cart_a, speeds['final_a'])
    final_kinetic_energy_error_cart_a = calculate_kinetic_energy_error(weight_cart_a, speeds['final_a'],
                                                                       speeds['final_a_err'], uncertainties)
    final_kinetic_energy_cart_b = calculate_kinetic_energy(weight_cart_b, speeds['final_b'])
    final_kinetic_energy_error_cart_b = calculate_kinetic_energy_error(weight_cart_b, speeds['final_b'],
                                                                       speeds['final_b_err'], uncertainties)

    total_kinetic_energy_before = initial_kinetic_energy_cart_a + initial_kinetic_energy_cart_b
    total_kinetic_energy_error_before = initial_kinetic_energy_error_cart_a + initial_kinetic_energy_error_cart_b
    total_kinetic_energy_after = final_kinetic_energy_cart_a + final_kinetic_energy_cart_b
    total_kinetic_energy_error_after = final_kinetic_energy_error_cart_a + final_kinetic_energy_error_cart_b

    new_row = [total_momentum_before, total_momentum_error_before,
               total_kinetic_energy_before, total_kinetic_energy_error_before,
               total_momentum_after, total_momentum_error_after,
               total_kinetic_energy_after, total_kinetic_energy_error_after]
    ind = (weight['name'], run)
    momenta_table.loc[ind, :] = new_row


def new_velocity_table():
    vel_header = ['Mass (kg)', 'Initial velocity (m/s)', 'Initial velocity err (m/s)',
                  'Final velocity (m/s)', 'Final velocity err (m/s)']

    vel_table = pd.DataFrame(columns=vel_header * 2)
    vel_table.columns = pd.MultiIndex.from_tuples(
        [('Cart A', i) for i in vel_header] + [('Cart B', i) for i in vel_header])
    vel_table.index.name = "Run"
    return vel_table


def new_before_after_table():
    multi_index = pd.MultiIndex.from_tuples([], names=['weight config', 'run'])
    momenta_header = ['Momentum (kg-m/s)', 'Momentum err (kg-m/s)', 'Kinetic energy (J)', 'Kinetic energy err (J)']
    momenta_table = pd.DataFrame(columns=momenta_header * 2, index=multi_index)
    momenta_table.columns = pd.MultiIndex.from_tuples([('Before', i) for i in momenta_header] +
                                                      [('After', i) for i in momenta_header])
    momenta_table.index.name = "Run"
    return momenta_table


def new_momentum_gain_table():
    multi_index = pd.MultiIndex.from_tuples([], names=['weight config', 'run'])
    gain_header = ['Momentum gain cart A (%)', 'Momentum gain cart B (%)']
    gain_table = pd.DataFrame(columns=gain_header, index=multi_index)
    gain_table.index.name = "Run"
    return gain_table


def add_row_to_gain(weight_config, run, speeds, gain_table):
    weight = weight_config[1]
    weight_cart_a = weight['cartA']
    weight_cart_b = weight['cartB']

    initial_momentum_cart_a = calculate_momentum(weight_cart_a, speeds['initial_a'])
    initial_momentum_cart_b = calculate_momentum(weight_cart_b, speeds['initial_b'])

    final_momentum_cart_a = calculate_momentum(weight_cart_a, speeds['final_a'])
    final_momentum_cart_b = calculate_momentum(weight_cart_b, speeds['final_b'])

    new_row = [abs((final_momentum_cart_a / initial_momentum_cart_a) * 100),
               (abs(final_momentum_cart_b / initial_momentum_cart_b) * 100)]
    ind = (weight['name'], run)
    gain_table.loc[ind, :] = new_row


def run_full_analysis(experiment, weights, vel_table, momenta_table, gain_table, uncertainties):
    runs = load_files(experiment, weights[0])
    for key in runs:
        run = int(key.split("-")[5].split(".")[0])
        print(key)
        print(run)
        print(weights)
        carts = build_dataframe(runs[key])
        speeds = do_analysis(carts, uncertainties)
        plt.title(f"{experiment} Collision ({weights[1]['info']})")
        add_velocities_to_table(speeds, weights[1], vel_table)
        add_momenta_to_table(speeds, run, weights, momenta_table, uncertainties)
        add_row_to_gain(weights, run, speeds, gain_table)
        plt.savefig(f"plots/{experiment}/{key.replace('.csv', '')}.png", dpi=300)
    plt.close()


def scatter_plot_with_line_fit(data, x_name, y_name, x_lbl, y_lbl, title, x_err_name=None, y_err_name=None):
    # multi_index = pd.MultiIndex.from_tuples([], names=['weight config', 'run'])
    # momenta_header = ['Momentum (kg-m/s)', 'Momentum (kg-m/s) std', 'Kinetic energy (J)', 'Kinetic energy (J) std']
    # df = pd.DataFrame(columns=momenta_header * 2, index=multi_index)
    # df.columns = pd.MultiIndex.from_tuples([('Before', i) for i in momenta_header] +
    #                                       [('After', i) for i in momenta_header])

    # for index, row in data.iterrows():
    #    err_val = error_tbl.loc[index[0]]
    #    bef_mom_std = err_val['Before']['Momentum std']
    #    bef_mom = row['Before']['Momentum (kg-m/s)']

    #    bef_kin_std = err_val['Before']['Kinetic energy std']
    #    bef_kin = row['Before']['Kinetic energy (J)']

    #    aft_mom_std = err_val['After']['Momentum std']
    #    aft_mom = row['After']['Momentum (kg-m/s)']

    #    aft_kin_std = err_val['After']['Kinetic energy std']
    #    aft_kin = row['After']['Kinetic energy (J)']
    #    new_row = [bef_mom, bef_mom_std, bef_kin, bef_kin_std, aft_mom, aft_mom_std, aft_kin, aft_kin_std]
    #    df.loc[index, :] = new_row

    df = data

    fig, ax = make_plot()

    x_values = df[x_name].values.tolist()
    y_values = df[y_name].values.tolist()
    x_err = df[x_err_name].values.tolist()
    y_err = df[y_err_name].values.tolist()

    coefficients, covariance_matrix = np.polyfit(x_values, y_values, 1, cov=True)
    slope = coefficients[0]
    intercept = coefficients[1]
    slope_variance = covariance_matrix[0, 0]

    if x_err and y_err:
        plt.errorbar(x_values, y_values, xerr=x_err, yerr=y_err, fmt='none', linewidth=1, capsize=3)
    # Plot data and linear fit
    plt.scatter(x_values, y_values, color='orange', label='Data')
    plt.plot(x_values, np.polyval(coefficients, x_values), label=f"Fit: slope={slope:.2f}, intercept={intercept:.2f}")
    # Plot error bars for uncertainty

    plt.plot([], [], ' ', label=f'Uncertainty: ±{np.sqrt(slope_variance):.2f}')

    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    ax.legend()
    plt.title(title)

    return


def run_statistical_analysis(data):
    print(data)
    b = data.groupby(level='weight config').std()
    b.columns = pd.MultiIndex.from_tuples(b.set_axis(b.columns.values, axis=1)
                                          .rename(columns={('Before', 'Momentum (kg-m/s)'): ('Before', 'Momentum std'),
                                                           ('Before', 'Kinetic energy (J)'): (
                                                               'Before', 'Kinetic energy std'),
                                                           ('After', 'Momentum (kg-m/s)'): ('After', 'Momentum std'),
                                                           ('After', 'Kinetic energy (J)'): (
                                                               'After', 'Kinetic energy std')}))
    return b


def analyze_experiment(experiment, uncertainties):
    weight_table = load_weights()
    vel_table = new_velocity_table()
    gain_table = new_momentum_gain_table()
    momenta_table = new_before_after_table()
    for selected_weight in weight_table.iterrows():
        run_full_analysis(experiment, selected_weight, vel_table, momenta_table, gain_table, uncertainties)

    momenta_table[('', '  ratio')] = (momenta_table[('After', 'Kinetic energy (J)')] /
                                      momenta_table[('Before', 'Kinetic energy (J)')])

    scatter_plot_with_line_fit(momenta_table,
                               ('Before', 'Momentum (kg-m/s)'),
                               ('After', 'Momentum (kg-m/s)'),
                               'Momentum before (kg-m/s)',
                               'Momentum after (kg-m/s)',
                               f"Conservation of momentum ({experiment})",
                               ('Before', 'Momentum err (kg-m/s)'), ('After', 'Momentum err (kg-m/s)'))
    plt.savefig(f"plots/{experiment}/MomentumBeforeAfter.png", dpi=300)
    plt.close()

    scatter_plot_with_line_fit(momenta_table, ('Before', 'Kinetic energy (J)'),
                               ('After', 'Kinetic energy (J)'),
                               'MKinetic energy before (J)',
                               'Kinetic energy after (J)',
                               f"Kinetic energy ({experiment})",
                               ('Before', 'Kinetic energy err (J)'), ('After', 'Kinetic energy err (J)'))
    plt.savefig(f"plots/{experiment}/KineticBeforeAfter.png", dpi=300)
    plt.close()

    print([f"{i[0]} {i[1]}" for i in vel_table.columns])

    vel_table.to_csv(f"results/{experiment}/{experiment}_collisions_velocities.csv",
                     header=[f"{i[0]} {i[1]}" for i in vel_table.columns])
    momenta_table.to_csv(f"results/{experiment}/{experiment}_collisions_momenta.csv",
                         header=[f"{i[0]} {i[1]}" for i in momenta_table.columns])
    gain_table.to_csv(f"results/{experiment}/{experiment}_collisions_gain.csv",
                      header=[f"{i[0]} {i[1]}" for i in gain_table.columns])

    export_to_tex(vel_table, f"results/tex/{experiment}/{experiment}_collisions_velocities.tex", experiment,
                  "Velocities")
    export_to_tex(momenta_table, f"results/tex/{experiment}/{experiment}_collisions_momenta.tex", experiment,
                  "Momenta", [('Before', 'Momentum (kg-m/s)'), ('Before', 'Kinetic energy (J)'),
                              ('After', 'Kinetic energy (J)'), ('After', 'Momentum (kg-m/s)'), ('', '  ratio')])
    export_to_tex(momenta_table, f"results/tex/{experiment}/{experiment}_collisions_momenta_err.tex", experiment,
                  "MomentaErrors", [('Before', 'Momentum err (kg-m/s)'),
                                    ('Before', 'Kinetic energy err (J)'),
                                    ('After', 'Kinetic energy err (J)'), ('After', 'Momentum err (kg-m/s)')])
    export_to_tex(gain_table, f"results/tex/{experiment}/{experiment}_collisions_gains.tex", experiment,
                  "Gain")


def replacenth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n - 1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    print(newString)


def export_to_tex(df, path, experiment, label_extra, export_columns=None):
    if export_columns is None:
        export_columns = df.columns.tolist()

    df = df[export_columns]

    if type(df.columns) == pd.MultiIndex:
        head = [i[0] for i in df.columns]
        sub_head = [i[1].split(" ") for i in df.columns]
        # Find the maximum length of any sub-array
        max_length = max([len(sublist) for sublist in sub_head])
        # Fill missing values with empty strings
        sub_head = [sublist + [''] * (max_length - len(sublist)) for sublist in sub_head]
        l = []
        for a, b in zip(head, sub_head):
            l.append([a] + b)
        h = list(zip(*l))
        df.columns = pd.MultiIndex.from_arrays(h)

    n_rows, n_cols = df.shape
    s = df.style
    s = s.format(precision=3)
    s = s.set_table_styles([
        {'selector': 'toprule', 'props': ':hline;'},
        {'selector': 'midrule', 'props': ':hline\\hline;'},
        {'selector': 'bottomrule', 'props': ':hline;'},
    ], overwrite=False)
    label = f"tbl:{experiment}{label_extra}"
    caption = f"Data for the {experiment.lower()} collision."
    df_tex = s.to_latex(multicol_align='c|', column_format="|" + "r|" * (n_cols + 2),
                        position="H", position_float="centering", label=label, caption=caption,
                        clines="skip-last;data")
    df_tex = df_tex.replace("%", "\\%")
    with open(path, "w") as f:
        f.write(df_tex)
        f.flush()


def main():
    create_folder_structure()
    experiments = ["Elastic", "Explosive", "Inelastic"]
    uncertainties = load_uncertainties()
    for experiment in experiments:
        analyze_experiment(experiment, uncertainties)


if __name__ == "__main__":
    main()
