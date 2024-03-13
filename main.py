import glob

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


def find_sign_changes(data):
    acceleration = data['Acceleration 1 (m/s²)']

    sign = acceleration.map(np.sign)
    diff1 = sign.diff(periods=1).fillna(0)
    diff2 = sign.diff(periods=-1).fillna(0)
    # print(diff1)
    # print(diff2)

    df1 = data.loc[diff1[diff1 != 0].index]
    df2 = data.loc[diff2[diff2 != 0].index]
    # print(df1)
    # print(df2)

    idx = np.where(abs(df1['Acceleration 1 (m/s²)'].values) < abs(df2['Acceleration 1 (m/s²)'].values),
                   df1.index.values, df2.index.values)
    print(idx)
    # print(data.loc[idx])

    l_mod = [data.index[0]] + list(idx) + [data.index[-1]]
    print(l_mod)
    ranges = [(l_mod[n], l_mod[n + 1]) for n in range(len(l_mod) - 1)]
    df_ranges = []
    for range_start, range_end in ranges:
        print(range_start, range_end)
        df_ranges.append(data.loc[range_start:range_end])
    print(df_ranges)


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


def load_files(experiment, selected_weight_config):
    data_path = f"data/{experiment}Data"

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


def do_analysis(carts):
    speeds = {}
    regions_cart_a = find_regions(carts[col_cart_a])
    regions_cart_b = find_regions(carts[col_cart_b])

    center_b = ((regions_cart_b[1][1] - regions_cart_b[1][0]) / 2) + regions_cart_b[1][0]
    intersecting_region = find_intersecting_region(regions_cart_a, center_b)

    # print(regions_cart_a)
    # print(regions_cart_b)

    fig, ax = make_plot()
    # plot_data(carts[col_cart_b][0:4.5], fig, ax)

    speeds['initial_a'] = carts[col_cart_a][col_vel][regions_cart_b[1][0]]
    speeds['initial_b'] = carts[col_cart_b][col_vel][regions_cart_b[1][0]]

    speeds['final_a'] = carts[col_cart_a][col_vel][regions_cart_b[1][1]]
    speeds['final_b'] = carts[col_cart_b][col_vel][regions_cart_b[1][1]]

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
    new_row = [weight_config['cart_a'], speeds['initial_a'], speeds['final_a'],
               weight_config['cart_b'], speeds['initial_b'], speeds['final_b']]
    vel_table.loc[len(vel_table)] = pd.Series(new_row, index=vel_table.columns)


def add_momenta_to_table(speeds, weight_config, momenta_table):
    weight_cart_a = weight_config['cart_a']
    weight_cart_b = weight_config['cart_b']
    total_momentum_before = weight_cart_a * speeds['initial_a'] + weight_cart_b * speeds['initial_b']
    total_kinetic_energy_before = (weight_cart_a / 2) * (speeds['initial_a'] ** 2) + (weight_cart_b / 2) * (
            speeds['initial_b'] ** 2)
    total_kinetic_energy_after = (weight_cart_a / 2) * (speeds['final_a'] ** 2) + (weight_cart_b / 2) * (
            speeds['final_b'] ** 2)
    total_momentum_after = weight_cart_a * speeds['final_a'] + weight_cart_b * speeds['final_b']

    new_row = [total_momentum_before, total_kinetic_energy_before, total_momentum_after, total_kinetic_energy_after]
    momenta_table.loc[len(momenta_table)] = pd.Series(new_row, index=momenta_table.columns)


def new_velocity_table():
    vel_header = ['Mass (kg)', 'Initial velocity (m/s)', 'Final velocity (m/s)']

    vel_table = pd.DataFrame(columns=vel_header * 2)
    vel_table.columns = pd.MultiIndex.from_tuples([('A', i) for i in vel_header] + [('B', i) for i in vel_header])
    vel_table.index.name = "Run"
    return vel_table


def new_before_after_table():
    momenta_header = ['Momentum (kg-m/s)', 'Kinetic energy (J)']
    momenta_table = pd.DataFrame(columns=momenta_header * 2)
    momenta_table.columns = pd.MultiIndex.from_tuples([('Before', i) for i in momenta_header] +
                                                      [('After', i) for i in momenta_header])
    momenta_table.index.name = "Run"

    return momenta_table


def run_full_analysis(experiment, weights, vel_table, momenta_table):
    runs = load_files(experiment, weights[0])
    for key in runs:
        print(key)
        print(weights)
        carts = build_dataframe(runs[key])
        speeds = do_analysis(carts)
        plt.title(f"{experiment} Collision ({weights[1]['title']})")
        add_velocities_to_table(speeds, weights[1], vel_table)
        add_momenta_to_table(speeds, weights[1], momenta_table)
        plt.savefig(f"plots/{experiment}/{key.replace('.csv','')}.png", dpi=300)
    plt.close()


def scatter_plot_with_line_fit(data, x_values, y_values, x_lbl, y_lbl, title):
    fig, ax = make_plot()

    coefficients, covariance_matrix = np.polyfit(x_values, y_values, 1, cov=True)
    slope = coefficients[0]
    intercept = coefficients[1]
    slope_variance = covariance_matrix[0, 0]

    # Plot data and linear fit
    plt.scatter(x_values, y_values, label='Data')
    plt.plot(x_values, np.polyval(coefficients, x_values), label=f"Fit: slope={slope:.2f}, intercept={intercept:.2f}")

    # Plot error bars for uncertainty
    #plt.errorbar(x_values, np.polyval(coefficients, x_values), yerr=np.sqrt(slope_variance), fmt='none',
    #             label=f'Uncertainty: ±{np.sqrt(slope_variance):.2f}')

    plt.plot([], [], ' ', label=f'Uncertainty: ±{np.sqrt(slope_variance):.2f}')

    ax.scatter(x=x_values, y=y_values)
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    ax.legend()
    plt.title(title)

    return


def analyze_experiment(experiment):
    weight_table = load_weights()
    vel_table = new_velocity_table()
    momenta_table = new_before_after_table()
    for selected_weight in weight_table.iterrows():
        run_full_analysis(experiment, selected_weight, vel_table, momenta_table)

    momenta_table[('', '  ratio')] = (momenta_table[('After', 'Kinetic energy (J)')] /
                                      momenta_table[('Before', 'Kinetic energy (J)')])

    scatter_plot_with_line_fit(momenta_table, momenta_table[('Before', 'Momentum (kg-m/s)')],
                                             momenta_table[('After', 'Momentum (kg-m/s)')], 'Momentum before (kg-m/s)',
                                             'Momentum after (kg-m/s)', "Momentum")
    plt.savefig(f"plots/{experiment}/MomentumBeforeAfter.png", dpi=300)
    plt.close()

    scatter_plot_with_line_fit(momenta_table, momenta_table[('Before', 'Kinetic energy (J)')],
                                             momenta_table[('After', 'Kinetic energy (J)')],
                                             'Kinetic energy before (J)',
                                             'Kinetic energy after (J)', "Kinetic energy")
    plt.savefig(f"plots/{experiment}/KineticBeforeAfter.png", dpi=300)
    plt.close()

    vel_table.to_csv(f"results/{experiment}/{experiment}_collisions_velocities.csv")
    momenta_table.to_csv(f"results/{experiment}/{experiment}_collisions_momenta.csv")
    export_to_tex(vel_table, f"results/tex/{experiment}/{experiment}_collisions_velocities.tex", experiment)
    export_to_tex(momenta_table, f"results/tex/{experiment}/{experiment}_collisions_momenta.tex", experiment)


def export_to_tex(df, path, experiment):
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
        {'selector': 'midrule', 'props': ':hline;'},
        {'selector': 'bottomrule', 'props': ':hline;'},
    ], overwrite=False)
    label = f"tbl:{experiment}"
    caption = f"Data for the {experiment.lower()} collision."
    df_tex = s.to_latex(multicol_align='c|', column_format="|" + "r|" * (n_cols + 1),
                        position="H", position_float="centering", label=label, caption=caption)
    with open(path, "w") as f:
        f.write(df_tex)
        f.flush()


def main():
    experiments = ["Elastic", "Explosive", "Inelastic"]
    for experiment in experiments:
        analyze_experiment(experiment)


if __name__ == "__main__":
    main()
