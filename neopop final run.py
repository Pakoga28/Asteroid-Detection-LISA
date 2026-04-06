import pandas as pd
import math
import numpy.random as rand


def kepler_equation(E, m, e):
    return E - e * math.sin(E) - m


def bisection_method(m, e, tol=1e-6, max_iter=100):
    E_low = 0
    E_high = 2 * math.pi
    nstep = 0
    if m == 0:
        return 0
    elif m == math.pi:
        return math.pi
    for _ in range(max_iter):
        E_mid = (E_low + E_high) / 2.0
        f_mid = kepler_equation(E_mid, m, e)
        if abs(f_mid) < tol:
            return E_mid
        if kepler_equation(E_low, m, e) * f_mid < 0:
            E_high = E_mid
        else:
            E_low = E_mid
        nstep += 1
    raise ValueError(f"Bisection method did not converge to {tol} in {nstep} steps.")


def calculate_coordinates(df1, num_days=36525):
    coords = []
    t_tots = []
    for idx, row in df1.iterrows():
        e = row['e']
        a = row['a']
        i = math.radians(row['i'])
        om = math.radians(row['om'])
        w = math.radians(row['w'])
        T = row['period']
        t_tot = rand.uniform(0.0, T)
        t_tots.append(t_tot)

        for n in range(num_days):
            t_E = ((t_tot / T) - int(t_tot / T)) * T
            M = 2 * math.pi * t_E / T
            E = bisection_method(M, e)

            b = a * math.sqrt(1 - e**2)
            x_i = a * (math.cos(E) - e)
            y_i = b * math.sin(E)

            x = ((math.cos(om) * math.cos(w) - math.cos(i) * math.sin(om) * math.sin(w)) * x_i
                 - (math.cos(om) * math.sin(w) + math.cos(i) * math.sin(om) * math.cos(w)) * y_i)

            y = ((math.sin(om) * math.cos(w) + math.cos(i) * math.cos(om) * math.sin(w)) * x_i
                 - (math.sin(om) * math.sin(w) - math.cos(i) * math.cos(om) * math.cos(w)) * y_i)

            z = math.sin(i) * math.sin(w) * x_i + math.sin(i) * math.cos(w) * y_i
            coords.append({'x': x, 'y': y, 'z': z})
            t_tot += (1 / 365)
    return coords, t_tots


def load_data():
    df1 = pd.read_csv('data/lisa_12_years_June_day.csv', header=None)
    df3 = pd.read_csv('neopop/filtered_population_6.csv')
    coords, t_tots = calculate_coordinates(df3)

    df1.columns = ['sat1_x', 'sat1_y', 'sat1_z', 'sat2_x', 'sat2_y', 'sat2_z', 'sat3_x', 'sat3_y', 'sat3_z']
    df2 = pd.DataFrame(coords)
    df2.columns = ['ast_x', 'ast_y', 'ast_z']

    # Save for traceability
    df2.to_csv('neopop/ast_coords.csv', index=False)

    return df1, df2, df3


def extract_coordinates(df1, df2):
    sat1_coords = df1[['sat1_x', 'sat1_y', 'sat1_z']].apply(tuple, axis=1).tolist()
    sat2_coords = df1[['sat2_x', 'sat2_y', 'sat2_z']].apply(tuple, axis=1).tolist()
    sat3_coords = df1[['sat3_x', 'sat3_y', 'sat3_z']].apply(tuple, axis=1).tolist()
    ast_coords = df2[['ast_x', 'ast_y', 'ast_z']].apply(tuple, axis=1).tolist()
    return sat1_coords, sat2_coords, sat3_coords, ast_coords


def split_list(input_list, num_parts, part_size):
    return [input_list[i * part_size:(i + 1) * part_size] for i in range(num_parts)]


def euclidean_distance(coord1, coord2):
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(coord1, coord2)))


def calculate_accelerations(split_parts, sat_coords, G, df3):
    AU_TO_METERS = 1.496e+11
    accelerations = []
    all_distances = []
    for parts, (_, row) in zip(split_parts, df3.iterrows()):
        m = row['mass']
        distances = []
        for coord1, coord2 in zip(parts, sat_coords):
            distance = euclidean_distance(coord1, coord2) * AU_TO_METERS
            distances.append(distance)
            all_distances.append(distance)
        for d in distances:
            acc = G * m / (d ** 2)
            accelerations.append(acc)
    return accelerations, all_distances


def filter_accelerations(acc_list, dist_list, chunk_size=365):
    indices = []
    max_accelerations = []
    min_distances = []

    for start in range(0, len(acc_list), chunk_size):
        chunk = acc_list[start:start + chunk_size]
        max_acc = max(chunk)
        max_accelerations.append(max_acc)
        indices.append(start + chunk.index(max_acc))

    for start in range(0, len(dist_list), chunk_size):
        chunk = dist_list[start:start + chunk_size]
        min_distances.append(min(chunk))

    return indices, max_accelerations, min_distances


def output_data(max_acc1, max_acc2, max_acc3,df3):
    # Combine max accelerations into a DataFrame for comparison
    max_acc_df = pd.DataFrame({
        'sat1': max_acc1,
        'sat2': max_acc2,
        'sat3': max_acc3
    })

    overall_max_acc = max_acc_df.max(axis=1)
    filtered_acc = overall_max_acc > 1e-16
    selected_ast_indices = overall_max_acc[filtered_acc].index
    selected_accelerations = overall_max_acc[filtered_acc].values

    high_acc_asteroids = df3.iloc[selected_ast_indices].copy()
    high_acc_asteroids["maximum acceleration"] = selected_accelerations

    return high_acc_asteroids


def main_loop():
    df1, df2, df3 = load_data()
    sat1, sat2, sat3, ast_coords = extract_coordinates(df1, df2)

    part_num = len(df3)
    split_parts = split_list(ast_coords, part_num, 365)

    G = 6.67430e-11

    sat1_acc, dist1 = calculate_accelerations(split_parts, sat1, G, df3)
    sat2_acc, dist2 = calculate_accelerations(split_parts, sat2, G, df3)
    sat3_acc, dist3 = calculate_accelerations(split_parts, sat3, G, df3)

    i1, max_a1, min_d1 = filter_accelerations(sat1_acc, dist1)
    i2, max_a2, min_d2 = filter_accelerations(sat2_acc, dist2)
    i3, max_a3, min_d3 = filter_accelerations(sat3_acc, dist3)

    filtered_data = output_data(max_a1, max_a2, max_a3, df3)
    length = len(filtered_data)
    return filtered_data, length

if __name__ == "__main__":
    high_acc_asteroids = pd.DataFrame()
    asteroid_number = []
    indices = []
    for i in range(100):
        filtered_data_ast, length = main_loop()
        high_acc_asteroids = pd.concat([high_acc_asteroids, filtered_data_ast], ignore_index=True)
        index = sum(asteroid_number)
        indices.append(index)
        asteroid_number.append(length)
        print(f'Run {i} complete')

    length_and_indices = pd.DataFrame({
        'Number of Asteroids': asteroid_number,
        'Index in file': indices
    })

    high_acc_asteroids.to_csv('neopop/output/high_acc_asteroids_pop_6_100_year.csv', index=False)
    length_and_indices.to_csv('neopop/output/length_and_indices_pop_6_100_year.csv', index=False)

