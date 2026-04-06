import pandas as pd
import math


# helper function for Kepler's equation
def kepler_equation(E, m, e):
    return E - e * math.sin(E) - m


def bisection_method(m, e, tol=1e-6, max_iter=1000):
    E_low = 0
    E_high = 2 * math.pi
    nstep = 0

    # manually solve for m = 0
    if m == 0:
        return 0
    for i in range(max_iter):
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


def calculate_time_elapsed(tp):
    # split the given date into its sub parts
    parts = tp.split('-')
    if len(parts) > 0:
        last_part = parts[-1]
        subparts = last_part.split('.')
        parts[-1] = subparts[0]
        if len(subparts) > 1:
            parts.append(subparts[1])

    year = int(parts[0]) if len(parts) > 0 else 0
    month = int(parts[1]) if len(parts) > 1 else 0
    day = int(parts[2]) if len(parts) > 2 else 0
    hour = int(parts[3]) if len(parts) > 3 else 0

    # calculate the total time difference between Date of Periapsis and Lisa starting date
    # the month and day can be adjusted to start LISA at different dates
    t_y = 2035 - year
    t_m = (6.0 - month) / 12.0
    t_d = (1.0 - day) / 365.0
    t_h = (-hour) / 8760.0

    return t_y + t_m + t_d + t_h


def calculate_coordinates(df):
    coords = []

    for idx, row in df.iterrows():
        e = row['e']
        a = row['a']
        i = math.radians(row['i'])
        om = math.radians(row['om'])
        w = math.radians(row['w'])
        T = row['period']
        tp = row['tp']

        t_tot = calculate_time_elapsed(tp)

        for n in range(4383):
            t_E = ((t_tot / T) - int(t_tot / T)) * T
            M = 2 * math.pi * t_E / T
            # use Kepler's equation to calculate the coordinates in 2d plane
            E = bisection_method(M, e)

            b = a * math.sqrt(1 - math.pow(e, 2))

            x_i = a * (math.cos(E) - e)
            y_i = b * math.sin(E)
            z_i = 0

            # use rotation matrix to bring them into 3d plane
            x = ((math.cos(om) * math.cos(w) - math.cos(i) * math.sin(om) * math.sin(w)) * x_i
                 - (math.cos(om) * math.sin(w) + math.cos(i) * math.sin(om) * math.cos(w)) * y_i)

            y = ((math.sin(om) * math.cos(w) + math.cos(i) * math.cos(om) * math.sin(w)) * x_i
                 - (math.sin(om) * math.sin(w) - math.cos(i) * math.cos(om) * math.cos(w)) * y_i)

            z = math.sin(i) * math.sin(w) * x_i + math.sin(i) * math.cos(w) * y_i

            coords.append({'x': x, 'y': y, 'z': z})

            t_tot += (1 / 365.25)

    return coords


def load_data():
    # Load the data from calculation of coordinates and LISA orbit data
    df1 = pd.read_csv('data/lisa_12_years_June_day.csv', header=None)
    df2 = pd.read_csv('data/ast_coords.csv')
    df3 = pd.read_csv('data/filtered_data.csv')

    # Define column names
    df1.columns = ['sat1_x', 'sat1_y', 'sat1_z', 'sat2_x', 'sat2_y', 'sat2_z', 'sat3_x', 'sat3_y', 'sat3_z']
    df2.columns = ['ast_x', 'ast_y', 'ast_z']

    return df1, df2, df3


def extract_coordinates(df1, df2):
    # Extract satellite coordinates
    sat1_coords = df1[['sat1_x', 'sat1_y', 'sat1_z']].apply(tuple, axis=1).tolist()
    sat2_coords = df1[['sat2_x', 'sat2_y', 'sat2_z']].apply(tuple, axis=1).tolist()
    sat3_coords = df1[['sat3_x', 'sat3_y', 'sat3_z']].apply(tuple, axis=1).tolist()

    # Extract asteroid coordinates
    ast_coords = df2[['ast_x', 'ast_y', 'ast_z']].apply(tuple, axis=1).tolist()

    return sat1_coords, sat2_coords, sat3_coords, ast_coords


def split_list(input_list, num_parts, part_size):
    # Split the list into specified number of parts with a given part size.
    split_lists = [input_list[i * part_size:(i + 1) * part_size] for i in range(num_parts)]
    return split_lists


# helper functions for distance calculation
def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2 + (coord1[2] - coord2[2])**2)


def calculate_accelerations(split_parts, sat_coords, G, df3):
    AU_TO_METERS = 1.496e+11
    accelerations = []
    all_distances = []
    for parts, (index, row) in zip(split_parts, df3.iterrows()):
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


def filter_accelerations(lists, lists2):
    # Define the chunk size
    chunk_size = 4383

    # Initialize lists to store the results
    indices = []
    max_accelerations = []
    min_distances = []
    # Iterate through the data in chunks
    for start in range(0, len(lists), chunk_size):
        end = start + chunk_size
        chunk = lists[start:end]

        # Find the maximum acceleration and its index in the chunk
        max_acceleration = max(chunk)
        max_accelerations.append(max_acceleration)
        max_index = start + chunk.index(max_acceleration)
        indices.append(max_index)
    for start in range(0, len(lists2), chunk_size):
        end = start + chunk_size
        chunk = lists2[start:end]

        # Find the minimum distance and its index in the chunk
        min_distance = min(chunk)
        min_distances.append(min_distance)
    return indices, max_accelerations, min_distances


def final_indices(list1, list2, list3, acc1, acc2, acc3):
    indices1 = []
    indices2 = []
    indices3 = []
    for item1, item2, item3, item4, item5, item6 in zip(list1, list2, list3, acc1, acc2, acc3):
        # if one of the max accelerations that correspond to indices then add all three indices
        if (item4 or item5 or item6) > 10 ** -18:
            indices1.append(item1)
            indices2.append(item2)
            indices3.append(item3)

    return indices1, indices2, indices3

def save_results_to_csv(sat1_acc, sat2_acc, sat3_acc, indices1, indices2, indices3,
                        max_acc1, max_acc2, max_acc3, dist1, dist2, dist3):

    # Create a DataFrame to store the results
    satellite_df = pd.DataFrame({
        'acceleration1': sat1_acc,
        'acceleration2': sat2_acc,
        'acceleration3': sat3_acc
    })

    # Save the DataFrame to a new CSV file
    satellite_df.to_csv('data/final_accelerations.csv', index=False)

    max_data_df = pd.DataFrame({
        'acceleration1': max_acc1,
        'acceleration2': max_acc2,
        'acceleration3': max_acc3
    })

    # Save the DataFrame to a new CSV file
    max_data_df.to_csv('data/max_data.csv', index=False)

    # Create a DataFrame for the filtered accelerations
    filtered_acc_df = pd.DataFrame({
        'Satellite 1 Indices': indices1,
        'Satellite 2 Indices': indices2,
        'Satellite 3 Indices': indices3,
    })

    # Save the filtered accelerations DataFrame to a CSV file
    filtered_acc_df.to_csv('data/indices_max_acc.csv', index=False)

    distance_df = pd.DataFrame({
        'Distance 1': dist1,
        'Distance 2': dist2,
        'Distance 3': dist3,
    })

    distance_df.to_csv('data/final_distances.csv', index=False)


if __name__ == "__main__":
    df = pd.read_csv('data/filtered_data.csv')
    print('File read')
    coords = calculate_coordinates(df)
    print('Coordinates calculated')
    coords_df = pd.DataFrame(coords)
    coords_df.to_csv('data/ast_coords.csv', index=False)
    print('Data saved')

    # Load the data
    df1, df2, df3 = load_data()
    print('Data loaded')
    # Extract coordinates
    sat1_coords, sat2_coords, sat3_coords, ast_coords = extract_coordinates(df1, df2)
    print('Coordinates extracted')
    # Split asteroid coordinates into parts
    split_parts = split_list(ast_coords, len(df3), 4383)
    print('Coordinates split')
    G = 6.67430e-11

    # Calculate accelerations
    sat1_acc, dist1 = calculate_accelerations(split_parts, sat1_coords, G, df3)
    sat2_acc, dist2 = calculate_accelerations(split_parts, sat2_coords, G, df3)
    sat3_acc, dist3 = calculate_accelerations(split_parts, sat3_coords, G, df3)
    print('accelerations calculated')

    # Filter accelerations
    indices1, max_acc1, min_dist1 = filter_accelerations(sat1_acc, dist1)
    indices2, max_acc2, min_dist2 = filter_accelerations(sat2_acc, dist2)
    indices3, max_acc3, min_dist3 = filter_accelerations(sat3_acc, dist3)
    print('Data filtered')

    indices1, indices2, indices3 = final_indices(indices1, indices2, indices3, max_acc1, max_acc2, max_acc3)
    print('Indices corrected')

    # Save results to CSV
    save_results_to_csv(sat1_acc, sat2_acc, sat3_acc, indices1, indices2, indices3,
                        max_acc1, max_acc2, max_acc3, min_dist1, min_dist2, min_dist3)
    print('Data written to file')