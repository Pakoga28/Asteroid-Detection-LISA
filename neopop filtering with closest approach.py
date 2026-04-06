import pandas as pd
import math
import numpy.random as rand


# loading in the data that was modified prior, naming the columns since they were unnamed before
def load_data(file_path):
    df = pd.read_csv(file_path, skiprows=60)
    df.columns = ['a', 'e', 'i', 'om', 'w', 'H', 'diameter', 'albedo']
    return df


# using Keppler's third law to calculate the period
def calculate_period(G, m, a):
    T = math.sqrt(((4 * math.pi ** 2) / (G * m)) * (a ** 3))
    return T


# using the Bronicki method to calculate the mass of the asteroids through albedo and statistical matching of densities
def calculate_mass(albedo, diameter):
    if abs(albedo - 0.03) <= 0.001:
        astType = "C"
    elif albedo > 0.18:
        astType = "S"
    else:
        if rand.uniform(0.0, 0.25) <= 0.17:
            astType = "S"
        else:
            astType = "M"

    if astType == "C":
        density = rand.uniform(1600.0, 3030.0)
    elif astType == "S":
        density = rand.uniform(3220.0, 3420.0)
    else:
        density = rand.uniform(4010.0, 7370.0)

    radius = float(diameter) * 1000 / 2
    volume = math.pow(radius, 3) * math.pi * (4 / 3)
    mass = volume * density
    return mass


# function that is going to calculate the closest distance between the two orbits and then uses the mass to calculate
# the maximum acceleration that we could get from the two orbits
def process_population(df, G, r_E):
    # create empty lists that will get filled as we iterate and filter through the entire population
    filtered_population = []
    masses = []
    periods = []
    Magnitudes = []
    Accelerations = []
    Distances = []

    for idx, row in df.iterrows():
        # access each column in the rows
        e = row['e']
        a = row['a']
        i = math.radians(row['i'])
        om = math.radians(row['om'])
        w = math.radians(row['w'])
        albedo = row['albedo']
        diameter = row['diameter']
        H = row["H"]

        mass = calculate_mass(albedo, diameter)
        T = calculate_period(G, mass, a)

        # the following part of the code is a calculation of the closest distance that was taken from Shane Larson's
        # Meteor paper
        n1 = math.sin(i) * math.sin((2 * math.pi) - w + om)
        n2 = -math.sin(i) * math.cos((2 * math.pi) - w + om)
        n3 = math.cos(i)
        rp = a * (1 - e)

        rp1 = (
                rp * (math.cos(w) * math.cos((2 * math.pi) - w + om) - (math.sin(w) * math.cos(i)
                                                                        * math.sin((2 * math.pi) - w + om)))
        )

        rp2 = (
                rp * (math.cos(w) * math.sin((2 * math.pi) - w + om) + (math.sin(w) * math.cos(w)
                                                                        * math.cos((2 * math.pi) - w + om)))
        )

        rp3 = (
                rp * (math.sin(w) * math.sin(i))
        )

        L1 = n2
        L2 = -n1
        L3 = 0

        cos_theta_numerator = (rp1 * L1) + (rp2 * L2) + (rp3 * L3)
        cos_theta_denominator = rp * (math.sqrt(math.pow(L1, 2) + math.pow(L2, 2) + math.pow(L3, 2)))
        cos_theta = cos_theta_numerator / cos_theta_denominator

        r_ast = (a * (1 - math.pow(e, 2)) / (1 + (e * cos_theta))) * 1.49597870700E11
        d = abs(r_E - r_ast)

        # use a = G * M / r^2 to calculate the acceleration
        accel = 6.67430 * math.pow(10, -11) * mass / math.pow(d, 2)

        # this is where we filter the data. If the maximum acceleration is over 10^-17 then we keep the row and add
        # the calculated mass and period
        if accel > math.pow(10, -16):
            filtered_population.append(row)
            masses.append(mass)
            periods.append(T)
            Magnitudes.append(H)
            Accelerations.append(accel)
            Distances.append(d)

    print('Data filtered')
    print(len(Accelerations))
    # simply add the data to dataframes to convert to a csv sheet
    filtered_df = pd.DataFrame(filtered_population)
    filtered_df['mass'] = masses
    filtered_df['period'] = periods
    Acc_H_df = pd.DataFrame({
        'H': Magnitudes,
        'Acceleration': Accelerations,
        'Mass': masses,
        'Distance': Distances,
    })

    return filtered_df, Acc_H_df


def save_filtered_population(df, file_path):
    df.to_csv(file_path, index=False)


# run all the functions
if __name__ == "__main__":
    G = 6.67430e-11
    r_E = 1.471E11

    input_file = 'neopop/population_6.csv'
    output_file = 'neopop/filtered_population_6.csv'

    df = load_data(input_file)
    print('Data loaded')
    filtered_df, Acc_H_df = process_population(df, G, r_E)
    save_filtered_population(filtered_df, output_file)
    Acc_H_df.to_csv('neopop/Acc_H.csv', index=False)
    print('Data saved to file')

