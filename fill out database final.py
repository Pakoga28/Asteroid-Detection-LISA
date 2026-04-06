import math
import matplotlib.pyplot as plt
import numpy.random as rand

# file has order: "e","a","i","om","w","H","albedo","per_y","tp_cal"
# pull in database raw data for filling
with open("data/sbdb_query_results.csv") as file:
    file.readline()  # skip header line
    data = file.readlines()

# open/create file to hold full dataset
with open("data/fullData.csv", "w+") as new_file:
    # add column headers to file
    new_file.write("'e', 'a', 'i', 'om', 'w', 'diameter', 'H', 'albedo', 'period', 'tp', 'type', 'mass',\n")

    for index, line in enumerate(data):
        # Process existing data into an array
        lineArr = line.strip("\n").split(",")

        newline = [""] * 12  # Ensure newline has 12 elements

        # booleans used in logic below
        hadH = lineArr[5] != ""  # apparently there CAN be blank H values
        hadAlbedo = lineArr[6] != ""
        hadPeriod = lineArr[7] != ""

        # populate new line with orbital parameters
        newline[0] = lineArr[0]  # e
        newline[1] = lineArr[1]  # a
        newline[2] = lineArr[2]  # i
        newline[3] = lineArr[3]  # om
        newline[4] = lineArr[4]  # w
        newline[5] = ""  # diameter not provided
        newline[6] = lineArr[5]  # H
        newline[7] = lineArr[6]  # albedo
        newline[8] = lineArr[7]  # period in sidereal years
        newline[9] = lineArr[8]  # time of perihelion passage

        if not hadH:
            print(f"line {index} skipped due to missing H-value.")
            continue

        # section for filling albedos and picking asteroid type
        if hadAlbedo:
            # grab given albedo
            albedo = float(lineArr[6])
            # find type from albedo
            if abs(albedo - 0.03) <= 0.001:
                astType = "C"
            elif albedo > 0.18:
                astType = "S"
            else:
                # if [0.1, 0.18), pick S or M based on rarity distribution
                if rand.uniform(0.0, 0.25) <= 0.17:
                    astType = "S"
                else:
                    astType = "M"
        else:
            # find albedo from a random type
            randNum = rand.uniform(0.0, 1.0)
            if randNum <= 0.75:  # 75% C
                astType = "C"
                albedo = 0.03
            elif randNum <= 0.92:  # 17% S
                astType = "S"
                albedo = rand.uniform(0.1, 0.22)
            else:  # 8% M
                astType = "M"
                albedo = rand.uniform(0.1, 0.18)
            newline[7] = str(albedo)
        newline[10] = astType

        # since diameter is not provided, we'll derive it from H and albedo
        diameter = 10 ** (-float(lineArr[5]) / 5.0) / math.sqrt(float(albedo)) * 1329
        newline[5] = str(diameter)

        # get density from type change the density based on units
        if astType == "C":
            density = rand.uniform(1600.0, 3030.0)
        elif astType == "S":
            density = rand.uniform(3220.0, 3420.0)
        else:
            density = rand.uniform(4010.0, 7370.0)

        # get radius in meters
        radius = float(newline[5]) * 1000 / 2
        # calculate mass
        volume = math.pow(radius, 3) * math.pi * (4 / 3)
        newline[11] = str(volume * density)

        # write to file
        for element in newline:
            new_file.write(element.strip("[]'") + ", ")
        new_file.write("\n")
