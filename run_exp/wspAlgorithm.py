#!/usr/bin/python2.7

# import yaml
import random
import pickle

def distance(point1, point2):
    dist = 0
    for (k1,v1),(k2,v2) in zip(point1.items(),point2.items()):
        dist += abs(v1-v2)
    return dist

def wsp(design_space,dmin):
    rmarray = list()
    processed = list()
    lowest = 0
    lowest_index = 0
    # print len(processed)
    # print len(design_space)-len(rmarray)
    while len(processed)<(len(design_space)-len(rmarray)) :
        initial = design_space[lowest_index]
        processed.append(lowest_index)
        for i in range(0,len(design_space)):
            if not(i in processed) and not (design_space[i] in rmarray):
                dist = distance(initial,design_space[i])
                if dist<dmin:
                    rmarray.append(design_space[i])
                else:
                    if lowest>dist:
                        lowest_index = i
    for i in rmarray:
        try:
            design_space.remove(i)
        except ValueError:
            print "Value not found", i

    return design_space

def get_data_points():
    design_space = list()
    sample = {'hs1': '1,45', 'hs2': '1,45', 'hs3': '1,45',
                'hs4': '1,45', 'hs5': '1,45'}
    for i in range(100):
        selected = list()
        result = dict(sample)
        for k in sample:
            vrange = sample[k]
            start = int(vrange.split(",")[0])
            end = int(vrange.split(",")[1])
            sel_idx = 0
            while True:
                idx = random.randint(start,end)
                if idx not in selected:
                    sel_idx = idx
                    break
            result[k] = sel_idx
            selected.append(sel_idx)
        design_space.append(dict(result))

    data_points = wsp(design_space, 40)
    return data_points

def main():

    while True:
        data_points = get_data_points()
        if len(data_points) == 100:
            break
        else:
            print len(data_points)
    
    exp_hotspots = list()
    for point in data_points:
        hotspots = list()
        for elem in point:
            hotspots.append(point[elem])
        exp_hotspots.append(sorted(hotspots))

    print exp_hotspots

    with open("hotspots.pkl", "w") as hs_file:
        pickle.dump(exp_hotspots, hs_file)

if __name__ == '__main__':
    main()
