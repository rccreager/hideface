

def get_ground_truth(image_number, path_to_test_file):
    with open(path_to_test_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        print('start loop')
        for i in range(0, len(lines)):
            line = lines[i]
            if (image_number+'.jpg') in line:
                num_faces = int(lines[i+1])
                print('num faces: ' + str(num_faces))
                for j in range(2,num_faces+2):
                    print(lines[i+j])
