from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
import cv2
import math
from scipy.spatial.transform import Rotation
import copy

def test_cases():
    # ================ Test find_idx_of_nearest() ================
    array = [0, 1, 2, 3, 4, 5]
    np.testing.assert_equal(3, find_idx_of_nearest(array, 3))

    array = [0, 1, 2, 3, 4, 5, 7]
    np.testing.assert_equal(1, find_idx_of_nearest(array, 1.3))

    with np.testing.assert_raises(ValueError):
        array = [7, 0, 1, 2, 3, 4, 5, 7]
        np.testing.assert_equal(1, find_idx_of_nearest(array, 1.3))

    with np.testing.assert_raises(ValueError):
        array = [0, 1, 1, 1, 2, 3, 4, 5, 7]
        np.testing.assert_equal(1, find_idx_of_nearest(array, 1.3))

    # ============== Test load_images_and_timestamps() ==========
    bag_path = 'scenario1_robotA'
    path_to_dataset = Path(Path.cwd(), 'dataset_AirMuseum_Seq1', bag_path, bag_path[10:], 'original')
    path_to_bag_100 = Path(path_to_dataset, 'cam100_imu.bag')
    topic_name = '/robotA/cam100/image_raw'
    timestamps, timestamps_num, images = load_images_and_timestamps(path_to_bag_100, topic_name, 3, 5)
    
    np.testing.assert_equal(len(timestamps), 3)
    np.testing.assert_equal(len(timestamps_num), 3)
    np.testing.assert_equal(len(images), 3)
    np.testing.assert_equal(1566474939.199000000, timestamps_num[0])
    np.testing.assert_equal(1566474939, timestamps[0].sec)
    np.testing.assert_equal(199000000, timestamps[0].nanosec)
    image_des_20 = [14, 12, 14, 13, 15, 13, 12, 11, 12, 11, 12, 11, 10, 10, 10, 11, 10, 11, 9, 11]
    np.testing.assert_array_equal(image_des_20, images[0][0,:20])

def calculate_avg_time_diff(ts_num_1, ts_num_2):
    total = 0
    for i in range(len(ts_num_1)):
        total += (np.abs(ts_num_1[i] - ts_num_2[i]))
    return total/len(ts_num_1)

def find_idx_of_nearest(array, value):
    # Make sure the array is sorted
    if not np.all(array[:-1] <= array[1:]):
        raise ValueError("Input array is not sorted")
    
    # Make sure that there are no duplicate values
    unique = np.unique(array)
    if len(unique) != len(array):
        raise ValueError("Array has duplicate entries")

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array)) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
        return idx-1
    else:
        return idx

def load_images_and_timestamps(bag_path, topic_name, num_images, starting_index):
    # Arrays
    timestamps = []
    timestamps_num = []
    images = []

    # Create reader instance and open for reading.
    i = 0
    with AnyReader([bag_path]) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if starting_index is not None and i < starting_index:
                i += 1
                continue
            if num_images is not None and i >= num_images + starting_index:
                break
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamps.append(msg.header.stamp)
            timestamps_num.append(msg.header.stamp.sec + (msg.header.stamp.nanosec / (10**9)))
            image_data = np.reshape(np.array(msg.data, dtype=np.uint8), (512, 640))
            images.append(image_data)
            i += 1

    return timestamps, timestamps_num, images

def extract_images_and_ts(bag_path, topic_name, path_name, visualize_flow=False, visualize_disparity=True):
    # Set entries to save
    num_dataset_entries = 100
    starting_index = 400

    # Set up a reader to read the rosbags and GT
    path_to_dataset = Path(Path.cwd(), 'dataset_AirMuseum_Seq1', bag_path, bag_path[10:], 'original')
    path_to_bag_100 = Path(path_to_dataset, 'cam100_imu.bag')
    path_to_bag_101 = Path(path_to_dataset, 'cam101.bag')
    path_to_gt_pose_original = Path(path_to_dataset, "cam100_stamped_groundtruth.txt").absolute()

    # Set up paths to save files
    path_to_save_100_images = Path(Path.cwd(), 'datasets', path_name, 'image_0').absolute()
    path_to_save_101_images = Path(Path.cwd(), 'datasets', path_name, 'image_1').absolute()
    path_to_save_100_flow = Path(Path.cwd(), 'datasets', path_name, 'flow').absolute()
    path_to_gt_pose_vdo = Path(Path.cwd(), 'datasets', path_name, "pose_gt.txt").absolute()
    path_to_times_file = Path(Path.cwd(), 'datasets', path_name, 'times.txt').absolute()

    # Read in all the pose lines
    pose_ts = []
    pose_data = []
    with open(path_to_gt_pose_original, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split(" ")
            pose_ts.append(line_split[0])
            rest = line_split[1:]
            pose_data.append(rest)
    pose_ts = np.array(pose_ts)

    # Remove comments from pose data
    pose_ts = pose_ts[1:]
    pose_data = pose_data[1:]
    pose_ts_num = copy.deepcopy(pose_ts).astype(np.float128)

    # Convert to floats
    for i in range(len(pose_ts)):
        pose_ts[i] = float(pose_ts[i])

    for i in range(len(pose_data)):
        for j in range(len(pose_data[i])):
            pose_data[i][j] = float(pose_data[i][j])

    # Keep only poses we want
    pose_ts = np.array(pose_ts[starting_index:starting_index+num_dataset_entries])
    pose_data = np.array(pose_data[starting_index:starting_index+num_dataset_entries]).astype(np.float128)
    pose_ts_num = pose_ts_num[starting_index:starting_index+num_dataset_entries]
    np.testing.assert_equal(num_dataset_entries, len(pose_ts))

    # Write the pose_gt.txt file
    with open(path_to_gt_pose_vdo, "w") as f:
        # Iterate through each pose
        for i in range(0, len(pose_ts)):        
            # Calculate the H matrix for this translation and quaternion
            T = np.array([pose_data[i][0:3]]).transpose()
            R = Rotation.from_quat(np.array(pose_data[i][3:])).as_matrix()
            H = np.concatenate((R, T), axis=1)
            H = np.concatenate((H, np.array([[0, 0, 0, 1]])), axis=0)

            # Write the result
            line = str(pose_ts[i])
            for row in H:
                for val in row:
                    line += " " + str(val)
            line += "\n"
            f.write(line)

    # Load images and timestamps for both cameras
    ts_1, ts_num_1, images_1 = load_images_and_timestamps(path_to_bag_100, topic_name + "/cam100/image_raw", None, None)
    ts_2, ts_num_2, images_2 = load_images_and_timestamps(path_to_bag_101, topic_name + "/cam101/image_raw", None, None)

    # Keep track of used images
    used_images_1_idx = []
    used_images_2_idx = []
    used_images_gt_ts = []
    used_images_1_ts = []
    used_images_2_ts = []

    # Write the closest camera 100 and camera 101 images to each pose
    for i in range(0, len(pose_ts_num)):
        # Get closest timestamp index for first camera
        idx1 = find_idx_of_nearest(ts_num_1, pose_ts_num[i])
        used_images_1_ts.append(ts_num_1[idx1])

        # Get closest timestamp index for second camera
        idx2 = find_idx_of_nearest(ts_num_2, pose_ts_num[i])
        used_images_2_ts.append(ts_num_2[idx2])

        # Make sure we haven't already used either of these images
        if idx1 in used_images_1_idx:
            raise AssertionError("This first camera image is closest to 2 GT poses! Something is wrong.")
        if idx2 in used_images_2_idx:
            raise AssertionError("This second camera image is closest to 2 GT poses! Something is wrong.")
        used_images_1_idx.append(idx1)
        used_images_2_idx.append(idx2)

        used_images_gt_ts.append(pose_ts_num[i])

        # Write both images
        image_name = str(i).rjust(6, '0') + ".jpeg"
        if not cv2.imwrite(Path(path_to_save_100_images, image_name), images_1[idx1]):
            print("Could not write image")
        if not cv2.imwrite(Path(path_to_save_101_images, image_name), images_2[idx2]):
            print("Could not write image")

    # Validate the the error is low enough
    print("Converting " + bag_path + "...")
    print("(Camera 1) Average Time Sync Difference to GT: ", calculate_avg_time_diff(used_images_gt_ts, used_images_1_ts))
    print("Standard Deviation: ", np.std(np.array(used_images_gt_ts) - np.array(used_images_1_ts)))
    print("(Camera 2) Average Time Sync Difference to GT: ", calculate_avg_time_diff(used_images_gt_ts, used_images_2_ts))
    print("Standard Deviation: ", np.std(np.array(used_images_gt_ts) - np.array(used_images_2_ts)))
    print("Images taken every 0.05 seconds (20 Hertz) and GT Pose every 0.1 seconds (10 Hz). Make sure timestamp error above is reasonable.\n")

    # Write the timestamps to a file
    with open(path_to_times_file, "w") as f:
        for i, stamp in enumerate(pose_ts_num):
            time_str = str(stamp)
            if i < len(pose_ts_num) - 1:
                time_str += "\n"
            f.write(time_str)

    # Calculate dense optical flow using OpenCV
    for i in range(len(used_images_1_idx) - 1):
        image_prev = images_1[used_images_1_idx[i]]
        image_next = images_1[used_images_1_idx[i+1]]

        # Calculate dense optical flow using Farneback's method
        flow = cv2.calcOpticalFlowFarneback(
            prev=image_prev,
            next=image_next,
            flow=None,
            pyr_scale=0.5,  # Image scale (<1) to build pyramids
            levels=3,       # Number of pyramid layers
            winsize=15,     # Averaging window size
            iterations=3,   # Number of iterations at each pyramid level
            poly_n=5,       # Size of pixel neighborhood
            poly_sigma=1.2, # Gaussian standard deviation
            flags=0         # Operation flags (0 for default)
        )

        # Save the calculated flow
        flow_name = str(i).rjust(6, '0') + ".flo"
        cv2.writeOpticalFlow(str(Path(path_to_save_100_flow, flow_name)), flow)

        # Visualize the optical flow (convert flow to HSV)
        if visualize_flow:
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(cv2.cvtColor(image_prev, cv2.COLOR_GRAY2BGR))
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
            hsv[..., 1] = 255                      # Saturation: full
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            #Display the result
            cv2.imshow('Dense Optical Flow', rgb_flow)
            cv2.waitKey(0)

    # Calculate disparity map using OpenCV StereoSGBM
    # for i in range(len(used_images_1_idx)):
    #     image_left = images_2[used_images_2_idx[i]]
    #     image_right = images_1[used_images_1_idx[i]]

    #     # Create a StereoBM object and compute the disparity map
    #     stereo = cv2.StereoBM.create(numDisparities=16*6, blockSize=25)
    #     disparity = stereo.compute(image_left,image_right)
    #     disparity = np.clip(disparity, 0, 255)
    #     print(disparity)

    #     # Normalize the disparity map for visualization
    #     if visualize_disparity:
    #         import matplotlib.pyplot as plt
    #         disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #         print(disparity)
    #         cv2.imshow('Dense Optical Flow', disparity)
    #         cv2.waitKey(0)

    cv2.destroyAllWindows()

def main():
    # Run test cases
    test_cases()

    # NOTE: At least for robot A, cam100 is right eye, cam101 is left eye
    extract_images_and_ts('scenario1_robotA', '/robotA', 'AirMuseum-Seq1-A')
    extract_images_and_ts('scenario1_robotB', '/robotB', 'AirMuseum-Seq1-B')

if __name__ == "__main__":
    main()