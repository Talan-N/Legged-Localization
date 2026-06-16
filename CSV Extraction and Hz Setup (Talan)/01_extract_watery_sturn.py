import os
import csv
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from tf_transformations import euler_from_quaternion


def detect_storage_id(bag_path):
    metadata_path = os.path.join(bag_path, "metadata.yaml")

    if not os.path.exists(metadata_path):
        return "sqlite3"

    with open(metadata_path, "r") as f:
        metadata = f.read()

    if "storage_identifier: mcap" in metadata:
        return "mcap"

    if "storage_identifier: sqlite3" in metadata:
        return "sqlite3"

    return "sqlite3"


def open_bag(bag_path):
    print(f"Opening bag: {bag_path}")

    storage_id = detect_storage_id(bag_path)
    print(f"Using storage_id: {storage_id}")

    reader = SequentialReader()

    storage_options = StorageOptions(
        uri=bag_path,
        storage_id=storage_id,
    )

    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)
    return reader


def get_topic_types(reader):
    topic_types = reader.get_all_topics_and_types()
    return {t.name: t.type for t in topic_types}


def quat_to_rpy(q):
    return euler_from_quaternion([q.x, q.y, q.z, q.w])


def require_topic(type_map, topic_name):
    if topic_name not in type_map:
        print("Available topics:")
        for topic, msg_type in type_map.items():
            print(f"  {topic}: {msg_type}")
        raise KeyError(f"Topic not found in bag: {topic_name}")


def extract_odometry(bag_path, topic_name, output_csv):
    reader = open_bag(bag_path)
    type_map = get_topic_types(reader)
    require_topic(type_map, topic_name)

    msg_type = get_message(type_map[topic_name])
    rows = []

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)
        stamp = t * 1e-9

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        roll, pitch, yaw = quat_to_rpy(q)

        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        rows.append([
            stamp,
            p.x, p.y, p.z,
            roll, pitch, yaw,
            v.x, v.y, v.z,
            w.x, w.y, w.z,
        ])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "x", "y", "z",
            "roll", "pitch", "yaw",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
        ])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def extract_vicon_pose(bag_path, topic_name, output_csv):
    reader = open_bag(bag_path)
    type_map = get_topic_types(reader)
    require_topic(type_map, topic_name)

    msg_type = get_message(type_map[topic_name])
    rows = []

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)
        stamp = t * 1e-9

        p = msg.pose.position
        q = msg.pose.orientation
        roll, pitch, yaw = quat_to_rpy(q)

        rows.append([
            stamp,
            p.x, p.y, p.z,
            roll, pitch, yaw,
        ])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "x", "y", "z",
            "roll", "pitch", "yaw",
        ])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def extract_lowstate(bag_path, topic_name, output_csv):
    reader = open_bag(bag_path)
    type_map = get_topic_types(reader)
    require_topic(type_map, topic_name)

    msg_type = get_message(type_map[topic_name])
    rows = []

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)
        stamp = t * 1e-9

        row = [stamp]

        row += list(msg.imu_state.accelerometer)
        row += list(msg.imu_state.gyroscope)
        row += list(msg.imu_state.rpy)

        row += list(msg.foot_force)
        row += list(msg.foot_force_est)

        for i in range(12):
            m = msg.motor_state[i]
            row += [m.q, m.dq, m.tau_est]

        rows.append(row)

    header = ["time_s"]

    header += ["accel_x", "accel_y", "accel_z"]
    header += ["gyro_x", "gyro_y", "gyro_z"]
    header += ["roll", "pitch", "yaw"]

    header += [f"foot_force_{i}" for i in range(4)]
    header += [f"foot_force_est_{i}" for i in range(4)]

    for i in range(12):
        header += [f"q_{i}", f"dq_{i}", f"tau_{i}"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


def main():
    os.makedirs("raw_csv", exist_ok=True)

    estimate_bag = "/mnt/c/Users/talan/Downloads/Official_Tests/Watery/Bags/Watery-S-Turn-Estimate"
    truth_bag = "/mnt/c/Users/talan/Downloads/Official_Tests/Watery/Bags/Watery-S-Turn-Truth"

    print("Estimate bag:", estimate_bag)
    print("Truth bag:", truth_bag)

    print("Estimate exists:", os.path.exists(estimate_bag))
    print("Truth exists:", os.path.exists(truth_bag))

    extract_odometry(
        estimate_bag,
        "/odometry/filtered",
        "raw_csv/watery_sturn_go2_odom.csv",
    )

    # Keep this commented out until unitree_go messages are installed/sourced.
    extract_lowstate(
       estimate_bag,
       "/lowstate",
       "raw_csv/watery_sturn_lowstate.csv",
    )

    extract_vicon_pose(
        truth_bag,
        "/vrpn_mocap/Go2_test/pose",
        "raw_csv/watery_sturn_vicon.csv",
    )


if __name__ == "__main__":
    rclpy.init()
    main()
    rclpy.shutdown()