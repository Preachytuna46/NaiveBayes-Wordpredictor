def split_file_into_sixths(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)
    sixth_lines = total_lines // 6  # Calculate one sixth of the total lines

    first_sixth = lines[:(sixth_lines)]  # Extract the first sixth

    # Save only the first sixth to a file
    with open('android_trainingNEW.txt', 'w') as first_file:
        first_file.writelines(first_sixth)


def main():
    split_file_into_sixths("android_train.txt")


if __name__ == "__main__":
    main()
