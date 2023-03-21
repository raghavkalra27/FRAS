import os
import face_recognition

def calculate_accuracy(dataset_path, tolerance=0.6):
    """
    Calculates the accuracy of a face recognition system based on a labeled dataset of images.

    Args:
        dataset_path (str): Path to the dataset directory containing subdirectories with images of each person.
        tolerance (float): Tolerance for face distance comparisons (default=0.6).

    Returns:
        float: The accuracy of the face recognition system as a percentage.
    """

    # Load the dataset
    print('Start....')
    dataset = {}
    for person_dir in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_dir)
        if os.path.isdir(person_path):
            images = []
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                image = face_recognition.load_image_file(image_path)
                images.append(image)
            dataset[person_dir] = images

    # Run the face recognition system on the dataset

    num_correct = 0
    num_total = 0
    for person in dataset:
        for image in dataset[person]:
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            for face_encoding in face_encodings:
                # Compare the face encoding to known encodings of each person in the dataset
                for known_person in dataset:
                    known_encodings = [face_recognition.face_encodings(img)[0] for img in dataset[known_person]]
                    results = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
                    trueCount = 0
                    for i in results:
                        if i == True: 
                            trueCount += 1
                    print('Accuracy', (trueCount / len(results)) * 100, '%' )

calculate_accuracy('images')