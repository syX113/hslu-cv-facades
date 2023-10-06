import cv2 as cv
import os

ls = ['test','train','valid']

for x in ls:  
    # Define the folder containing the original images
    input_folder = f'./YOLOv8/building-facade-segmentation-instance-1/{x}/images/'

    # Define the folder where you want to save the grayscale images
    grayscale_folder = f'./YOLOv8/building-facade-segmentation-instance-1/{x}_grayscale/images'

    # Define the folder where you want to save the processed images
    processed_folder = f'./YOLOv8/building-facade-segmentation-instance-1/{x}_processedCLAHE/images'

    # Create the output folders if they don't exist
    os.makedirs(grayscale_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    # Get a list of files in the input folder
    file_list = os.listdir(input_folder)

    # Process the first 5 files
    for i, file_name in enumerate(file_list):
        # Check if the file is a JPG image
        if file_name.endswith('.jpg'):
            # Construct the full path to the input image
            input_image_path = os.path.join(input_folder, file_name)
            
            # Read the input image in grayscale
            img = cv.imread(input_image_path, cv.IMREAD_GRAYSCALE)
            
            # Check if the image was successfully read
            assert img is not None, f"File {input_image_path} could not be read, check with os.path.exists()"
            
            # Construct the full path to the grayscale output image
            grayscale_output_path = os.path.join(grayscale_folder, file_name)
            
            # Write the grayscale image to the grayscale folder
            cv.imwrite(grayscale_output_path, img)
            
            # Create a CLAHE object (Arguments are optional).
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(img)
            
            # Construct the full path to the processed output image
            processed_output_path = os.path.join(processed_folder, file_name)
            
            # Write the processed image to the processed folder
            cv.imwrite(processed_output_path, cl1)
            
            print(f"Processed image {i + 1}: {input_image_path} -> {grayscale_output_path} (Grayscale), {processed_output_path} (Processed)")
