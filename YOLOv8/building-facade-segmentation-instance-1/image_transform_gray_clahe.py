import cv2 as cv
import os

ls = ['test','train','valid']

for x in ls:  
    # folder containing the original images
    input_folder = f'./YOLOv8/building-facade-segmentation-instance-1/{x}/images/'

    # folder to save the grayscale images
    grayscale_folder = f'./YOLOv8/building-facade-segmentation-instance-1/{x}_grayscale/images'

    # folder to save the processed images CLAHE (b/w) easy (cliplimit 2.0)
    processedCLAHEbw2_folder = f'./YOLOv8/building-facade-segmentation-instance-1/{x}_processedCLAHEbw2/images'
    
     # folder to save the processed images CLAHE (b/w) extrem (cliplimit 40.0)
    processedCLAHEbw40_folder = f'./YOLOv8/building-facade-segmentation-instance-1/{x}_processedCLAHEbw40/images'
    
    # folder to save the processed images CLAHE (color) extrem (cliplimit 40.0)
    processedCLAHEcol40_folder = f'./YOLOv8/building-facade-segmentation-instance-1/{x}_processedCLAHEcol40/images'
    
    

    # Create the output folders if they don't exist
    os.makedirs(grayscale_folder, exist_ok=True)
    os.makedirs(processedCLAHEbw2_folder, exist_ok=True)
    os.makedirs(processedCLAHEbw40_folder, exist_ok=True)
    os.makedirs(processedCLAHEcol40_folder, exist_ok=True)


    # Get a list of files in the input folder
    file_list = os.listdir(input_folder)

    # Process the first 5 files
    for i, file_name in enumerate(file_list):
        # Check if the file is a JPG image
        if file_name.endswith('.jpg'):
            # Construct the full path to the input image
            input_image_path = os.path.join(input_folder, file_name)
            
            # Read the input image in col
            img_col = cv.imread(input_image_path,cv.IMREAD_COLOR)
            
            # Read the input image in grayscale
            img = cv.imread(input_image_path, cv.IMREAD_GRAYSCALE)
            
            # Check if the image was successfully read
            assert img is not None, f"File {input_image_path} could not be read, check with os.path.exists()"
            
            # Construct the full path to the grayscale output image
            grayscale_output_path = os.path.join(grayscale_folder, file_name)
            
            # Write the grayscale image to the grayscale folder
            #cv.imwrite(grayscale_output_path, img)
            
            # Create a CLAHE (b/w) easy (cliplimit 2.0) object 
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(img)
            
            # Construct the full path to the processed output image
            processedCLAHEbw2_folder_output_path = os.path.join(processedCLAHEbw2_folder, file_name)
            
            # Write the processed image to the processed folder
            #cv.imwrite(processedCLAHEbw2_folder_output_path, cl1)
            
            print(f"Processed image {i + 1}: {input_image_path} -> {grayscale_output_path} (Grayscale), {processedCLAHEbw2_folder_output_path} (Processed)")

            # Create a CLAHE (b/w) extrem (cliplimit 40.0)
            clahe = cv.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(img)
            
            # Construct the full path to the processed output image
            processedCLAHEbw40_folder_output_path = os.path.join(processedCLAHEbw40_folder, file_name)
            
            # Write the processed image to the processed folder
            cv.imwrite(processedCLAHEbw40_folder_output_path, cl1)
            
            print(f"Processed image {i + 1}: {input_image_path} -> {grayscale_output_path} (Grayscale), {processedCLAHEbw40_folder_output_path} (Processed)")


            # Create a CLAHE (col) extrem (cliplimit 40.0): wh have to split the channels: 
            channels = cv.split(img_col)
            enhanced_channels = [clahe.apply(channel) for channel in channels]
            enhanced_image = cv.merge(enhanced_channels)
                        
            # Construct the full path to the processed output image
            processedCLAHEcol40_folder_output_path = os.path.join(processedCLAHEcol40_folder, file_name)
            
            # Write the processed image to the processed folder
            cv.imwrite(processedCLAHEcol40_folder_output_path, enhanced_image)
            
            print(f"Processed image {i + 1}: {input_image_path} -> {grayscale_output_path} (Grayscale), {processedCLAHEcol40_folder_output_path} (Processed)")
