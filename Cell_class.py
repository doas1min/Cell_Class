import cv2
import numpy as np
from skimage import measure, filters, feature
import matplotlib.pyplot as plt
import os
import pandas as pd
import joblib
from tqdm import tqdm


def extract_features(image):
    features = []
    # Histogram of Oriented Gradients
    features.extend(feature.hog(image, pixels_per_cell=(10, 10), cells_per_block=(2, 2)))
    # Mean, Std, Max, Min intensity
    features.extend([np.mean(image), np.std(image), np.max(image), np.min(image)])
    # Local Binary Pattern (uniform)
    features.extend(feature.local_binary_pattern(image, P=8, R=1, method='uniform').flatten())
    return features


def detect_and_classify_cells(image_path, model, scaler):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read the image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply multiple thresholds
    thresh1 = filters.threshold_otsu(blurred)
    binary1 = blurred < thresh1

    thresh2 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    binary2 = thresh2 > 0

    # Combine the two binary images
    binary = binary1 | binary2

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    labels = measure.label(binary)
    props = measure.regionprops(labels, intensity_image=gray)

    cell_types = {'N': 0, 'L': 0, 'M': 0}
    image_with_boxes = image.copy()

    color_map = {
        'N': (255, 0, 0),    # Red
        'L': (0, 0, 255),    # Blue
        'M': (255, 255, 0)   # Yellow
    }

    cell_data = []
    cell_number = 1

    for prop in props:
        if 20 < prop.area < 1000:
            minr, minc, maxr, maxc = prop.bbox
            cell_image = gray[minr:maxr, minc:maxc]
            cell_image = cv2.resize(cell_image, (50, 50))
            features = extract_features(cell_image)
            features_scaled = scaler.transform([features])
            cell_type = model.predict(features_scaled)[0]

            cell_types[cell_type] += 1

            # Draw bounding box for cell
            cv2.rectangle(image_with_boxes, (minc, minr), (maxc, maxr), (0, 255, 0), 1)

            # Display cell type and number
            color = color_map[cell_type]
            label_text = f"{cell_number}:{cell_type}"
            cv2.putText(
                image_with_boxes, label_text, (minc, minr - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
            )

            cell_data.append({
                'Cell Number': cell_number,
                'Cell Type': cell_type,
                'X': minc,
                'Y': minr,
                'Width': maxc - minc,
                'Height': maxr - minr,
                'Area': prop.area
            })

            cell_number += 1

    return image_with_boxes, cell_types, pd.DataFrame(cell_data)


def process_target2_images():
    print("Loading updated model and scaler...")
    try:
        model = joblib.load('./cell_classifier_model_updated.joblib')
        scaler = joblib.load('./cell_classifier_scaler_updated.joblib')
    except:
        print("Could not load updated model. Trying original model...")
        try:
            model = joblib.load('./cell_classifier_model.joblib')
            scaler = joblib.load('./cell_classifier_scaler.joblib')
        except:
            print("No models found. Please train a model first.")
            return None

    target_folder = './target2'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created {target_folder} directory.")
        return None

    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
    summary_data = []

    for filename in tqdm(os.listdir(target_folder)):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(target_folder, filename)

            try:
                result_image, cell_types, cell_df = detect_and_classify_cells(
                    image_path, model, scaler
                )

                total_cells = sum(cell_types.values())
                if total_cells > 0:
                    cell_percentages = {
                        cell_type: (count / total_cells) * 100
                        for cell_type, count in cell_types.items()
                    }
                else:
                    cell_percentages = {cell_type: 0.0 for cell_type in cell_types.keys()}

                base_path = os.path.splitext(image_path)[0]

                # Save result image
                image_result_path = base_path + '_result.png'
                cv2.imencode('.png', result_image)[1].tofile(image_result_path)

                # Generate and save plot
                plt.figure(figsize=(12, 12))
                plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                plt.title(
                    f"Detected Cells: {total_cells}\n"
                    f"N: {cell_types['N']} ({cell_percentages['N']:.1f}%)\n"
                    f"L: {cell_types['L']} ({cell_percentages['L']:.1f}%)\n"
                    f"M: {cell_types['M']} ({cell_percentages['M']:.1f}%)"
                )
                plt.axis('off')

                plot_path = base_path + '_plot.png'
                plt.savefig(plot_path, bbox_inches='tight', dpi=300)
                plt.close()

                # Save CSV
                csv_path = base_path + '_cell_data.csv'
                cell_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

                # Add summary data
                summary_data.append({
                    'Filename': filename,
                    'Total Cells': total_cells,
                    'N Count': cell_types['N'],
                    'L Count': cell_types['L'],
                    'M Count': cell_types['M'],
                    'N Percentage': cell_percentages['N'],
                    'L Percentage': cell_percentages['L'],
                    'M Percentage': cell_percentages['M']
                })

                print(f"\nProcessed {filename}:")
                print(f"Total cells: {total_cells}")
                for cell_type, count in cell_types.items():
                    print(f"{cell_type}: {count} ({cell_percentages[cell_type]:.1f}%)")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('./target2_analysis_summary.csv', index=False)
        print("\nAnalysis summary saved to: target2_analysis_summary.csv")
        return summary_df
    else:
        print("No images were processed successfully.")
        return None


if __name__ == "__main__":
    summary_df = process_target2_images()
    if summary_df is not None:
        print("\nProcessing completed successfully!")
        print("\nSummary of all processed images:")
        print(summary_df)
