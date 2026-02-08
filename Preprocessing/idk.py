import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hessian_matrix
from skimage.measure import shannon_entropy, regionprops, label
from skimage.filters import frangi, gabor_kernel, threshold_otsu, gaussian
from skimage.morphology import disk, opening, closing
import pywt
import mahotas
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
from multiprocessing import Pool, cpu_count
import gc
import time

input_folder = Path(r'D:\AI IN DIF')
output_csv = 'dif_features.csv'
IMG_SIZE = (224, 224)
MAX_IMAGE_SIZE = 800

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
    def forward(self, x):
        return self.model(x)

preprocess = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def extract_deposition_pattern_features(img_binary, img_gray):
    features = {}
    try:
        img_binary = img_binary.astype(bool)
        kernel_small = disk(3)
        opened = opening(img_binary, kernel_small)
        closed = closing(img_binary, kernel_small)
        features['pattern_continuity'] = np.sum(opened) / (np.sum(img_binary) + 1e-6)
        features['pattern_fragmentation'] = np.sum(np.logical_xor(closed, img_binary)) / (np.sum(img_binary) + 1e-6)
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        features['linear_pattern_count'] = len(lines) if lines is not None else 0
        if lines is not None and len(lines) > 0:
            lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) for line in lines for x1,y1,x2,y2 in line]
            features['linear_pattern_avg_length'] = np.mean(lengths)
            features['linear_pattern_max_length'] = np.max(lengths)
        else:
            features['linear_pattern_avg_length'] = 0
            features['linear_pattern_max_length'] = 0
        labeled = label(img_binary.astype(np.uint8))
        regions = regionprops(labeled)
        if len(regions) > 0:
            areas = [r.area for r in regions]
            eccentricities = [r.eccentricity for r in regions]
            solidity = [r.solidity for r in regions]
            features['num_components'] = len(regions)
            features['avg_component_area'] = np.mean(areas)
            features['std_component_area'] = np.std(areas)
            features['avg_eccentricity'] = np.mean(eccentricities)
            features['avg_solidity'] = np.mean(solidity)
            features['component_area_skew'] = skew(areas)
        else:
            features.update({'num_components': 0, 'avg_component_area': 0, 'std_component_area': 0, 'avg_eccentricity': 0, 'avg_solidity': 0, 'component_area_skew': 0})
    except Exception as e:
        features.update({'pattern_continuity': 0, 'pattern_fragmentation': 0, 'linear_pattern_count': 0, 'linear_pattern_avg_length': 0, 'linear_pattern_max_length': 0, 'num_components': 0, 'avg_component_area': 0, 'std_component_area': 0, 'avg_eccentricity': 0, 'avg_solidity': 0, 'component_area_skew': 0})
    return features

def extract_advanced_texture_features(img_gray):
    features = {}
    try:
        for radius in [1, 2, 3, 5]:
            n_points = 8 * radius
            lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            features[f'lbp_r{radius}_mean'] = np.mean(hist)
            features[f'lbp_r{radius}_var'] = np.var(hist)
            features[f'lbp_r{radius}_energy'] = np.sum(hist**2)
            features[f'lbp_r{radius}_entropy'] = shannon_entropy(lbp)
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        img_quantized = (img_gray / 4).astype(np.uint8)
        glcm = graycomatrix(img_quantized, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in props:
            values = graycoprops(glcm, prop).ravel()
            features[f'glcm_{prop}_mean'] = np.mean(values)
            features[f'glcm_{prop}_std'] = np.std(values)
            features[f'glcm_{prop}_range'] = np.ptp(values)
        haralick_feats = mahotas.features.haralick(img_quantized, ignore_zeros=True, return_mean_ptp=True)
        for i, val in enumerate(haralick_feats):
            features[f'haralick_{i}'] = val
        zernike_feats = mahotas.features.zernike_moments(img_gray, radius=30)
        for i, val in enumerate(zernike_feats):
            features[f'zernike_{i}'] = val
        coeffs2 = pywt.dwt2(img_gray, 'haar')
        cA, (cH, cV, cD) = coeffs2
        for coeff_name, coeff in [('approx', cA), ('horiz', cH), ('vert', cV), ('diag', cD)]:
            features[f'wavelet_{coeff_name}_energy'] = np.sum(coeff**2)
            features[f'wavelet_{coeff_name}_mean'] = np.mean(np.abs(coeff))
            features[f'wavelet_{coeff_name}_std'] = np.std(coeff)
            features[f'wavelet_{coeff_name}_entropy'] = shannon_entropy(np.abs(coeff))
        gabor_features = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for frequency in [0.1, 0.2, 0.3]:
                kernel = gabor_kernel(frequency, theta=theta)
                filtered = ndimage.convolve(img_gray, np.real(kernel), mode='wrap')
                gabor_features.extend([np.mean(filtered), np.std(filtered), np.mean(np.abs(filtered))])
        for i, val in enumerate(gabor_features):
            features[f'gabor_{i}'] = val
        features['fractal_dimension'] = calculate_fractal_dimension(img_gray)
    except Exception as e:
        for radius in [1, 2, 3, 5]:
            features.update({f'lbp_r{radius}_mean': 0, f'lbp_r{radius}_var': 0, f'lbp_r{radius}_energy': 0, f'lbp_r{radius}_entropy': 0})
    return features

def extract_vascular_features(img_gray):
    features = {}
    try:
        frangi_response = frangi(img_gray, sigmas=range(1, 8, 3), black_ridges=False)
        features['frangi_mean'] = np.mean(frangi_response)
        features['frangi_max'] = np.max(frangi_response)
        features['frangi_std'] = np.std(frangi_response)
        features['frangi_coverage'] = np.sum(frangi_response > 0.1) / frangi_response.size
        del frangi_response
        gc.collect()
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        features['frangi_mean'] = 0
        features['frangi_max'] = 0
        features['frangi_std'] = 0
        features['frangi_coverage'] = 0
    try:
        Hxx, Hxy, Hyy = hessian_matrix(img_gray, sigma=3, order='rc')
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy * Hxy
        discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
        lambda1 = (trace + discriminant) / 2
        lambda2 = (trace - discriminant) / 2
        features['hessian_lambda1_mean'] = np.mean(lambda1)
        features['hessian_lambda2_mean'] = np.mean(lambda2)
        features['hessian_ratio'] = np.mean(np.abs(lambda1) / (np.abs(lambda2) + 1e-6))
        ridge_strength = np.maximum(0, lambda1) * (lambda2 < 0)
        features['ridge_strength_mean'] = np.mean(ridge_strength)
        features['ridge_strength_max'] = np.max(ridge_strength)
        del Hxx, Hxy, Hyy, trace, det, discriminant, lambda1, lambda2, ridge_strength
        gc.collect()    
    except (MemoryError, np.core._exceptions._ArrayMemoryError):
        features.update({'hessian_lambda1_mean': 0, 'hessian_lambda2_mean': 0, 'hessian_ratio': 0, 'ridge_strength_mean': 0, 'ridge_strength_max': 0})
    return features

def extract_spatial_features(img_binary):
    features = {}
    if img_binary.dtype == bool:
        img_binary = img_binary.astype(np.uint8)
    vertical_profile = np.sum(img_binary, axis=1)
    if np.sum(vertical_profile) > 0:
        vertical_profile_norm = vertical_profile / np.sum(vertical_profile)
        positions = np.arange(len(vertical_profile))
        features['vertical_center_of_mass'] = np.sum(vertical_profile_norm * positions) / len(vertical_profile)
        mean_pos = np.sum(vertical_profile_norm * positions)
        variance = np.sum(vertical_profile_norm * (positions - mean_pos)**2)
        features['vertical_spread'] = np.sqrt(variance)
        features['vertical_skewness'] = skew(vertical_profile)
    else:
        features.update({'vertical_center_of_mass': 0, 'vertical_spread': 0, 'vertical_skewness': 0})
    horizontal_profile = np.sum(img_binary, axis=0)
    if np.sum(horizontal_profile) > 0:
        horizontal_profile_norm = horizontal_profile / np.sum(horizontal_profile)
        features['horizontal_uniformity'] = shannon_entropy(horizontal_profile_norm)
    else:
        features['horizontal_uniformity'] = 0
    center_y, center_x = np.array(img_binary.shape) // 2
    Y, X = np.ogrid[:img_binary.shape[0], :img_binary.shape[1]]
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    if np.sum(img_binary) > 0:
        avg_distance = np.sum(dist_from_center * img_binary) / np.sum(img_binary)
        features['centrality'] = 1.0 / (1.0 + avg_distance)
    else:
        features['centrality'] = 0
    return features

def extract_intensity_features(img_gray):
    features = {}
    features['intensity_mean'] = np.mean(img_gray)
    features['intensity_std'] = np.std(img_gray)
    features['intensity_median'] = np.median(img_gray)
    features['intensity_max'] = np.max(img_gray)
    features['intensity_min'] = np.min(img_gray)
    features['intensity_range'] = features['intensity_max'] - features['intensity_min']
    features['intensity_skewness'] = skew(img_gray.ravel())
    features['intensity_kurtosis'] = kurtosis(img_gray.ravel())
    hist, _ = np.histogram(img_gray, bins=64, range=(0, 256), density=True)
    features['intensity_entropy'] = shannon_entropy(img_gray)
    features['intensity_uniformity'] = np.sum(hist**2)
    peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0]
    features['intensity_num_peaks'] = len(peaks)
    return features

def extract_edge_features(img_gray):
    features = {}
    edges = cv2.Canny(img_gray, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    features['edge_magnitude_mean'] = np.mean(edge_magnitude)
    features['edge_magnitude_std'] = np.std(edge_magnitude)
    features['edge_magnitude_max'] = np.max(edge_magnitude)
    edge_orientation = np.arctan2(sobely, sobelx)
    hist_orient, _ = np.histogram(edge_orientation.ravel(), bins=36, range=(-np.pi, np.pi), density=True)
    features['edge_orientation_entropy'] = -np.sum(hist_orient * np.log(hist_orient + 1e-10))
    return features

def calculate_fractal_dimension(img):
    try:
        threshold = threshold_otsu(img)
        binary = img > threshold
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1)
            return len(np.where((S > 0) & (S < k*k))[0])
        p = min(binary.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        n = int(np.log(n)/np.log(2))
        sizes = 2**np.arange(n, 1, -1)
        counts = []
        for size in sizes:
            counts.append(boxcount(binary, size))
        if len(counts) < 2:
            return 0
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]
    except:
        return 0

def preprocess_image_cpu(img_bgr):
    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_green)
    img_smooth = gaussian(img_enhanced, sigma=1)
    img_smooth = (img_smooth * 255).astype(np.uint8)
    img_binary = cv2.adaptiveThreshold(img_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) > 0
    return img_enhanced, img_smooth, img_binary

def extract_cpu_features(args):
    img_path, label = args
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        img_enhanced, img_smooth, img_binary = preprocess_image_cpu(img_bgr)
        features = {}
        features.update(extract_intensity_features(img_enhanced))
        features.update(extract_advanced_texture_features(img_smooth))
        features.update(extract_deposition_pattern_features(img_binary, img_enhanced))
        features.update(extract_vascular_features(img_smooth))
        features.update(extract_spatial_features(img_binary))
        features.update(extract_edge_features(img_enhanced))
        return {'filename': img_path.name, 'label': label, 'features': features, 'img_bgr': img_bgr}
    except Exception as e:
        print(f"Error in {img_path.name}: {e}")
        return None

def collect_image_paths(root_path):
    image_list = []
    for class_dir in root_path.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            for item in class_dir.iterdir():
                if item.suffix.lower() in ['.jpg', '.png', '.tif', '.jpeg', '.bmp']:
                    image_list.append((item, label))
    return image_list

def process_dataset_parallel(root_path, feature_extractor, device, batch_size, num_workers):
    @torch.no_grad()
    
    def extract_deep_features_batch(img_batch):
        batch_tensors = []
        for img_bgr in img_batch:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = preprocess(pil_img)
            batch_tensors.append(tensor)
        batch_tensor = torch.stack(batch_tensors).to(device)
        features_batch = feature_extractor(batch_tensor)
        return features_batch.cpu().numpy()

    image_paths = collect_image_paths(root_path)
    total_images = len(image_paths)
    first_batch = True
    with Pool(num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap(extract_cpu_features, image_paths), 1):
            if result is not None:
                results.append(result)
            if len(results) >= batch_size:
                print(f"{i}/{total_images}")
                img_batch = [r['img_bgr'] for r in results]
                deep_features_batch = extract_deep_features_batch(img_batch)
                batch_data = []
                for j, result in enumerate(results):
                    row_data = {'filename': result['filename'], 'label': result['label']}
                    row_data.update(result['features'])
                    for k, val in enumerate(deep_features_batch[j]):
                        row_data[f'deep_{k}'] = val
                    batch_data.append(row_data)
                df_batch = pd.DataFrame(batch_data)
                if first_batch:
                    df_batch.to_csv(output_csv, mode='w', index=False)
                    first_batch = False
                else:
                    df_batch.to_csv(output_csv, mode='a', header=False, index=False)
                del img_batch, deep_features_batch, batch_data, df_batch
                results = []
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"{i}/{total_images} ({i/total_images*100:.1f}%)")
    if results:
        img_batch = [r['img_bgr'] for r in results]
        deep_features_batch = extract_deep_features_batch(img_batch)
        batch_data = []
        for j, result in enumerate(results):
            row_data = {'filename': result['filename'], 'label': result['label']}
            row_data.update(result['features'])
            for k, val in enumerate(deep_features_batch[j]):
                row_data[f'deep_{k}'] = val
            batch_data.append(row_data)
        df_batch = pd.DataFrame(batch_data)
        df_batch.to_csv(output_csv, mode='a', header=first_batch, index=False)
        del img_batch, deep_features_batch, batch_data, df_batch
        gc.collect()
    return pd.read_csv(output_csv)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32 if torch.cuda.is_available() else 8
    NUM_WORKERS = max(1, cpu_count() - 2)
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()
    start_time = time.time()
    df_features = process_dataset_parallel(input_folder, feature_extractor, device, BATCH_SIZE, NUM_WORKERS)
    df_features.to_csv(output_csv, index=False)
    elapsed = time.time() - start_time
    print(f"Output: {output_csv}")
    print(f"\nClasses:\n{df_features['label'].value_counts()}")