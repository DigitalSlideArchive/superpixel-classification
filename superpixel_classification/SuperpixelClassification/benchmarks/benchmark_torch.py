''' Benchmark script for the SuperpixelClassificationTorch class
Originally written by feeding "tests/test_torch.py" to ChatGPT and asking for a benchmarking using timeit.
'''
import shutil
import numpy as np
import h5py
import os
import tempfile
import timeit
from unittest.mock import MagicMock
import csv
import matplotlib.pyplot as plt
from datetime import datetime

from IPython.utils.path import ensure_dir_exists
from more_itertools.more import side_effect
from superpixel_classification.SuperpixelClassification.SuperpixelClassificationBase import SuperpixelClassificationBase
from superpixel_classification.SuperpixelClassification.SuperpixelClassificationTorch import SuperpixelClassificationTorch
from superpixel_classification.SuperpixelClassification.progress_helper import ProgressHelper

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SuperpixelClassificationTorch.")
    parser.add_argument('--mnist-image-size', type=int, default=100, help='patchsize of individual images')
    parser.add_argument('--color-dim', type=int, default=3, help='Number of color channels')
    parser.add_argument('--image-sizes', default=list(map(int, [1e3, 1e4])), help='Output path for the pyramidal TIF file')
    parser.add_argument('--epochs', default=3, type=int, help='Number of epochs to train')
    parser.add_argument('--out-dir', default='benchmark_results', type=str, help='default output directory for benchmark results')

    return parser.parse_args()


def create_sample_data(num_images, tmpdir, image_size, color_dim):
    h5_path = os.path.join(tmpdir, "test_data.h5")
    images = np.random.randint(0, 255, size=(num_images, image_size, image_size, color_dim), dtype=np.uint8)

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('used_indices', data=np.arange(num_images - 2))

    return h5_path

def train_model(num_images, num_epochs, h5_path):
    base: SuperpixelClassificationBase = SuperpixelClassificationTorch()
    base.feature_is_image = True
    base.certainty = 'batchbald'

    # Mock girder client
    gc = MagicMock()
    def mv_to_dst(_, dst):
        return shutil.copy(h5_path, dst)
    gc.downloadFile = MagicMock(side_effect=mv_to_dst)
    def mv_to_src(_, src):
        dst = os.path.dirname(os.path.dirname(h5_path))
        return shutil.copy(src, dst)
    gc.uploadFileToFolder = MagicMock(side_effect=mv_to_src, return_value=True)

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elem = {
        'girderId': 'test_girder_id',
        'categories': [
            {"label": c} for c in labels
        ],
        'values':
            [] \
            + np.random.randint(1, len(labels) - 1, size=(num_images - 2), dtype=np.uint8).tolist()
            + [0, 0],  # last two images unlabeled
        'transform': {'matrix': [[1.0]]}
    }

    item = {'_id': 'test_h5_file', 'name': 'test'}
    annotrec = {'_id': '1', '_version': 0, 'annotation': {'name': 'TorchTest'}}
    items = [(item, annotrec, elem)]

    with ProgressHelper('Superpixel Classification', 'Test training', True) as prog:
        prog.progress(0)
        prog.items(items)
        modelFile, modelTrainingFile = base.trainModel(
            gc=gc,
            annotationName="TorchTest",
            itemsAndAnnot=items,
            features={'test_h5_file': {'_id': 'feature_id', 'name': 'test_h5_file'}},
            modelFolderId="test_folder_id",
            batchSize=4,
            epochs=1,
            trainingSplit=0.5,
            randomInput=False,
            labelList='',
            excludeLabelList=[],
            prog=prog,
            use_cuda=True,
        )

    return modelFile, modelTrainingFile

def create_benchmark_plot(results, out_dir):
    plt.figure(figsize=(12, 6))
    
    # Number of image sizes and runs
    n_sizes = len(results)
    n_runs = len(results[0]['times'])
    
    # Create positions for bars
    ind = np.arange(n_sizes)
    width = 0.25  # Width of bars
    
    # Plot bars for each run
    for i in range(n_runs):
        times = [result['times'][i] for result in results]
        plt.bar(ind + i*width, times, width, label=f'Run {i+1}')
    
    plt.xlabel('Number of Images')
    plt.ylabel('Time (seconds)')
    plt.title('Model Training Benchmark Times')
    
    # Set x-axis labels
    plt.xticks(ind + width, [str(result['num_images']) for result in results])
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dst_pth = os.path.join(out_dir, f'benchmark_results_{timestamp}.png')
    plt.savefig(dst_pth)
    plt.close()

    return dst_pth

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args = parse_args()
    ensure_dir_exists(args.out_dir)
    csv_filename = os.path.join(args.out_dir, f'benchmark_results_{timestamp}.csv')
    results = []

    # Write CSV header
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Num Images', 'Run 1', 'Run 2', 'Run 3', 'Average', 'Best'])

    for num_images in args.image_sizes:
        print(f"\nBenchmarking with NUM_IMAGES = {num_images}")
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = create_sample_data(num_images, tmpdir, args.mnist_image_size, args.color_dim)
            timer = timeit.Timer(lambda: train_model(num_images, args.epochs, h5_path))
        
            try:
                times = timer.repeat(repeat=3, number=1)
                avg_time = sum(times) / len(times)
                best_time = min(times)

                # Store results for plotting
                results.append({
                    'num_images': num_images,
                    'times': times,
                    'average': avg_time,
                    'best': best_time
                })

                # Write results to CSV
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        num_images,
                        round(times[0], 3),
                        round(times[1], 3),
                        round(times[2], 3),
                        round(avg_time, 3),
                        round(best_time, 3)
                    ])

                print(f"Times for each run (seconds): {[round(t, 3) for t in times]}")
                print(f"Average time (seconds): {round(avg_time, 3)}")
                print(f"Best time (seconds): {round(best_time, 3)}")

            except Exception as e:
                print(f"Error during benchmark: {str(e)}")
                # Write error to CSV
                with open(csv_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([num_images, f"Error: {str(e)}", "", "", "", ""])
            finally:
                shutil.rmtree(tmpdir)

    # Create and save the plot
    out_file = create_benchmark_plot(results, args.out_dir)
    print(f"\nResults saved to {csv_filename}")
    print(f"Plot saved as {out_file}")

if __name__ == "__main__":
    main()