from PIL import Image
from pathlib import Path
import numpy as np
import multiprocessing
from multiprocessing import Pool


import argparse

def resize_img(img, max_dim=512):
  """Resizes largest dim to max_dim but keeps aspect ratio."""

  dims = img.size
  larger_dim_idx = np.argmax(dims)
  smaller_dim_idx = np.abs(larger_dim_idx - 1)

  downscale_factor = max_dim / dims[larger_dim_idx]
  smaller_dim = int(dims[smaller_dim_idx] * downscale_factor)

  rescale_dims = [0, 0]
  rescale_dims[larger_dim_idx] = max_dim
  rescale_dims[smaller_dim_idx] = smaller_dim

  return img.resize(size=rescale_dims)


def preprocess_img(img_path, out_folder, square_dim=512):
  try:
    img = Image.open(img_path)
    img = resize_img(img, max_dim=square_dim)
    img.save(out_folder / img_path.name)
  except Exception as error:
    print(f'{img_path.name} Error: {error}')


def main(args):
  in_path = args.in_path
  out_path = in_path + '_preprocessed' if args.out_path is None else args.out_path

  img_folder = Path(in_path)
  out_folder = Path(out_path)
  out_folder.mkdir(parents=True, exist_ok=True)

  img_paths = list(img_folder.glob('*.jpg'))
  assert len(img_paths) > 0,\
    f'No images found in {in_path}. Please make sure you specified the right folder'

  # n_cores -1: use all cores except one
  n_cores = (multiprocessing.cpu_count() - 1) \
            if args.n_cores < 0 else args.n_cores

  print(f'Going to convert {len(img_paths)} images from {args.in_path}'
        f' and will write them to {out_path}.')

  # ~0.7s/img * n_images / cores / 60 to get minutes
  print(f'Using {n_cores} cores. Expected time is roughly: '
        f'{0.7 * len(img_paths) / n_cores / 60:.1f} minutes.')

  pool = Pool(n_cores)
  pool.starmap(preprocess_img, [
      (path, out_folder, args.square_dim) for path in img_paths
  ])
  pool.close()

  print(f'Finished converting {len(img_paths)} images from {args.in_path}'
        f' and wrote them to {out_path}.')


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--in_path',
                      type=str,
                      default='PATH/TO/RAW/DATA/img/img_little2')
  parser.add_argument('--out_path', type=str,
                      default='PATH/TO/PREPROCESSED/DATA')
  parser.add_argument('--n_cores', default=-1, type=int)
  parser.add_argument('--square_dim', default=512, type=int)

  main(args=parser.parse_args())
