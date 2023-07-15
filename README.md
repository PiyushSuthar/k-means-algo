# K-Means Algorithm

Implementation of Image segmentation using K-Means algorithm in rust.

> Code mostly taken from [here](https://applied-math-coding.medium.com/data-science-implementing-k-means-in-rust-457e4af55ece)

## Usage

```bash
cargo run --release -- <input_image> <output_image> <k>
```

## Example

```bash
cargo run --release -- images/lena.png images/lena_out.png 5
```

<!-- Table showing images in example folder -->

_Input_ and _Output_ images for different values of `k`:

| Input                    | K = 2                    | K = 5                    | K = 7                    |
| ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| ![Me](examples/real.jpg) | ![K_2](examples/k_2.png) | ![K_5](examples/k_5.jpg) | ![K_7](examples/k_7.jpg) |
