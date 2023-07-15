use ndarray::Array;
use ndarray::{ArrayBase, Axis, Dim, OwnedRepr};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::ops::AddAssign;
use std::sync::Arc;
use std::vec;

use image;
use image::GenericImageView;

// K-Means Algorithm code taken from
// https://applied-math-coding.medium.com/data-science-implementing-k-means-in-rust-457e4af55ece

type Matrix = ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>;
type Cluster = Vec<Vec<usize>>;
type Centroids = Matrix;
type Features = Matrix;

fn assign_to_centroids(centroids: &Centroids, x: &Arc<Features>) -> (Cluster, f64) {
    let mut total_err = 0.0;
    let mut cluster = vec![vec![]; centroids.ncols()];

    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let (closet_centroid_idx, err) = centroids
            .axis_iter(Axis(1))
            .map(|centroid| {
                let d = &centroid - &row.t();
                (&d * &d).sum()
            })
            .enumerate()
            .min_by(|(_, v_1), (_, v_2)| v_1.total_cmp(v_2))
            .unwrap();

        cluster[closet_centroid_idx].push(i);
        total_err += err;
    }
    (cluster, total_err)
}

fn compute_centroids_from_cluster(cluster: &Cluster, x: &Arc<Features>) -> Centroids {
    let mut centroids = Array::zeros((x.ncols(), cluster.len()));
    for (cluster_idx, row_indexes) in cluster.iter().enumerate() {
        let n = f64::try_from(u32::try_from(row_indexes.len()).expect("overflows u32")).unwrap();

        if row_indexes.len() == 0 {
            centroids
                .column_mut(cluster_idx)
                .assign(&Array::random((x.ncols(), 1), Uniform::new(0.0, 1.0)).column(0));
        }

        for row_idx in row_indexes {
            centroids
                .column_mut(cluster_idx)
                .add_assign(&(1.0 / n * &x.row(*row_idx).t()));
        }
    }

    centroids
}

fn k_means(k: usize, max_iter: usize, x: &Arc<Features>) -> (Centroids, Cluster) {
    let mut centroids = Array::random((x.ncols(), k), Uniform::new(0.0, 1.0));

    let mut cluster = vec![vec![]; centroids.ncols()];
    let mut prev_total_err = f64::INFINITY;
    let mut total_err = 0.0;
    let mut iter = 0;

    while iter < max_iter && (iter == 0 || (prev_total_err - total_err) / total_err > 0.01) {
        prev_total_err = if iter > 0 { total_err } else { prev_total_err };
        (cluster, total_err) = assign_to_centroids(&centroids, &x);

        centroids = compute_centroids_from_cluster(&cluster, &x);
        iter += 1;
    }
    (centroids, cluster)
}
fn main() {
    let (filename, seg_filename, k_val) = parse_args();

    println!("Loaded Image");
    let img = image::open(filename).unwrap();

    // Use kmeans for color quantization
    let (width, height) = img.dimensions();
    let mut ex = Array::zeros((width as usize * height as usize, 3));

    print!("Started Processing!");
    for (x, y, pixel) in img.pixels() {
        let r = f64::from(pixel[0]);
        let g = f64::from(pixel[1]);
        let b = f64::from(pixel[2]);

        let i = (x * height + y) as usize;
        ex[[i, 0]] = r;
        ex[[i, 1]] = g;
        ex[[i, 2]] = b;
    }

    let x = Arc::new(ex);

    println!("Running Algorithm!");

    let (centroids, cluster) = k_means(k_val, 100, &x);

    println!("Finished Algorithm!");
    let mut new_img = image::ImageBuffer::new(width, height);

    for (cluster_idx, row_indexes) in cluster.iter().enumerate() {
        let r = centroids[[0, cluster_idx]] as u8;
        let g = centroids[[1, cluster_idx]] as u8;
        let b = centroids[[2, cluster_idx]] as u8;

        for row_idx in row_indexes {
            let x = *row_idx / height as usize;
            let y = *row_idx % height as usize;
            new_img.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
        }
    }

    new_img.save(seg_filename).unwrap();
}

fn parse_args() -> (String, String, usize) {
    let file = std::env::args().nth(1).unwrap_or("image.jpg".to_string());
    // let segmented_name = std::env::args()
    //     .nth(2)
    //     .unwrap_or("segmented_image.jpg".to_string());
    let segmented_name = std::path::Path::new(&file)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
        + "_segmented.jpg";
    let k_val = std::env::args()
        .nth(2)
        .unwrap_or("2".to_string())
        .parse::<usize>()
        .unwrap_or(2);
    (file, segmented_name, k_val)
}
