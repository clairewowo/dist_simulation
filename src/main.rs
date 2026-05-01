use hdf5::File;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::time::Instant;
use clap::{Arg, ArgAction, Command};
use rayon::prelude::*;

#[derive(Clone, Debug)]
struct Config {
    n_train: usize,
    n_test: usize,
    dim: usize,
    n_clusters: usize,
    n_superclusters: usize,
    gt_k: usize,
    center_radius: f32,
    supercluster_sigma: f32,
    local_sigma: f32,
    output: String,
    seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_train: 70_000,
            n_test: 10_000,
            dim: 768,
            n_clusters: 256,
            n_superclusters: 24,
            gt_k: 100,
            center_radius: 5.0,
            supercluster_sigma: 3.5,
            local_sigma: 0.85,
            output: "skewed-70000-euclidean.hdf5".to_string(),
            seed: 42,
        }
    }
}

impl Config {
    pub fn new(n_clusters: usize, n_superclusters: usize, seed: u64, output: &String, 
    center_rad: f32, local_sig: f32, super_sigma: f32) -> Self {
        let mut output_file = output.clone();
        if !output.ends_with(".hdf5") {
            output_file += ".hdf5";
        }
        
        Self {
            n_train: 70_000,
            n_test: 10_000,
            dim: 768,
            n_clusters: n_clusters,
            n_superclusters: n_superclusters,
            gt_k: 100,
            center_radius: center_rad,
            supercluster_sigma: super_sigma,
            local_sigma: local_sig,
            output: output_file,
            seed: seed,
        }
    }
    pub fn new_x_skew(x_skew: f32, n_clusters: usize, n_superclusters: usize, seed: u64, output: &String) -> Self {
        let mut output_file = output.clone();
        if !output.ends_with(".hdf5") {
            output_file += ".hdf5";
        }
        Self {
            n_train: 70_000,
            n_test: 10_000,
            dim: 768,
            n_clusters: n_clusters,
            n_superclusters: n_superclusters,
            gt_k: 100,
            center_radius: 5.0 + x_skew,
            supercluster_sigma: 1.0 + 0.2 * x_skew,
            local_sigma: 0.85 - 0.1 * x_skew,
            output: output_file,
            seed: seed,
        }
        
    }
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

fn flatten_rows(rows: &[Vec<f32>]) -> Vec<f32> {
    let total: usize = rows.iter().map(|r| r.len()).sum();
    let mut out = Vec::with_capacity(total);
    for r in rows {
        out.extend_from_slice(r);
    }
    out
}

fn flatten_rows_i32(rows: &[Vec<i32>]) -> Vec<i32> {
    let total: usize = rows.iter().map(|r| r.len()).sum();
    let mut out = Vec::with_capacity(total);
    for r in rows {
        out.extend_from_slice(r);
    }
    out
}

fn assign_cluster_sizes(total: usize, n_clusters: usize, even: &bool) -> Vec<usize> {
    // uneven cluster sizes
    if !*even { 
        // Generate random-ish weights using cluster index as seed
        let mut sizes = Vec::with_capacity(n_clusters);
        let mut remaining = total;

        for i in 0..n_clusters {
            let size = if i == n_clusters - 1 {
                remaining // last cluster gets whatever is left
            } else {
                remaining / 2
            };
            sizes.push(size);
            remaining -= size;
        }
        sizes
    }
    else {
        let base = total / n_clusters;
        let rem = total % n_clusters;
        let mut sizes = vec![base; n_clusters];
        for i in 0..rem {
            sizes[i] += 1;
        }
        sizes
    }
    
}

fn sample_unit_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    let mut v = vec![0.0f32; dim];
    let mut norm2 = 0.0f32;
    for x in &mut v {
        *x = normal.sample(rng);
        norm2 += *x * *x;
    }
    let norm = norm2.sqrt().max(1e-12);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn sample_gaussian_point(center: &[f32], sigma: f32, rng: &mut StdRng) -> Vec<f32> {
    let normal = Normal::<f32>::new(0.0, sigma).unwrap();
    center.iter().map(|&c| c + normal.sample(rng)).collect()
}

fn build_cluster_centers(cfg: &Config, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let per_super = cfg.n_clusters.div_ceil(cfg.n_superclusters);

    let mut super_centers = Vec::with_capacity(cfg.n_superclusters);
    for _ in 0..cfg.n_superclusters {
        let dir = sample_unit_vector(cfg.dim, rng);
        let center: Vec<f32> = dir.into_iter().map(|x| x * cfg.center_radius).collect();
        super_centers.push(center);
    }

    let mut cluster_centers = Vec::with_capacity(cfg.n_clusters);
    for sc in 0..cfg.n_superclusters {
        for _ in 0..per_super {
            if cluster_centers.len() >= cfg.n_clusters {
                break;
            }
            let center = sample_gaussian_point(&super_centers[sc], cfg.supercluster_sigma, rng);
            cluster_centers.push(center);
        }
    }
    cluster_centers
}

fn test_dataset(
    n_points: usize,
    cfg: &Config,
    cluster_centers: &[Vec<f32>], // only one supercluster's centers
    rng: &mut StdRng,
    even: &bool,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    //let cluster_sizes = assign_cluster_sizes(n_points, cfg.n_clusters, even);

    let mut vectors = Vec::with_capacity(n_points);
    let mut labels = Vec::with_capacity(n_points);

    for _ in 0..n_points {
        vectors.push(sample_gaussian_point(
            &cluster_centers[0],
            cfg.local_sigma,
            rng,
        ));
        labels.push(0);
    }
    

    let mut perm: Vec<usize> = (0..n_points).collect();
    perm.shuffle(rng);

    let shuffled_vectors: Vec<Vec<f32>> = perm.iter().map(|&i| vectors[i].clone()).collect();
    let shuffled_labels: Vec<usize> = perm.iter().map(|&i| labels[i]).collect();

    (shuffled_vectors, shuffled_labels)
}

fn train_dataset(
    n_points: usize,
    cfg: &Config,
    cluster_centers: &[Vec<f32>],
    rng: &mut StdRng,
    even: &bool,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let cluster_sizes = assign_cluster_sizes(n_points, cfg.n_clusters, even);

    let mut vectors = Vec::with_capacity(n_points);
    let mut labels = Vec::with_capacity(n_points);

    for (cid, &sz) in cluster_sizes.iter().enumerate() {
        for _ in 0..sz {
            vectors.push(sample_gaussian_point(
                &cluster_centers[cid],
                cfg.local_sigma,
                rng,
            ));
            labels.push(cid);
        }
    }

    let mut perm: Vec<usize> = (0..n_points).collect();
    perm.shuffle(rng);

    let shuffled_vectors: Vec<Vec<f32>> = perm.iter().map(|&i| vectors[i].clone()).collect();
    let shuffled_labels: Vec<usize> = perm.iter().map(|&i| labels[i]).collect();

    (shuffled_vectors, shuffled_labels)
}



fn topk_truth(train: &[Vec<f32>], test: &[Vec<f32>], k: usize) -> (Vec<Vec<i32>>, Vec<Vec<f32>>) {
    let results: Vec<(Vec<i32>, Vec<f32>)> = test
        .par_iter()
        .map(|q| {
            let mut pairs: Vec<(usize, f32)> = train
                .iter()
                .enumerate()
                .map(|(i, x)| (i, l2(q, x)))
                .collect();

            pairs.sort_by(|a, b| a.1.total_cmp(&b.1));
            pairs.truncate(k);

            let neigh: Vec<i32> = pairs.iter().map(|(i, _)| *i as i32).collect();
            let dist: Vec<f32> = pairs.iter().map(|(_, d)| *d).collect();

            (neigh, dist)
        })
        .collect();

    // unzip into two Vec<Vec<_>>
    let (all_neighbors, all_distances): (Vec<_>, Vec<_>) =
        results.into_iter().unzip();

    (all_neighbors, all_distances)
}

fn summarize_pair_distribution(
    train: &[Vec<f32>],
    labels: &[usize],
    rng: &mut StdRng,
    n_samples: usize,
) {
    let mut all = Vec::with_capacity(n_samples);
    let mut same = Vec::new();
    let mut diff = Vec::new();

    for _ in 0..n_samples {
        let i = rng.gen_range(0..train.len());
        let j = rng.gen_range(0..train.len());
        if i == j {
            continue;
        }
        let d = l2(&train[i], &train[j]);
        all.push(d);
        if labels[i] == labels[j] {
            same.push(d);
        } else {
            diff.push(d);
        }
    }

    fn quantiles(mut xs: Vec<f32>) -> Option<(f32, f32, f32, f32, f32)> {
        if xs.is_empty() {
            return None;
        }
        xs.sort_by(|a, b| a.total_cmp(b));
        let n = xs.len();
        let q = |p: f32| -> f32 {
            let idx = ((n - 1) as f32 * p).round() as usize;
            xs[idx]
        };
        Some((q(0.01), q(0.10), q(0.50), q(0.90), q(0.99)))
    }

    if let Some((p01, p10, p50, p90, p99)) = quantiles(all.clone()) {
        println!(
            "\nDistance diagnostics (Euclidean)\n  all      p01={:.4} p10={:.4} p50={:.4} p90={:.4} p99={:.4}",
            p01, p10, p50, p90, p99
        );
    }
    if let Some((p01, p10, p50, p90, p99)) = quantiles(same) {
        println!(
            "  same-cl  p01={:.4} p10={:.4} p50={:.4} p90={:.4} p99={:.4}",
            p01, p10, p50, p90, p99
        );
    }
    if let Some((p01, p10, p50, p90, p99)) = quantiles(diff) {
        println!(
            "  diff-cl  p01={:.4} p10={:.4} p50={:.4} p90={:.4} p99={:.4}",
            p01, p10, p50, p90, p99
        );
    }

    let mut sorted = all;
    sorted.sort_by(|a, b| a.total_cmp(b));
    if !sorted.is_empty() {
        let p05 = sorted[((sorted.len() - 1) as f32 * 0.05).round() as usize];
        let p95 = sorted[((sorted.len() - 1) as f32 * 0.95).round() as usize];

        let below = sorted.iter().filter(|&&x| x < p05).count();
        let middle = sorted.iter().filter(|&&x| x >= p05 && x <= p95).count();
        let above = sorted.iter().filter(|&&x| x > p95).count();

        println!(
            "  crude skew bins using [p05, p95]: below={} middle={} above={}",
            below, middle, above
        );
    }
}

fn write_hdf5(
    path: String,
    train: &[Vec<f32>],
    test: &[Vec<f32>],
    neighbors: &[Vec<i32>],
    distances: &[Vec<f32>],
) -> hdf5::Result<()> {
    let file = File::create(path)?;

    let train_flat = flatten_rows(train);
    let test_flat = flatten_rows(test);
    let neighbors_flat = flatten_rows_i32(neighbors);
    let distances_flat = flatten_rows(distances);

    let train_rows = train.len();
    let train_dim = if train.is_empty() { 0 } else { train[0].len() };

    let test_rows = test.len();
    let test_dim = if test.is_empty() { 0 } else { test[0].len() };

    let neigh_rows = neighbors.len();
    let neigh_dim = if neighbors.is_empty() { 0 } else { neighbors[0].len() };

    let dist_rows = distances.len();
    let dist_dim = if distances.is_empty() { 0 } else { distances[0].len() };

    let ds_train = file
        .new_dataset::<f32>()
        .shape((train_rows, train_dim))
        .create("train")?;
    ds_train.write_raw(&train_flat)?;

    let ds_test = file
        .new_dataset::<f32>()
        .shape((test_rows, test_dim))
        .create("test")?;
    ds_test.write_raw(&test_flat)?;

    let ds_neighbors = file
        .new_dataset::<i32>()
        .shape((neigh_rows, neigh_dim))
        .create("neighbors")?;
    ds_neighbors.write_raw(&neighbors_flat)?;

    let ds_distances = file
        .new_dataset::<f32>()
        .shape((dist_rows, dist_dim))
        .create("distances")?;
    ds_distances.write_raw(&distances_flat)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("Distance simulation")
        .arg( // parameter to control skewness. larger x = more skew. -5 > x > 8
            Arg::new("skew")
            .long("skew")
            .help("Controls skewness: central radius = 5 + x, local_sigma = 0.85 - 0.1x, supercluster_sigma = 0.5 + 0.1x")
            .required(false)
            .value_parser(clap::value_parser!(f32))
            .action(ArgAction::Set)
        )
        .arg(
            Arg::new("n_clusters")
            .long("n_clusters")
            .required(false)
            .default_value("6")
            .value_parser(clap::value_parser!(usize))
            .action(ArgAction::Set)
        )
        .arg(
            Arg::new("n_superclusters")
            .long("n_superclusters")
            .required(false)
            .default_value("3")
            .value_parser(clap::value_parser!(usize))
            .action(ArgAction::Set)
        )
        .arg(
            Arg::new("output")
            .short('o')
            .long("output")
            .required(false)
            .default_value("skewed-70000-euclidean.hdf5")
            .action(ArgAction::Set)
        )
        .arg(
            Arg::new("center_radius")
            .long("center_radius")
            .required(false)
            .default_value("5.0")
            .value_parser(clap::value_parser!(f32))
            .action(ArgAction::Set)
        )
        .arg(
            Arg::new("local_sigma")
            .long("local_sigma")
            .required(false)
            .default_value("0.85")
            .value_parser(clap::value_parser!(f32))
            .action(ArgAction::Set)
        )
        .arg(
            Arg::new("supercluster_sigma")
            .long("supercluster_sigma")
            .required(false)
            .default_value("0.4")
            .value_parser(clap::value_parser!(f32))
            .action(ArgAction::Set)
        )
        .arg( // controls whether cluster sizes are even
            Arg::new("even")
            .long("even")
            .required(false)
            .default_value("true")
            .value_parser(clap::value_parser!(bool))
            .action(ArgAction::Set)
        )
        .arg(
            Arg::new("seed")
            .short('s')
            .long("seed")
            .help("Random seed")
            .required(false)
            .value_parser(clap::value_parser!(u64))
            .default_value("42")
            .action(ArgAction::Set)
        ).get_matches();

    let num_clusters = *matches.get_one::<usize>("n_clusters").unwrap();
    let num_superclusters = *matches.get_one::<usize>("n_superclusters").unwrap();
    let output = matches.get_one::<String>("output").unwrap();
    let seed = *matches.get_one::<u64>("seed").unwrap();
    let center_radius = *matches.get_one::<f32>("center_radius").unwrap();
    let local_sigma = *matches.get_one::<f32>("local_sigma").unwrap();
    let even = matches.get_one::<bool>("even").unwrap();
    let supercluster_sigma = *matches.get_one::<f32>("supercluster_sigma").unwrap();

    let cfg: Config;
    if let Some(skew) = matches.get_one::<f32>("skew") {
        // skew is a general measure of skewness changing both center radius and local sigma at the same time
        cfg = Config::new_x_skew(*skew, num_clusters, num_superclusters, seed, output);
    }
    else {
        // if skew is not provided, assign values for local sigma and center radius
        cfg = Config::new(num_clusters, num_superclusters, seed, output, center_radius, local_sigma, supercluster_sigma);
    }
    

    println!("Generating skewed Euclidean dataset");
    println!("  n_train            = {}", cfg.n_train);
    println!("  n_test             = {}", cfg.n_test);
    println!("  dim                = {}", cfg.dim);
    println!("  n_clusters         = {}", cfg.n_clusters);
    println!("  n_superclusters    = {}", cfg.n_superclusters);
    println!("  gt_k               = {}", cfg.gt_k);
    println!("  center_radius      = {}", cfg.center_radius);
    println!("  local_sigma        = {}", cfg.local_sigma);
    println!("  supercluster_sigma = {}", cfg.supercluster_sigma);
    println!("  output             = {}", cfg.output);

    let t0 = Instant::now();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let centers = build_cluster_centers(&cfg, &mut rng);

    let train_centers = centers[0..centers.len() -1].to_vec(); // use the last cluster for testing
    let test_center = vec![centers[centers.len() - 1].clone()];
    let (train, train_labels) = train_dataset(cfg.n_train, &cfg, &train_centers, &mut rng, even);
    let (test, _test_labels) = test_dataset(cfg.n_test, &cfg, &test_center, &mut rng, even);

    println!("Sampling done in {:?}", t0.elapsed());

    let t1 = Instant::now();
    let (neighbors, distances) = topk_truth(&train, &test, cfg.gt_k);
    println!("Ground truth done in {:?}", t1.elapsed());

    //summarize_pair_distribution(&train, &train_labels, &mut rng, 200_000);

    println!(
        "\nGenerated file should behave like: many very-near intra-cluster pairs,\n\
         many very-far inter-cluster pairs, and relatively fewer medium-distance pairs.\n\
         Increase `center_radius` or decrease `local_sigma` to make it more extreme."
    );

    let t2 = Instant::now();
    write_hdf5(cfg.output, &train, &test, &neighbors, &distances)?;
    println!("HDF5 write done in {:?}", t2.elapsed());
    println!("Total done in {:?}", t0.elapsed());
    println!("Wrote {}", output);

    Ok(())
}
