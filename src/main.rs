pub mod activation;
pub mod aggregation;
pub mod config;
pub mod checkpoint;
pub mod fitness;
pub mod genome;
pub mod innovation;
pub mod speciation;

use config::Config;
use fitness::FitnessConfig;
use genome::{Genome, GenomeConfig, Population};

use std::collections::HashMap;

fn eval_fitness(genomes: &Population) -> HashMap<u32, f32> {
    return HashMap::new();
}

fn main() {
    let fitness_config = FitnessConfig::new(eval_fitness);

    let genome_config = GenomeConfig::new(Vec::new(), Vec::new());

    let mut config = Config::new(fitness_config, genome_config);
    config.run();
}
