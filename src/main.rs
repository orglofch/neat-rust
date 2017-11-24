pub mod activation;
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
    let genome_config = GenomeConfig::new(3, 1);

    let mut config = Config::new(fitness_config, genome_config);
    config.run();
}
